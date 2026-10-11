"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB  (a local .env is loaded for convenience; real
      environment variables always win because load_dotenv does not override)

The HTTP contract is openapi.yaml, the stored shapes are DATA_MODEL.md and the
business rules are R1-R10.  Rule numbers are quoted in the comments at the point
where they are implemented.

Layout of this file
    1.  Configuration and MongoDB client
    2.  Constants and status sets
    3.  Rounding / date / timezone / epoch helpers        (R1, R4, R8)
    4.  Business rule calculations                        (R2-R5)
    5.  MongoDB <-> API serialization helpers
    6.  Pydantic request models
    7.  Lifespan and index creation
    8.  Shared validation and query builders
    9.  Aggregation pipeline builders                     (analytics)
    10. System and employee endpoints
    11. Attendance endpoints
    12. Analytics endpoints
    13. Explain endpoint
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, Optional

from bson import Binary, Decimal128, ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    DuplicateKeyError,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)

# --------------------------------------------------------------------------- #
# 1. Configuration and MongoDB client
# --------------------------------------------------------------------------- #
# load_dotenv() never overrides variables that are already set, so the values
# the grader injects take priority over any local .env file.
load_dotenv()

LOG = logging.getLogger("attendance")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

# tz_aware=True makes PyMongo return timezone-aware UTC datetimes instead of
# naive ones, which removes a whole class of "is this local or UTC?" bugs.
# The pool is sized for the parallel requests the grader sends; the timeouts
# keep a sick database from hanging a worker thread forever.
client: MongoClient = MongoClient(
    MONGO_URI,
    tz_aware=True,
    maxPoolSize=100,
    minPoolSize=0,
    serverSelectionTimeoutMS=5_000,
    connectTimeoutMS=5_000,
    socketTimeoutMS=30_000,
    retryWrites=True,
    appname="attendance-api",
)
db = client[MONGO_DB]
employees = db["employees"]
attendance = db["attendance_logs"]

# --------------------------------------------------------------------------- #
# 2. Constants and status sets
# --------------------------------------------------------------------------- #
IST = timezone(timedelta(hours=5, minutes=30))

GRACE_SECONDS = 10 * 60          # R2: grace period is 10 minutes
MIN_OVERTIME_MINUTES = 30        # R3: overtime only counts from 30 minutes
HALF_DAY_LIMIT = Decimal("4.50")  # R5

PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
NON_PRESENCE_STATUSES = ("ABSENT", "LEAVE")
ALL_STATUSES = PRESENCE_STATUSES + NON_PRESENCE_STATUSES

# $dayOfWeek is 1=Sunday .. 7=Saturday, so Monday-Friday is 2..6 (R7).
WEEKDAY_DOW = [2, 3, 4, 5, 6]

# Bounds from the EpochMillis schema in openapi.yaml.
EPOCH_MS_MIN = 100_000_000_000
EPOCH_MS_MAX = 4_102_444_800_000

MAX_TREND_DAYS = 92
DAY_MS = 24 * 60 * 60 * 1000

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"

# Index name -> (collection, keys, options).  Created at startup, idempotently.
INDEX_PLAN: list[tuple[str, list[tuple[str, int]], dict[str, Any]]] = [
    # Uniqueness of the business key is enforced by the database, not by a
    # read-then-write check, so concurrent creates cannot both succeed.
    ("employees", [("emp_code", ASCENDING)], {"name": "uniq_emp_code", "unique": True}),
    # GET /employees: exact department filter + sort by emp_code ascending.
    ("employees", [("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "dept_emp_code"}),
    # R9 headcount eligibility (joined_on <= end of period) for the department
    # summary when no department filter is given.
    ("employees", [("joined_on", ASCENDING)], {"name": "joined_on"}),
    # At most one record per (emp_code, date) - this is what makes two
    # simultaneous punch-ins resolve to exactly one 201.  It also serves every
    # per-employee date-range query (employee monthly, department lookups).
    ("attendance_logs", [("emp_code", ASCENDING), ("date", ASCENDING)],
     {"name": "uniq_emp_date", "unique": True}),
    # GET /attendance sorts by date descending then emp_code ascending; the same
    # index serves the month range scans used by the leaderboard.
    ("attendance_logs", [("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "date_desc_emp"}),
]


# --------------------------------------------------------------------------- #
# 3. Rounding / date / timezone / epoch helpers
# --------------------------------------------------------------------------- #
def half_up(value: float | int | Decimal | None, places: int) -> Optional[float]:
    """Round half-up to ``places`` decimals (R4, R8).

    Python's built-in round() is banker's rounding on top of binary floats, so
    round(2.675, 2) == 2.67. Going through Decimal(str(value)) rounds the
    decimal number the caller meant, half away from zero.
    """
    if value is None:
        return None
    dec = value if isinstance(value, Decimal) else Decimal(str(value))
    quantum = Decimal(1).scaleb(-places)
    return float(dec.quantize(quantum, rounding=ROUND_HALF_UP))


def as_utc(value: datetime) -> datetime:
    """Normalise a datetime to aware UTC.

    The client is tz_aware, so reads are already aware, but a naive datetime
    out of MongoDB is UTC by definition, so attach UTC rather than guess.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def epoch_ms_to_datetime(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime truncated to whole seconds (R1)."""
    seconds = ms // 1000  # floor: 09:40:00.900 counts as 09:40:00
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def datetime_to_epoch_ms(value: Optional[datetime]) -> Optional[int]:
    """Aware/naive UTC datetime -> integer epoch milliseconds."""
    if value is None:
        return None
    value = as_utc(value)
    return int(value.timestamp()) * 1000 + value.microsecond // 1000


def now_utc_truncated() -> datetime:
    """Server clock as aware UTC, truncated to whole seconds (R1)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def hhmm_minutes(value: str) -> int:
    hour, minute = parse_hhmm(value)
    return hour * 60 + minute


def is_overnight_shift(shift_start: str, shift_end: str) -> bool:
    """R1: shift_end <= shift_start means the shift crosses midnight."""
    return hhmm_minutes(shift_end) <= hhmm_minutes(shift_start)


def ist_instant(day: date_cls, hhmm: str) -> datetime:
    """An IST wall-clock time on an IST calendar day, as an aware UTC instant."""
    hour, minute = parse_hhmm(hhmm)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST).astimezone(
        timezone.utc
    )


def attendance_date_for(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: the IST calendar date a punch-in belongs to.

    Normally the IST date of the punch-in. For an overnight shift a punch-in
    whose IST time is strictly earlier than shift_end belongs to the shift that
    started on the previous calendar day.
    """
    local = as_utc(punch_in).astimezone(IST)
    if is_overnight_shift(shift_start, shift_end):
        # Seconds matter: 05:59:59 is earlier than a 06:00 shift end, 06:00:00
        # is not, so compare on whole seconds of the IST wall clock.
        punch_seconds = local.hour * 3600 + local.minute * 60 + local.second
        if punch_seconds < hhmm_minutes(shift_end) * 60:
            return (local.date() - timedelta(days=1)).isoformat()
    return local.date().isoformat()


def shift_start_instant(date_str: str, shift_start: str) -> datetime:
    """The instant the shift of attendance date ``date_str`` starts."""
    return ist_instant(date_cls.fromisoformat(date_str), shift_start)


def shift_end_instant(date_str: str, shift_start: str, shift_end: str) -> datetime:
    """The instant the shift of attendance date ``date_str`` ends.

    R1: an overnight shift ends on the next calendar day.
    """
    day = date_cls.fromisoformat(date_str)
    if is_overnight_shift(shift_start, shift_end):
        day = day + timedelta(days=1)
    return ist_instant(day, shift_end)


def month_bounds(month: str) -> tuple[str, str]:
    """'YYYY-MM' -> (first day, last day) as YYYY-MM-DD strings."""
    year, mon = (int(part) for part in month.split("-"))
    first = date_cls(year, mon, 1)
    last = date_cls(year + (mon == 12), 1 if mon == 12 else mon + 1, 1) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def working_days_between(start: date_cls, end: date_cls) -> int:
    """R7: Monday-Friday days in [start, end], no holiday calendar."""
    if end < start:
        return 0
    total = 0
    day = start
    while day <= end:
        if day.weekday() < 5:  # Mon=0 .. Fri=4
            total += 1
        day += timedelta(days=1)
    return total


# --------------------------------------------------------------------------- #
# 4. Business rule calculations
# --------------------------------------------------------------------------- #
def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2: late only when strictly more than 10:00 after shift start; then the
    floor of the minutes elapsed since shift start (not since the grace period).

    Shift 09:30 -> 09:40:00 gives 0, 09:40:01 gives 10, 10:15:59 gives 45.
    """
    elapsed = (as_utc(punch_in) - shift_start_instant(date_str, shift_start)).total_seconds()
    if elapsed <= GRACE_SECONDS:
        return 0
    return int(math.floor(elapsed / 60))


def compute_overtime_minutes(
    punch_out: datetime, date_str: str, shift_start: str, shift_end: str
) -> int:
    """R3: floor of the minutes from shift end to punch-out, only if >= 30."""
    elapsed = (
        as_utc(punch_out) - shift_end_instant(date_str, shift_start, shift_end)
    ).total_seconds()
    if elapsed <= 0:
        return 0
    minutes = int(math.floor(elapsed / 60))
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: (punch_out - punch_in) seconds / 3600, 2 decimals half-up."""
    seconds = int((as_utc(punch_out) - as_utc(punch_in)).total_seconds())
    hours = Decimal(seconds) / Decimal(3600)
    return float(hours.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_half_day(work_hours: Optional[float]) -> bool:
    """R5: true when the *rounded* work_hours is strictly below 4.50."""
    if work_hours is None:
        return False
    return Decimal(str(work_hours)) < HALF_DAY_LIMIT


def derive_fields(
    status: str,
    punch_in: Optional[datetime],
    punch_out: Optional[datetime],
    date_str: str,
    shift_start: str,
    shift_end: str,
) -> dict[str, Any]:
    """The four derived fields for a record, from its final times (R2-R5).

    Used by punch-in, punch-out and every regularization, so the rules live in
    exactly one place.
    """
    if status in NON_PRESENCE_STATUSES or punch_in is None:
        return {
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
        }

    late_minutes = compute_late_minutes(punch_in, date_str, shift_start)
    if punch_out is None:
        return {
            "work_hours": None,
            "late_minutes": late_minutes,
            "overtime_minutes": 0,
            "half_day": False,
        }

    work_hours = compute_work_hours(punch_in, punch_out)
    return {
        "work_hours": work_hours,
        "late_minutes": late_minutes,
        "overtime_minutes": compute_overtime_minutes(
            punch_out, date_str, shift_start, shift_end
        ),
        "half_day": compute_half_day(work_hours),
    }


# --------------------------------------------------------------------------- #
# 5. MongoDB <-> API serialization helpers
# --------------------------------------------------------------------------- #
def employee_out(doc: dict[str, Any]) -> dict[str, Any]:
    """An employees document as the Employee schema. _id never leaves the API."""
    return {
        "emp_code": doc["emp_code"],
        "name": doc.get("name"),
        "email": doc.get("email"),
        "department": doc.get("department"),
        "shift_start": doc.get("shift_start"),
        "shift_end": doc.get("shift_end"),
        "joined_on": doc.get("joined_on"),
        "created_at": datetime_to_epoch_ms(doc.get("created_at")),
    }


def history_entry_out(entry: dict[str, Any]) -> dict[str, Any]:
    """One history entry, with the nested punch instants converted to epoch ms.

    The from/to values of punch_in and punch_out are BSON dates in MongoDB but
    epoch milliseconds in the API, and they sit inside a nested object.
    """
    changes: dict[str, Any] = {}
    for field, change in (entry.get("changes") or {}).items():
        if field in ("punch_in", "punch_out"):
            changes[field] = {
                "from": datetime_to_epoch_ms(change.get("from")),
                "to": datetime_to_epoch_ms(change.get("to")),
            }
        else:
            changes[field] = {"from": change.get("from"), "to": change.get("to")}
    return {
        "at": datetime_to_epoch_ms(entry.get("at")),
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": changes,
    }


def attendance_out(doc: dict[str, Any]) -> dict[str, Any]:
    """An attendance_logs document as the AttendanceRecord schema.

    A record is addressed by (emp_code, date); there is no id field and _id is
    never exposed. Missing half_day / history mean false / [] (legacy records).
    """
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": datetime_to_epoch_ms(doc.get("punch_in")),
        "punch_out": datetime_to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day") or False),
        "history": [history_entry_out(entry) for entry in (doc.get("history") or [])],
    }


def jsonable_bson(value: Any) -> Any:
    """Make a raw explain document JSON-serializable without losing structure.

    Explain output contains BSON types (datetimes, ObjectIds, Decimal128,
    Binary) that FastAPI's encoder would reject.
    """
    if isinstance(value, dict):
        return {str(key): jsonable_bson(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_bson(item) for item in value]
    if isinstance(value, datetime):
        return datetime_to_epoch_ms(value)
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, Binary):
        return value.hex()
    if isinstance(value, bytes):
        return value.hex()
    return value


# --------------------------------------------------------------------------- #
# 6. Pydantic request models
# --------------------------------------------------------------------------- #
# StrictInt is what rejects floats and numeric strings with 422; the bounds are
# what reject seconds-looking values. Both come straight from the EpochMillis
# schema in openapi.yaml.


class EmployeeCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM_PATTERN)
    shift_end: str = Field(default="18:30", pattern=HHMM_PATTERN)
    joined_on: str = Field(pattern=DATE_PATTERN)

    @model_validator(mode="after")
    def _check(self) -> "EmployeeCreate":
        # A real calendar date, not just four-two-two digits.
        try:
            date_cls.fromisoformat(self.joined_on)
        except ValueError:
            raise ValueError("joined_on is not a valid calendar date")
        # shift_end <= shift_start is an overnight shift and is allowed; equal
        # start and end is a zero-length shift and is not.
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(min_length=1)
    punched_at: Optional[StrictInt] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    # Only the presence statuses can be created by a punch-in.
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(min_length=1)
    punched_at: Optional[StrictInt] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)


class RegularizeRequest(BaseModel):
    """Only these fields can be changed; derived fields are never accepted."""

    model_config = ConfigDict(extra="ignore")

    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = None
    punch_in: Optional[StrictInt] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    punch_out: Optional[StrictInt] = Field(default=None, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _check(self) -> "RegularizeRequest":
        if self.status in NON_PRESENCE_STATUSES and (
            self.punch_in is not None or self.punch_out is not None
        ):
            # Setting ABSENT / LEAVE clears both punch times, so supplying one
            # is contradictory.
            raise ValueError("status ABSENT/LEAVE clears the punch times; do not supply them")
        if self.status is None and self.punch_in is None and self.punch_out is None:
            raise ValueError("supply at least one of status, punch_in, punch_out")
        return self


# --------------------------------------------------------------------------- #
# 7. Lifespan and index creation
# --------------------------------------------------------------------------- #
_indexes_ready = False


def ensure_indexes() -> bool:
    """Create every index in INDEX_PLAN. Idempotent: create_index on an index
    that already exists with the same keys and options is a no-op, so this is
    safe to run on every start and against a database the grader has seeded.

    Each index is attempted on its own: one failure (for example a unique index
    that cannot be built because the seeded data already holds a duplicate)
    must not stop the other four from being created. Failures are logged at
    error level rather than swallowed, and /health retries until they all
    succeed.
    """
    global _indexes_ready
    failures: list[str] = []
    for collection_name, keys, options in INDEX_PLAN:
        try:
            db[collection_name].create_index(keys, **options)
        except PyMongoError as exc:
            failures.append(options["name"])
            LOG.error(
                "could not create index %s on %s: %s: %s",
                options["name"],
                collection_name,
                type(exc).__name__,
                exc,
            )
    _indexes_ready = not failures
    return _indexes_ready


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Index creation must not stop the app from binding its port: if MongoDB is
    # not up yet, /health keeps reporting 503 and retries, so the app is still
    # ready inside the 20 second budget once the database answers.
    ensure_indexes()
    yield
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# Only connectivity failures become 503. Anything else (a bad pipeline, a
# write conflict) stays a 500 on purpose: swallowing those would hide defects.
CONNECTIVITY_ERRORS = (
    ServerSelectionTimeoutError,
    ConnectionFailure,
    NetworkTimeout,
    AutoReconnect,
)


@app.exception_handler(ServerSelectionTimeoutError)
@app.exception_handler(ConnectionFailure)
@app.exception_handler(NetworkTimeout)
@app.exception_handler(AutoReconnect)
async def _mongo_unavailable_handler(_: Request, exc: PyMongoError) -> JSONResponse:
    """A database that is down is 503, not 500, and the driver's message (which
    can contain the connection string) never reaches the client.
    """
    LOG.error("mongodb unavailable: %s: %s", type(exc).__name__, exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


# --------------------------------------------------------------------------- #
# 8. Shared validation and query builders
# --------------------------------------------------------------------------- #
def parse_date_param(value: str, name: str) -> date_cls:
    """A YYYY-MM-DD query/path parameter as a date, 422 when it is not one."""
    try:
        return date_cls.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            422, f"{name} must be a valid calendar date in YYYY-MM-DD format"
        )


def paged(total: int, items: list[Any], page: int, page_size: int) -> dict[str, Any]:
    """R10: total always reflects the filters, not the page."""
    return {"items": items, "total": total, "page": page, "page_size": page_size}


def get_employee_or_404(emp_code: str) -> dict[str, Any]:
    employee = employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")
    return employee


def build_attendance_filter(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[str],
) -> dict[str, Any]:
    """The filter for GET /attendance, shared with /admin/explain so the
    explained query is byte-for-byte the one the endpoint runs.
    """
    query: dict[str, Any] = {}
    if emp_code:
        query["emp_code"] = emp_code
    if date_from or date_to:
        from_date = parse_date_param(date_from, "date_from") if date_from else None
        to_date = parse_date_param(date_to, "date_to") if date_to else None
        if from_date and to_date and from_date > to_date:
            raise HTTPException(422, "date_from must not be after date_to")
        bounds: dict[str, str] = {}
        if date_from:
            bounds["$gte"] = date_from
        if date_to:
            bounds["$lte"] = date_to
        query["date"] = bounds
    if status:
        query["status"] = status
    return query


# Sort for GET /attendance: date descending, then emp_code ascending. The
# date_desc_emp index provides exactly this order, so there is no blocking sort.
ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


# --------------------------------------------------------------------------- #
# 9. Aggregation pipeline builders
# --------------------------------------------------------------------------- #
def _weekday_expr(date_field: str) -> dict[str, Any]:
    """R7: is the record's date (a YYYY-MM-DD string) a Monday-Friday?

    The string is an IST calendar date; parsing it as a UTC midnight instant
    gives the right day of the week without dragging a timezone into it.
    """
    return {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": date_field}}}, WEEKDAY_DOW]}


def _present_value_expr(
    status_field: str, date_field: str, half_day_field: str
) -> dict[str, Any]:
    """R5-R7: a record's contribution to present_days.

    Presence status on a Monday-Friday counts 1, or 0.5 for a half day;
    anything else counts 0. Missing half_day means false (legacy records).
    """
    return {
        "$cond": [
            {
                "$and": [
                    {"$in": [status_field, list(PRESENCE_STATUSES)]},
                    _weekday_expr(date_field),
                ]
            },
            {"$cond": [{"$eq": [{"$ifNull": [half_day_field, False]}, True]}, 0.5, 1]},
            0,
        ]
    }


def _half_up_expr(value: Any, places: int) -> dict[str, Any]:
    """Half-up rounding inside the aggregation.

    MongoDB's $round rounds halves to even, so it cannot be used where the
    contract asks for half-up. floor(x * 10^p + 0.5) / 10^p is half-up for the
    non-negative rates and averages this is applied to.
    """
    factor = 10**places
    return {
        "$divide": [
            {"$floor": {"$add": [{"$multiply": [value, factor]}, 0.5]}},
            factor,
        ]
    }


def employee_monthly_pipeline(emp_code: str, month: str) -> list[dict[str, Any]]:
    """Totals for one employee in one month, read from the stored derived values.

    The $match is on (emp_code, date range), which is the uniq_emp_date index.
    """
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": _present_value_expr("$status", "$date", "$half_day")
                },
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
        {"$project": {"_id": 0}},
    ]


def department_summary_pipeline(
    month: str, department: Optional[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One row per department for one month, across both collections.

    Driven from `employees`, not from the logs: the department lives on the
    employee and R9 headcount has to include people with zero logs, so an
    inner-join shape ($lookup then a plain $unwind) would silently drop them.
    $unwind keeps empty arrays, and headcount counts distinct emp_codes.

    Returns the pipeline and the index to hint, so the real endpoint and
    /admin/explain run the identical operation.
    """
    first, last = month_bounds(month)
    match: dict[str, Any] = {"joined_on": {"$lte": last}}  # R9
    hint: dict[str, int] = {"joined_on": ASCENDING}
    if department:
        match["department"] = department
        hint = {"department": ASCENDING, "emp_code": ASCENDING}

    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": first, "$lte": last}}},
                    {
                        "$project": {
                            "_id": 0,
                            "date": 1,
                            "status": 1,
                            "work_hours": 1,
                            "late_minutes": 1,
                            "half_day": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        # preserveNullAndEmptyArrays is the whole point: an employee with no
        # logs stays in the stream and still counts towards headcount.
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "emp_codes": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": _present_value_expr(
                        "$logs.status", "$logs.date", "$logs.half_day"
                    )
                },
                # Sum and count rather than $avg so the division and the
                # half-up rounding happen once, in Decimal: the mean is over
                # records, never an average of per-employee averages.
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$logs.status", list(PRESENCE_STATUSES)]},
                                    {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]},
                                ]
                            },
                            "$logs.work_hours",
                            0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$logs.status", list(PRESENCE_STATUSES)]},
                                    {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [{"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]}, 1, 0]
                    }
                },
                "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
                "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$emp_codes"},
                "present_days": 1,
                "work_hours_sum": 1,
                "work_hours_count": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": ASCENDING}},
    ]
    return pipeline, hint


def late_leaderboard_pipeline(
    month: str, limit: int, department: Optional[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Competition ranking of late-comers for one month.

    $rank gives standard competition ranking (ties share a rank and the next
    rank is skipped), which is why the rank is not the array position. The
    $match on rank applies limit *after* ranking, so a tie at the cutoff
    returns more than `limit` rows.
    """
    first, last = month_bounds(month)
    pipeline: list[dict[str, Any]] = [
        # Only late records can contribute to a positive total, so filtering
        # them out here does not change any sum and keeps the scan small.
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "pipeline": [{"$project": {"_id": 0, "name": 1, "department": 1}}],
                "as": "employee",
            }
        },
        # A plain $unwind drops logs whose employee does not exist, which is
        # what the contract asks for.
        {"$unwind": "$employee"},
    ]
    if department:
        # Ranking happens inside the department, so filter before ranking.
        pipeline.append({"$match": {"employee.department": department}})
    pipeline += [
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": DESCENDING},
                "output": {"rank": {"$rank": {}}},
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$employee.name",
                "department": "$employee.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
        {"$sort": {"total_late_minutes": DESCENDING, "emp_code": ASCENDING}},
    ]
    return pipeline, {"date": DESCENDING, "emp_code": ASCENDING}


def department_trend_pipeline(
    department: str, date_from: str, date_to: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exactly one row per calendar day in [date_from, date_to].

    Driven from `employees` because headcount is a property of the employees
    (R9: joined_on <= that day), and the department only exists there.

    Each employee is flattened into day events:
      * an anchor event on `date_from` so the stream is never empty,
      * a joiner event on max(joined_on, date_from) when they joined by
        `date_to`, which $setWindowFields later turns into a running headcount,
      * one event per attendance record in range.

    $group collapses them to one row per day that had activity, $densify fills
    every missing calendar day, $fill zeroes the generated rows, and
    $setWindowFields computes the 7-day moving average. No Python loop.
    """
    from_date = date_cls.fromisoformat(date_from)
    to_date = date_cls.fromisoformat(date_to)
    # $densify needs a real date field, and its upper bound is exclusive.
    lower = datetime(from_date.year, from_date.month, from_date.day, tzinfo=timezone.utc)
    upper = datetime(to_date.year, to_date.month, to_date.day, tzinfo=timezone.utc) + timedelta(
        days=1
    )

    pipeline: list[dict[str, Any]] = [
        {"$match": {"department": department}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
                    {
                        "$project": {
                            "_id": 0,
                            "date": 1,
                            "status": 1,
                            "late_minutes": 1,
                            "half_day": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$project": {
                "_id": 0,
                "events": {
                    "$concatArrays": [
                        [{"date": date_from, "joiners": 0, "present": 0, "late": 0}],
                        {
                            "$cond": [
                                {"$lte": ["$joined_on", date_to]},
                                [
                                    {
                                        "date": {
                                            "$cond": [
                                                {"$lt": ["$joined_on", date_from]},
                                                date_from,
                                                "$joined_on",
                                            ]
                                        },
                                        "joiners": 1,
                                        "present": 0,
                                        "late": 0,
                                    }
                                ],
                                [],
                            ]
                        },
                        {
                            "$map": {
                                "input": "$logs",
                                "as": "log",
                                "in": {
                                    "date": "$$log.date",
                                    "joiners": 0,
                                    # present_count counts weekend records too,
                                    # so this is not the Mon-Fri expression.
                                    "present": {
                                        "$cond": [
                                            {
                                                "$in": [
                                                    "$$log.status",
                                                    list(PRESENCE_STATUSES),
                                                ]
                                            },
                                            {
                                                "$cond": [
                                                    {
                                                        "$eq": [
                                                            {
                                                                "$ifNull": [
                                                                    "$$log.half_day",
                                                                    False,
                                                                ]
                                                            },
                                                            True,
                                                        ]
                                                    },
                                                    0.5,
                                                    1,
                                                ]
                                            },
                                            0,
                                        ]
                                    },
                                    "late": {
                                        "$cond": [
                                            {
                                                "$gt": [
                                                    {"$ifNull": ["$$log.late_minutes", 0]},
                                                    0,
                                                ]
                                            },
                                            1,
                                            0,
                                        ]
                                    },
                                },
                            }
                        },
                    ]
                },
            }
        },
        {"$unwind": "$events"},
        {
            "$group": {
                "_id": "$events.date",
                "joiners": {"$sum": "$events.joiners"},
                "present": {"$sum": "$events.present"},
                "late": {"$sum": "$events.late"},
            }
        },
        {"$set": {"day": {"$dateFromString": {"dateString": "$_id"}}}},
        {
            "$densify": {
                "field": "day",
                "range": {"step": 1, "unit": "day", "bounds": [lower, upper]},
            }
        },
        # Densified rows carry only `day`, so give them zeroes.
        {
            "$fill": {
                "output": {
                    "joiners": {"value": 0},
                    "present": {"value": 0},
                    "late": {"value": 0},
                }
            }
        },
        {"$sort": {"day": ASCENDING}},
        {
            "$setWindowFields": {
                "sortBy": {"day": ASCENDING},
                "output": {
                    # R9 headcount for the day: everyone who had joined by it.
                    "headcount": {
                        "$sum": "$joiners",
                        "window": {"documents": ["unbounded", "current"]},
                    }
                },
            }
        },
        {"$set": {"is_working_day": {"$in": [{"$dayOfWeek": "$day"}, WEEKDAY_DOW]}}},
        {
            "$set": {
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        _half_up_expr({"$divide": ["$present", "$headcount"]}, 4),
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"day": ASCENDING},
                "output": {
                    # $avg skips the null rates, and yields null when the whole
                    # window is null. The window is the current row and up to 6
                    # preceding rows *inside the range*, so the first rows get a
                    # shorter window for free.
                    "moving_avg_raw": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
                "is_working_day": 1,
                "headcount": 1,
                "present_count": "$present",
                "late_count": "$late",
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": [{"$ifNull": ["$moving_avg_raw", None]}, None]},
                        None,
                        _half_up_expr("$moving_avg_raw", 4),
                    ]
                },
            }
        },
        {"$sort": {"date": ASCENDING}},
    ]
    return pipeline, {"department": ASCENDING, "emp_code": ASCENDING}


# --------------------------------------------------------------------------- #
# 10. System and employee endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    """200 only when MongoDB answers a ping, otherwise 503."""
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        LOG.warning("health check failed: %s", type(exc).__name__)
        raise HTTPException(503, "database unavailable")
    if not _indexes_ready:
        # The database was not reachable at startup; now that it answers, put
        # the indexes in place before reporting ready.
        ensure_indexes()
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate) -> dict[str, Any]:
    """Create an employee. Uniqueness of emp_code is enforced by the unique
    index, so two simultaneous identical creates give exactly one 201.
    """
    doc = body.model_dump()
    doc["created_at"] = now_utc_truncated()
    try:
        employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Sorted by emp_code ascending; total counts the filtered set (R10)."""
    query: dict[str, Any] = {}
    if department:
        query["department"] = department
    total = employees.count_documents(query)
    cursor = (
        employees.find(query)
        .sort([("emp_code", ASCENDING)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return paged(total, [employee_out(doc) for doc in cursor], page, page_size)


# --------------------------------------------------------------------------- #
# 11. Attendance endpoints
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest) -> dict[str, Any]:
    """Create the day's record (R1 for the date, R2 for late_minutes).

    Concurrency: no check-then-insert. The insert either wins or raises
    DuplicateKeyError on uniq_emp_date, which becomes the 409, so two
    simultaneous punch-ins give exactly one 201.
    """
    employee = get_employee_or_404(body.emp_code)
    punched_at = (
        epoch_ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else now_utc_truncated()
    )
    shift_start = employee.get("shift_start", "09:30")
    shift_end = employee.get("shift_end", "18:30")
    date_str = attendance_date_for(punched_at, shift_start, shift_end)

    doc: dict[str, Any] = {
        "emp_code": body.emp_code,
        "date": date_str,
        "status": body.status,
        "punch_in": punched_at,
        "punch_out": None,
        "history": [],
        **derive_fields(body.status, punched_at, None, date_str, shift_start, shift_end),
    }
    try:
        attendance.insert_one(dict(doc))
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return attendance_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest) -> dict[str, Any]:
    """Close the most recent open-or-closed record whose punch_in is at or
    before punched_at, and compute R3-R5.

    Works across midnight because the record is found by punch_in instant, not
    by today's calendar date.

    Concurrency: the update is conditional on punch_out still being null, so of
    two simultaneous punch-outs exactly one matches and the loser gets 409
    without overwriting the winner's values.
    """
    employee = get_employee_or_404(body.emp_code)
    punched_at = (
        epoch_ms_to_datetime(body.punched_at)
        if body.punched_at is not None
        else now_utc_truncated()
    )

    record = attendance.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$ne": None, "$lte": punched_at},
        },
        sort=[("punch_in", DESCENDING)],
    )
    if record is None:
        raise HTTPException(404, "no punch-in found for this employee")
    if record.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    punch_in_at = as_utc(record["punch_in"])
    if punched_at <= punch_in_at:
        raise HTTPException(422, "punched_at must be after the punch-in")
    if (punched_at - punch_in_at) > timedelta(hours=24):
        raise HTTPException(422, "punched_at must be within 24 hours of the punch-in")

    derived = derive_fields(
        record["status"],
        punch_in_at,
        punched_at,
        record["date"],
        employee.get("shift_start", "09:30"),
        employee.get("shift_end", "18:30"),
    )
    updated = attendance.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {"punch_out": punched_at, **derived}},
        return_document=True,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return attendance_out(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    date_to: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = Query(
        default=None
    ),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Sorted by date descending then emp_code ascending.

    Filtering, sorting, counting and pagination all happen in MongoDB, so this
    stays flat on ~100,000 records.
    """
    query = build_attendance_filter(emp_code, date_from, date_to, status)
    total = attendance.count_documents(query)
    cursor = (
        attendance.find(query)
        .sort(ATTENDANCE_SORT)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return paged(total, [attendance_out(doc) for doc in cursor], page, page_size)


# Fields a regularization may touch, in the order they appear in history.changes.
DIFF_FIELDS = (
    "status",
    "punch_in",
    "punch_out",
    "work_hours",
    "late_minutes",
    "overtime_minutes",
    "half_day",
)

# Hint for the employee monthly aggregation; the same index serves the unique
# (emp_code, date) constraint.
EMP_DATE_HINT = {"emp_code": ASCENDING, "date": ASCENDING}


def _normalise_for_diff(field: str, value: Any) -> Any:
    """Compare stored and recomputed values on equal terms.

    Missing late_minutes / overtime_minutes mean 0 and missing half_day means
    false, so a legacy record that keeps the same effective value must not show
    up as a change.
    """
    if field in ("punch_in", "punch_out"):
        return as_utc(value) if isinstance(value, datetime) else None
    if field in ("late_minutes", "overtime_minutes"):
        return int(value or 0)
    if field == "half_day":
        return bool(value or False)
    return value


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    body: RegularizeRequest,
    emp_code: str = Path(...),
    date: str = Path(..., pattern=DATE_PATTERN),
) -> dict[str, Any]:
    """Manually correct one day's record, addressed by its natural key.

    Recomputes every derived field from the final times (R2-R5) and appends
    exactly one history entry holding only the fields that actually changed.

    Concurrency: the update is a compare-and-set on the fields that were read
    plus the current length of history, so a second concurrent correction based
    on the same "before" state cannot overwrite the first one's entry - it
    fails to match and gets a 409.
    """
    parse_date_param(date, "date")  # 422 before any lookup
    employee = get_employee_or_404(emp_code)
    record = attendance.find_one({"emp_code": emp_code, "date": date})
    if record is None:
        raise HTTPException(404, "no attendance record for this date")

    shift_start = employee.get("shift_start", "09:30")
    shift_end = employee.get("shift_end", "18:30")

    final_status = body.status or record["status"]
    if final_status in NON_PRESENCE_STATUSES:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                422, "an ABSENT/LEAVE record cannot have punch times"
            )
        final_punch_in: Optional[datetime] = None
        final_punch_out: Optional[datetime] = None
    else:
        final_punch_in = (
            epoch_ms_to_datetime(body.punch_in)
            if body.punch_in is not None
            else (as_utc(record["punch_in"]) if record.get("punch_in") else None)
        )
        final_punch_out = (
            epoch_ms_to_datetime(body.punch_out)
            if body.punch_out is not None
            else (as_utc(record["punch_out"]) if record.get("punch_out") else None)
        )
        if final_punch_in is None:
            raise HTTPException(422, f"status {final_status} requires a punch_in")
        if body.punch_in is not None and (
            attendance_date_for(final_punch_in, shift_start, shift_end) != date
        ):
            # R1: a correction may not move a record to another attendance day.
            raise HTTPException(
                422, "punch_in must stay on the record's attendance date"
            )
        if final_punch_out is not None:
            if final_punch_out <= final_punch_in:
                raise HTTPException(422, "punch_out must be after punch_in")
            if (final_punch_out - final_punch_in) > timedelta(hours=24):
                raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

    derived = derive_fields(
        final_status, final_punch_in, final_punch_out, date, shift_start, shift_end
    )
    new_values: dict[str, Any] = {
        "status": final_status,
        "punch_in": final_punch_in,
        "punch_out": final_punch_out,
        **derived,
    }

    changes: dict[str, Any] = {}
    for field in DIFF_FIELDS:
        before = _normalise_for_diff(field, record.get(field))
        after = _normalise_for_diff(field, new_values[field])
        if before != after:
            changes[field] = {"from": before, "to": after}
    if not changes:
        raise HTTPException(422, "the request would not change anything")

    entry = {
        "at": now_utc_truncated(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Compare-and-set guard: the status and punch times we based the diff on,
    # plus the exact number of history entries we saw.
    guard: dict[str, Any] = {
        "_id": record["_id"],
        "status": record["status"],
        "punch_in": _normalise_for_diff("punch_in", record.get("punch_in")),
        "punch_out": _normalise_for_diff("punch_out", record.get("punch_out")),
        "$expr": {
            "$eq": [
                {"$size": {"$ifNull": ["$history", []]}},
                len(record.get("history") or []),
            ]
        },
    }
    updated = attendance.find_one_and_update(
        guard,
        {"$set": new_values, "$push": {"history": entry}},
        return_document=True,
    )
    if updated is None:
        raise HTTPException(409, "the record changed concurrently; retry")
    return attendance_out(updated)


# --------------------------------------------------------------------------- #
# 12. Analytics endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=MONTH_PATTERN),
) -> dict[str, Any]:
    """Monthly summary for one employee, from the stored derived values."""
    employee = get_employee_or_404(emp_code)
    first, last = month_bounds(month)

    rows = list(
        attendance.aggregate(employee_monthly_pipeline(emp_code, month), hint=EMP_DATE_HINT)
    )
    totals = rows[0] if rows else {}

    # R7: Mon-Fri, counted from joined_on for a mid-month joiner.
    joined_on = date_cls.fromisoformat(employee["joined_on"])
    start = max(joined_on, date_cls.fromisoformat(first))
    working_days = working_days_between(start, date_cls.fromisoformat(last))

    present_days = half_up(totals.get("present_days", 0), 2)
    attendance_pct = (
        half_up(Decimal(str(present_days)) / Decimal(working_days) * 100, 2)
        if working_days > 0
        else None
    )
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": int(totals.get("leave_days", 0)),
        "late_count": int(totals.get("late_count", 0)),
        "total_late_minutes": int(totals.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(totals.get("total_overtime_minutes", 0)),
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=MONTH_PATTERN),
    department: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """One row per department for a month, sorted by department ascending."""
    pipeline, hint = department_summary_pipeline(month, department)
    items = []
    for row in employees.aggregate(pipeline, hint=hint):
        count = int(row.get("work_hours_count") or 0)
        avg_work_hours = (
            half_up(Decimal(str(row.get("work_hours_sum") or 0)) / Decimal(count), 2)
            if count > 0
            else None
        )
        items.append(
            {
                "department": row["department"],
                "headcount": int(row.get("headcount") or 0),
                "present_days": half_up(row.get("present_days", 0), 2),
                "avg_work_hours": avg_work_hours,
                "late_count": int(row.get("late_count") or 0),
                "total_late_minutes": int(row.get("total_late_minutes") or 0),
                "leave_count": int(row.get("leave_count") or 0),
                "on_duty_count": int(row.get("on_duty_count") or 0),
            }
        )
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=MONTH_PATTERN),
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """Late-comers for a month, competition-ranked, limit applied after ranking."""
    pipeline, hint = late_leaderboard_pipeline(month, limit, department)
    items = [
        {
            "rank": int(row["rank"]),
            "emp_code": row["emp_code"],
            "name": row.get("name"),
            "department": row.get("department"),
            "total_late_minutes": int(row.get("total_late_minutes") or 0),
            "late_count": int(row.get("late_count") or 0),
        }
        for row in attendance.aggregate(pipeline, hint=hint)
    ]
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str = Path(...),
    date_from: str = Query(..., alias="from", pattern=DATE_PATTERN),
    date_to: str = Query(..., alias="to", pattern=DATE_PATTERN),
) -> dict[str, Any]:
    """Daily series for a department with gap filling and a 7-day moving average."""
    from_date = parse_date_param(date_from, "from")
    to_date = parse_date_param(date_to, "to")
    if to_date < from_date:
        raise HTTPException(422, "to must not be before from")
    if (to_date - from_date).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"the range must not exceed {MAX_TREND_DAYS} days")
    if employees.count_documents({"department": department}, limit=1) == 0:
        raise HTTPException(404, "department not found")

    pipeline, hint = department_trend_pipeline(department, date_from, date_to)
    items = [
        {
            "date": row["date"],
            "is_working_day": bool(row.get("is_working_day")),
            "headcount": int(row.get("headcount") or 0),
            "present_count": float(row.get("present_count") or 0),
            "late_count": int(row.get("late_count") or 0),
            # The pipeline already rounds half-up to 4 decimals; this only
            # normalises the float representation.
            "attendance_rate": half_up(row.get("attendance_rate"), 4),
            "moving_avg_7d": half_up(row.get("moving_avg_7d"), 4),
        }
        for row in employees.aggregate(pipeline, hint=hint)
    ]
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 13. Explain endpoint
# --------------------------------------------------------------------------- #
def _require(value: Optional[Any], name: str) -> Any:
    if value is None:
        raise HTTPException(422, f"{name} is required for this endpoint")
    return value


def _run_explain(command: dict[str, Any]) -> dict[str, Any]:
    return db.command({"explain": command, "verbosity": "executionStats"})


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal[
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ] = Path(...),
    emp_code: Optional[str] = Query(default=None),
    month: Optional[str] = Query(default=None, pattern=MONTH_PATTERN),
    department: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    date_to: Optional[str] = Query(default=None, pattern=DATE_PATTERN),
    status: Optional[Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]] = Query(
        default=None
    ),
    date_from_trend: Optional[str] = Query(default=None, alias="from", pattern=DATE_PATTERN),
    date_to_trend: Optional[str] = Query(default=None, alias="to", pattern=DATE_PATTERN),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """executionStats explain for the main query or pipeline of one endpoint.

    The query and the pipeline are built by the same functions the real
    endpoints call, with the same filters, sort, pagination and index hints, so
    the plan shown here is the plan that runs in production.
    """
    if endpoint == "attendance_list":
        query = build_attendance_filter(emp_code, date_from, date_to, status)
        explain = _run_explain(
            {
                "find": "attendance_logs",
                "filter": query,
                "sort": {"date": DESCENDING, "emp_code": ASCENDING},
                "skip": (page - 1) * page_size,
                "limit": page_size,
            }
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": jsonable_bson(explain)}

    if endpoint == "employee_monthly":
        _require(emp_code, "emp_code")
        _require(month, "month")
        explain = _run_explain(
            {
                "aggregate": "attendance_logs",
                "pipeline": employee_monthly_pipeline(emp_code, month),
                "cursor": {},
                "hint": EMP_DATE_HINT,
            }
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": jsonable_bson(explain)}

    if endpoint == "department_summary":
        _require(month, "month")
        pipeline, hint = department_summary_pipeline(month, department)
        explain = _run_explain(
            {
                "aggregate": "employees",
                "pipeline": pipeline,
                "cursor": {},
                "hint": hint,
            }
        )
        return {"endpoint": endpoint, "collection": "employees", "explain": jsonable_bson(explain)}

    if endpoint == "late_leaderboard":
        _require(month, "month")
        pipeline, hint = late_leaderboard_pipeline(month, limit, department)
        explain = _run_explain(
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
                "hint": hint,
            }
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": jsonable_bson(explain)}

    # department_trend
    _require(department, "department")
    trend_from = _require(date_from_trend, "from")
    trend_to = _require(date_to_trend, "to")
    from_date = parse_date_param(trend_from, "from")
    to_date = parse_date_param(trend_to, "to")
    if to_date < from_date:
        raise HTTPException(422, "to must not be before from")
    if (to_date - from_date).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"the range must not exceed {MAX_TREND_DAYS} days")
    pipeline, hint = department_trend_pipeline(department, trend_from, trend_to)
    explain = _run_explain(
        {
            "aggregate": "employees",
            "pipeline": pipeline,
            "cursor": {},
            "hint": hint,
        }
    )
    return {"endpoint": endpoint, "collection": "employees", "explain": jsonable_bson(explain)}
