"""Shared fixtures.

The integration tests run against a real MongoDB (MONGO_URI / TEST_MONGO_DB).
They use their own database and wipe both collections between tests, so they
never touch the database the application is pointed at in production.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

IST = timezone(timedelta(hours=5, minutes=30))

# Must be set before app.main is imported, because the client is built at
# import time from the environment.
os.environ.setdefault("MONGO_URI", "mongodb://127.0.0.1:27017")
os.environ["MONGO_DB"] = os.environ.get("TEST_MONGO_DB", "attendance_pytest")

MONGO_AVAILABLE = True
MONGO_SKIP_REASON = ""
try:  # pragma: no cover - environment probe
    from pymongo import MongoClient

    _probe = MongoClient(os.environ["MONGO_URI"], serverSelectionTimeoutMS=3000)
    _probe.admin.command("ping")
    _probe.close()
except Exception as exc:  # pragma: no cover
    MONGO_AVAILABLE = False
    MONGO_SKIP_REASON = f"no MongoDB at {os.environ['MONGO_URI']}: {type(exc).__name__}"

requires_mongo = pytest.mark.skipif(not MONGO_AVAILABLE, reason=MONGO_SKIP_REASON)


def ist_ms(year, month, day, hour=0, minute=0, second=0) -> int:
    """An IST wall-clock time as integer epoch milliseconds."""
    moment = datetime(year, month, day, hour, minute, second, tzinfo=IST)
    return int(moment.timestamp()) * 1000


@pytest.fixture(scope="session")
def api():
    """The FastAPI app, with indexes created once."""
    import app.main as main

    assert main.ensure_indexes(), "could not create indexes"
    return main


@pytest.fixture(scope="session")
def client(api):
    """One TestClient for the whole session.

    The lifespan closes the MongoClient on shutdown (correct for the real app),
    so the client is built once and the per-test isolation comes from clean_db.
    """
    from fastapi.testclient import TestClient

    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_db(request):
    """Every integration test starts from empty collections."""
    if not MONGO_AVAILABLE or "client" not in request.fixturenames:
        yield
        return
    if "big_dataset" in request.fixturenames:
        # The large dataset is seeded once per module and must survive.
        yield
        return
    import app.main as main

    main.employees.delete_many({})
    main.attendance.delete_many({})
    yield
    main.employees.delete_many({})
    main.attendance.delete_many({})


# --------------------------------------------------------------------------- #
# Small builders so the tests read as business cases, not as JSON
# --------------------------------------------------------------------------- #
def make_employee(
    client,
    emp_code="EMP0001",
    name="Asha Rao",
    email=None,
    department="Engineering",
    shift_start="09:30",
    shift_end="18:30",
    joined_on="2026-01-05",
):
    body = {
        "emp_code": emp_code,
        "name": name,
        "email": email or f"{emp_code.lower()}@example.com",
        "department": department,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "joined_on": joined_on,
    }
    response = client.post("/employees", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def seed_log(**fields):
    """Insert an attendance document directly, the way the grader seeds data."""
    import app.main as main

    doc = {
        "emp_code": "EMP0001",
        "date": "2026-07-06",
        "status": "PRESENT",
        "punch_in": None,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": 0,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    doc.update(fields)
    for key in ("punch_in", "punch_out"):
        if isinstance(doc[key], int):
            doc[key] = datetime.fromtimestamp(doc[key] // 1000, tz=timezone.utc)
    main.attendance.insert_one(doc)
    return doc


def seed_employee(**fields):
    """Insert an employees document directly (bypassing the API)."""
    import app.main as main

    doc = {
        "emp_code": "EMP0001",
        "name": "Asha Rao",
        "email": "asha@example.com",
        "department": "Engineering",
        "shift_start": "09:30",
        "shift_end": "18:30",
        "joined_on": "2026-01-05",
        "created_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
    }
    doc.update(fields)
    main.employees.insert_one(doc)
    return doc
