"""Unit tests for the business rule helpers (R1-R8). No database needed."""
from datetime import date, datetime, timedelta, timezone

import pytest

import app.main as m

IST = timezone(timedelta(hours=5, minutes=30))


def ist(year, month, day, hour=0, minute=0, second=0, micro=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=IST)


# --------------------------------------------------------------------------- #
# R8 / R4 half-up rounding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,places,expected",
    [
        (2.675, 2, 2.68),  # round() gives 2.67 here
        (2.665, 2, 2.67),
        (0.125, 2, 0.13),
        (8.455, 2, 8.46),
        (1.0 / 3, 4, 0.3333),
        (2.0 / 3, 4, 0.6667),
        (0.00005, 4, 0.0001),
        (4.495, 2, 4.50),
    ],
)
def test_half_up(value, places, expected):
    assert m.half_up(value, places) == expected


def test_half_up_passes_none_through():
    assert m.half_up(None, 2) is None


def test_builtin_round_would_have_been_wrong():
    """Guards the reason half_up exists at all."""
    assert round(2.675, 2) == 2.67
    assert m.half_up(2.675, 2) == 2.68


# --------------------------------------------------------------------------- #
# R1 epoch milliseconds, truncation to whole seconds
# --------------------------------------------------------------------------- #
def test_epoch_ms_truncates_to_whole_seconds():
    base = int(ist(2026, 7, 6, 9, 40, 0).timestamp()) * 1000
    assert m.epoch_ms_to_datetime(base + 900) == m.epoch_ms_to_datetime(base)
    assert m.epoch_ms_to_datetime(base + 900).microsecond == 0


def test_epoch_round_trip():
    ms = 1783312500000
    assert m.datetime_to_epoch_ms(m.epoch_ms_to_datetime(ms)) == ms


def test_naive_datetime_is_treated_as_utc():
    naive = datetime(2026, 7, 6, 3, 58, 0)
    aware = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
    assert m.datetime_to_epoch_ms(naive) == m.datetime_to_epoch_ms(aware)


# --------------------------------------------------------------------------- #
# R1 attendance date, including overnight shifts
# --------------------------------------------------------------------------- #
def test_attendance_date_is_the_ist_date_not_the_utc_date():
    # 00:30 IST on the 7th is 19:00 UTC on the 6th: the IST date must win.
    assert m.attendance_date_for(ist(2026, 7, 7, 0, 30), "00:00", "08:00") == "2026-07-07"


def test_attendance_date_day_shift():
    assert m.attendance_date_for(ist(2026, 7, 6, 9, 28), "09:30", "18:30") == "2026-07-06"


@pytest.mark.parametrize(
    "punch,expected",
    [
        (ist(2026, 7, 6, 21, 55), "2026-07-06"),  # before midnight: same day
        (ist(2026, 7, 7, 0, 5), "2026-07-06"),  # after midnight, before 06:00
        (ist(2026, 7, 7, 5, 59, 59), "2026-07-06"),  # one second before shift end
        (ist(2026, 7, 7, 6, 0, 0), "2026-07-07"),  # exactly shift end: new shift
        (ist(2026, 7, 7, 22, 0), "2026-07-07"),
    ],
)
def test_attendance_date_overnight_shift(punch, expected):
    assert m.attendance_date_for(punch, "22:00", "06:00") == expected


def test_overnight_detection():
    assert m.is_overnight_shift("22:00", "06:00")
    assert m.is_overnight_shift("09:30", "09:30")  # equal counts as overnight per R1
    assert not m.is_overnight_shift("09:30", "18:30")


def test_shift_end_of_overnight_shift_is_the_next_day():
    end = m.shift_end_instant("2026-07-06", "22:00", "06:00")
    assert end.astimezone(IST) == ist(2026, 7, 7, 6, 0)
    day_end = m.shift_end_instant("2026-07-06", "09:30", "18:30")
    assert day_end.astimezone(IST) == ist(2026, 7, 6, 18, 30)


# --------------------------------------------------------------------------- #
# R2 late arrival boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "punch,expected",
    [
        (ist(2026, 7, 6, 9, 0), 0),  # early
        (ist(2026, 7, 6, 9, 30, 0), 0),  # exactly on time
        (ist(2026, 7, 6, 9, 39, 59), 0),
        (ist(2026, 7, 6, 9, 40, 0), 0),  # exactly 10:00 late -> inside grace
        (ist(2026, 7, 6, 9, 40, 1), 10),  # one second later -> 10 minutes
        (ist(2026, 7, 6, 9, 41, 0), 11),
        (ist(2026, 7, 6, 10, 15, 59), 45),
        (ist(2026, 7, 6, 10, 16, 0), 46),
    ],
)
def test_late_minutes_boundaries(punch, expected):
    assert m.compute_late_minutes(punch, "2026-07-06", "09:30") == expected


def test_late_minutes_for_overnight_shift_after_midnight():
    # Shift started 22:00 on the 6th; punching in at 00:20 IST on the 7th is
    # 140 minutes late, and the shift start must be resolved on the 6th.
    assert m.compute_late_minutes(ist(2026, 7, 7, 0, 20), "2026-07-06", "22:00") == 140


# --------------------------------------------------------------------------- #
# R3 overtime threshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "punch_out,expected",
    [
        (ist(2026, 7, 6, 18, 0), 0),  # left early
        (ist(2026, 7, 6, 18, 30), 0),
        (ist(2026, 7, 6, 18, 59, 59), 0),  # 29 minutes -> below the minimum
        (ist(2026, 7, 6, 19, 0, 0), 30),  # exactly 30 minutes -> counts
        (ist(2026, 7, 6, 19, 0, 59), 30),  # floor
        (ist(2026, 7, 6, 19, 10), 40),
    ],
)
def test_overtime_boundaries(punch_out, expected):
    assert m.compute_overtime_minutes(punch_out, "2026-07-06", "09:30", "18:30") == expected


def test_overtime_overnight_uses_next_day_shift_end():
    # Shift 22:00-06:00 started on the 6th: punching out 06:40 on the 7th is
    # 40 minutes of overtime, not ~18 hours.
    assert m.compute_overtime_minutes(ist(2026, 7, 7, 6, 40), "2026-07-06", "22:00", "06:00") == 40


# --------------------------------------------------------------------------- #
# R4 work hours and R5 half day
# --------------------------------------------------------------------------- #
def test_work_hours_sample_documents():
    assert m.compute_work_hours(ist(2026, 7, 6, 9, 28), ist(2026, 7, 6, 18, 35)) == 9.12
    assert m.compute_work_hours(ist(2026, 7, 6, 10, 5), ist(2026, 7, 6, 19, 10)) == 9.08
    assert m.compute_work_hours(ist(2026, 7, 7, 9, 30), ist(2026, 7, 7, 13, 0)) == 3.5
    # Overnight: 21:55 -> 06:40 next day
    assert m.compute_work_hours(ist(2026, 7, 6, 21, 55), ist(2026, 7, 7, 6, 40)) == 8.75


def test_work_hours_rounds_half_up():
    # 4h 30m 18s = 4.505 hours exactly -> 4.51, not 4.50
    assert m.compute_work_hours(ist(2026, 7, 6, 9, 0, 0), ist(2026, 7, 6, 13, 30, 18)) == 4.51


@pytest.mark.parametrize(
    "work_hours,expected",
    [
        (None, False),
        (0.0, True),
        (4.49, True),
        (4.50, False),  # exactly 4.50 is not a half day
        (4.51, False),
        (9.12, False),
    ],
)
def test_half_day(work_hours, expected):
    assert m.compute_half_day(work_hours) is expected


def test_half_day_uses_the_rounded_value():
    # 4h 29m 58s = 4.49944... -> rounds to 4.50 -> not a half day.
    hours = m.compute_work_hours(ist(2026, 7, 6, 9, 0, 0), ist(2026, 7, 6, 13, 29, 58))
    assert hours == 4.50
    assert m.compute_half_day(hours) is False
    # One second less rounds to 4.49 -> half day.
    hours = m.compute_work_hours(ist(2026, 7, 6, 9, 0, 0), ist(2026, 7, 6, 13, 29, 39))
    assert hours == 4.49
    assert m.compute_half_day(hours) is True


# --------------------------------------------------------------------------- #
# R7 working days, R9 month bounds
# --------------------------------------------------------------------------- #
def test_month_bounds():
    assert m.month_bounds("2026-07") == ("2026-07-01", "2026-07-31")
    assert m.month_bounds("2026-02") == ("2026-02-01", "2026-02-28")
    assert m.month_bounds("2024-02") == ("2024-02-01", "2024-02-29")
    assert m.month_bounds("2026-12") == ("2026-12-01", "2026-12-31")


def test_working_days():
    assert m.working_days_between(date(2026, 7, 1), date(2026, 7, 31)) == 23
    # Mid-month joiner: from Monday the 13th to the end of July.
    assert m.working_days_between(date(2026, 7, 13), date(2026, 7, 31)) == 15
    # A single Saturday.
    assert m.working_days_between(date(2026, 7, 11), date(2026, 7, 11)) == 0
    # Joined after the period.
    assert m.working_days_between(date(2026, 8, 1), date(2026, 7, 31)) == 0


# --------------------------------------------------------------------------- #
# derive_fields puts R2-R5 together
# --------------------------------------------------------------------------- #
def test_derive_fields_open_record():
    derived = m.derive_fields(
        "PRESENT", ist(2026, 7, 6, 10, 5), None, "2026-07-06", "09:30", "18:30"
    )
    assert derived == {
        "work_hours": None,
        "late_minutes": 35,
        "overtime_minutes": 0,
        "half_day": False,
    }


def test_derive_fields_absent_clears_everything():
    derived = m.derive_fields("ABSENT", None, None, "2026-07-06", "09:30", "18:30")
    assert derived == {
        "work_hours": None,
        "late_minutes": 0,
        "overtime_minutes": 0,
        "half_day": False,
    }


def test_derive_fields_full_day():
    derived = m.derive_fields(
        "PRESENT",
        ist(2026, 7, 6, 10, 5),
        ist(2026, 7, 6, 19, 10),
        "2026-07-06",
        "09:30",
        "18:30",
    )
    assert derived == {
        "work_hours": 9.08,
        "late_minutes": 35,
        "overtime_minutes": 40,
        "half_day": False,
    }
