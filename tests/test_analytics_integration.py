"""Integration tests for the four analytics aggregations.

July 2026 is the reference month: it starts on a Wednesday, has 23 Mon-Fri
working days, and its weekends are the 4th/5th, 11th/12th, 18th/19th, 25th/26th.
"""
import pytest

from conftest import ist_ms, requires_mongo, seed_employee, seed_log

pytestmark = requires_mongo

JULY = {"month": "2026-07"}


def present(emp_code, date, work_hours=9.0, late=0, overtime=0, half_day=False, status="PRESENT"):
    seed_log(
        emp_code=emp_code,
        date=date,
        status=status,
        punch_in=ist_ms(*[int(part) for part in date.split("-")], 9, 30),
        punch_out=ist_ms(*[int(part) for part in date.split("-")], 18, 30),
        work_hours=work_hours,
        late_minutes=late,
        overtime_minutes=overtime,
        half_day=half_day,
    )


# --------------------------------------------------------------------------- #
# GET /analytics/employees/{emp_code}/monthly
# --------------------------------------------------------------------------- #
def test_monthly_unknown_employee_is_404(client):
    assert (
        client.get("/analytics/employees/EMP9999/monthly", params=JULY).status_code == 404
    )


def test_monthly_employee_with_no_logs_is_not_a_404(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    body = client.get("/analytics/employees/EMP0001/monthly", params=JULY).json()
    assert body == {
        "emp_code": "EMP0001",
        "month": "2026-07",
        "working_days": 23,
        "present_days": 0.0,
        "leave_days": 0,
        "late_count": 0,
        "total_late_minutes": 0,
        "total_overtime_minutes": 0,
        "attendance_pct": 0.0,
    }


def test_monthly_counts_weekdays_half_days_leave_and_totals(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", late=35, overtime=40)  # Monday
    present("EMP0001", "2026-07-07", work_hours=3.5, half_day=True)  # Tuesday, half day
    present("EMP0001", "2026-07-08", status="LEAVE")  # Wednesday
    present("EMP0001", "2026-07-09", status="ABSENT")  # Thursday
    present("EMP0001", "2026-07-10", status="WFH")  # Friday
    present("EMP0001", "2026-07-11", late=12, overtime=35)  # Saturday

    body = client.get("/analytics/employees/EMP0001/monthly", params=JULY).json()
    # R7: present_days counts only Mon-Fri -> 1 + 0.5 + 1 (WFH) = 2.5
    assert body["present_days"] == 2.5
    assert body["leave_days"] == 1
    # R7: the weekend record still adds to the late and overtime totals.
    assert body["late_count"] == 2
    assert body["total_late_minutes"] == 47
    assert body["total_overtime_minutes"] == 75
    assert body["working_days"] == 23
    assert body["attendance_pct"] == 10.87  # 2.5 / 23 * 100 = 10.8695... half-up


def test_monthly_mid_month_joiner_counts_working_days_from_joined_on(client):
    seed_employee(emp_code="EMP0002", joined_on="2026-07-13")
    present("EMP0002", "2026-07-14")
    body = client.get("/analytics/employees/EMP0002/monthly", params=JULY).json()
    assert body["working_days"] == 15  # 13 July (Mon) to 31 July
    assert body["present_days"] == 1.0
    assert body["attendance_pct"] == 6.67


def test_monthly_employee_who_joined_after_the_month(client):
    seed_employee(emp_code="EMP0003", joined_on="2026-09-01")
    body = client.get("/analytics/employees/EMP0003/monthly", params=JULY).json()
    assert body["working_days"] == 0
    assert body["attendance_pct"] is None


def test_monthly_ignores_records_outside_the_month(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    present("EMP0001", "2026-06-30")
    present("EMP0001", "2026-07-01")
    present("EMP0001", "2026-08-03")
    body = client.get("/analytics/employees/EMP0001/monthly", params=JULY).json()
    assert body["present_days"] == 1.0


def test_monthly_legacy_record_without_half_day(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    import app.main as main

    main.attendance.insert_one(
        {
            "emp_code": "EMP0001",
            "date": "2026-07-06",
            "status": "PRESENT",
            "work_hours": 8.98,
            "late_minutes": 0,
            "overtime_minutes": 0,
        }
    )
    body = client.get("/analytics/employees/EMP0001/monthly", params=JULY).json()
    assert body["present_days"] == 1.0  # missing half_day means false


@pytest.mark.parametrize("month", ["2026-13", "2026-7", "202607", "not-a-month"])
def test_monthly_invalid_month_is_422(client, month):
    assert (
        client.get("/analytics/employees/EMP0001/monthly", params={"month": month}).status_code
        == 422
    )


def test_monthly_requires_month(client):
    assert client.get("/analytics/employees/EMP0001/monthly").status_code == 422


# --------------------------------------------------------------------------- #
# GET /analytics/departments/summary
# --------------------------------------------------------------------------- #
def test_department_summary_counts_employees_with_zero_logs(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06")  # EMP0002 has no logs at all

    items = client.get("/analytics/departments/summary", params=JULY).json()["items"]
    assert len(items) == 1
    row = items[0]
    assert row["headcount"] == 2  # R9: the employee with no logs still counts
    assert row["present_days"] == 1.0
    assert row["avg_work_hours"] == 9.0


def test_department_summary_excludes_employees_who_join_after_the_month(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-07-31")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-08-01")
    seed_employee(emp_code="EMP0003", department="Sales", joined_on="2026-08-01")

    items = client.get("/analytics/departments/summary", params=JULY).json()["items"]
    # Engineering keeps the 31 July joiner; Sales has nobody eligible and is omitted.
    assert [row["department"] for row in items] == ["Engineering"]
    assert items[0]["headcount"] == 1


def test_department_summary_average_is_over_records_not_over_employees(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", work_hours=9.0)
    for day in ("2026-07-06", "2026-07-07", "2026-07-08"):
        present("EMP0002", day, work_hours=3.0)

    row = client.get("/analytics/departments/summary", params=JULY).json()["items"][0]
    # Over records: (9 + 3 + 3 + 3) / 4 = 4.5.
    # An average of per-employee averages would be (9 + 3) / 2 = 6.0.
    assert row["avg_work_hours"] == 4.5


def test_department_summary_ignores_records_without_work_hours(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", work_hours=8.0)
    seed_log(emp_code="EMP0001", date="2026-07-07", status="PRESENT", work_hours=None)
    seed_log(emp_code="EMP0001", date="2026-07-08", status="LEAVE", work_hours=None)
    row = client.get("/analytics/departments/summary", params=JULY).json()["items"][0]
    assert row["avg_work_hours"] == 8.0


def test_department_summary_avg_is_null_when_no_record_has_work_hours(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_log(emp_code="EMP0001", date="2026-07-08", status="ABSENT")
    row = client.get("/analytics/departments/summary", params=JULY).json()["items"][0]
    assert row["avg_work_hours"] is None
    assert row["present_days"] == 0


def test_department_summary_sorted_filtered_and_full_counts(client):
    seed_employee(emp_code="EMP0001", department="Sales", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0003", department="Support", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", late=35, overtime=40, status="ON_DUTY")
    present("EMP0001", "2026-07-07", status="LEAVE")
    present("EMP0001", "2026-07-11", late=5)  # Saturday

    body = client.get("/analytics/departments/summary", params=JULY).json()
    assert body["month"] == "2026-07"
    assert [row["department"] for row in body["items"]] == ["Engineering", "Sales", "Support"]

    sales = [row for row in body["items"] if row["department"] == "Sales"][0]
    assert sales["on_duty_count"] == 1
    assert sales["leave_count"] == 1
    assert sales["late_count"] == 2  # the Saturday record counts here
    assert sales["total_late_minutes"] == 40
    assert sales["present_days"] == 1.0  # only the Monday ON_DUTY record

    filtered = client.get(
        "/analytics/departments/summary", params={**JULY, "department": "Sales"}
    ).json()
    assert [row["department"] for row in filtered["items"]] == ["Sales"]


def test_department_summary_half_days_count_as_half(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", work_hours=3.5, half_day=True)
    present("EMP0001", "2026-07-07", work_hours=9.0)
    row = client.get("/analytics/departments/summary", params=JULY).json()["items"][0]
    assert row["present_days"] == 1.5
    assert row["avg_work_hours"] == 6.25  # (3.5 + 9.0) / 2


def test_department_summary_empty_database(client):
    assert client.get("/analytics/departments/summary", params=JULY).json() == {
        "month": "2026-07",
        "items": [],
    }


# --------------------------------------------------------------------------- #
# GET /analytics/leaderboard/late
# --------------------------------------------------------------------------- #
def _leaderboard_fixture():
    for code, dept in (
        ("EMP0001", "Engineering"),
        ("EMP0002", "Engineering"),
        ("EMP0003", "Sales"),
        ("EMP0004", "Sales"),
    ):
        seed_employee(emp_code=code, department=dept, joined_on="2026-01-05", name=f"N {code}")
    present("EMP0001", "2026-07-06", late=60)
    present("EMP0002", "2026-07-06", late=50)
    present("EMP0003", "2026-07-06", late=50)
    present("EMP0004", "2026-07-06", late=40)


def test_leaderboard_uses_competition_ranking(client):
    _leaderboard_fixture()
    items = client.get("/analytics/leaderboard/late", params=JULY).json()["items"]
    assert [(row["emp_code"], row["total_late_minutes"], row["rank"]) for row in items] == [
        ("EMP0001", 60, 1),
        ("EMP0002", 50, 2),
        ("EMP0003", 50, 2),  # tie shares the rank
        ("EMP0004", 40, 4),  # rank 3 is skipped
    ]


def test_leaderboard_limit_applies_after_ranking(client):
    _leaderboard_fixture()
    items = client.get("/analytics/leaderboard/late", params={**JULY, "limit": 2}).json()["items"]
    # Three rows for limit=2, because two employees tie at rank 2.
    assert [row["emp_code"] for row in items] == ["EMP0001", "EMP0002", "EMP0003"]
    assert [row["rank"] for row in items] == [1, 2, 2]

    items = client.get("/analytics/leaderboard/late", params={**JULY, "limit": 1}).json()["items"]
    assert [row["emp_code"] for row in items] == ["EMP0001"]


def test_leaderboard_ties_are_ordered_by_emp_code(client):
    for code in ("EMP0005", "EMP0003", "EMP0004"):
        seed_employee(emp_code=code, joined_on="2026-01-05", name=f"N {code}")
        present(code, "2026-07-06", late=30)
    items = client.get("/analytics/leaderboard/late", params=JULY).json()["items"]
    assert [row["emp_code"] for row in items] == ["EMP0003", "EMP0004", "EMP0005"]
    assert {row["rank"] for row in items} == {1}


def test_leaderboard_only_includes_positive_totals(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", late=0)
    present("EMP0002", "2026-07-06", late=15)
    items = client.get("/analytics/leaderboard/late", params=JULY).json()["items"]
    assert [row["emp_code"] for row in items] == ["EMP0002"]


def test_leaderboard_sums_and_counts_per_month(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05", name="Asha Rao")
    present("EMP0001", "2026-07-06", late=20)
    present("EMP0001", "2026-07-07", late=25)
    present("EMP0001", "2026-07-08", late=0)
    present("EMP0001", "2026-08-03", late=99)  # another month
    row = client.get("/analytics/leaderboard/late", params=JULY).json()["items"][0]
    assert row["total_late_minutes"] == 45
    assert row["late_count"] == 2
    assert row["name"] == "Asha Rao"
    assert row["department"] == "Engineering"


def test_leaderboard_ranks_inside_one_department(client):
    _leaderboard_fixture()
    items = client.get(
        "/analytics/leaderboard/late", params={**JULY, "department": "Sales"}
    ).json()["items"]
    # Ranking restarts inside the department: EMP0003 is rank 1 there.
    assert [(row["emp_code"], row["rank"]) for row in items] == [("EMP0003", 1), ("EMP0004", 2)]


def test_leaderboard_ignores_logs_without_an_employee(client):
    seed_employee(emp_code="EMP0001", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", late=10)
    present("EMP9999", "2026-07-06", late=99)  # no such employee
    items = client.get("/analytics/leaderboard/late", params=JULY).json()["items"]
    assert [row["emp_code"] for row in items] == ["EMP0001"]


def test_leaderboard_empty_month(client):
    assert client.get("/analytics/leaderboard/late", params=JULY).json()["items"] == []


@pytest.mark.parametrize("limit", [0, -1, 51])
def test_leaderboard_invalid_limit_is_422(client, limit):
    assert (
        client.get("/analytics/leaderboard/late", params={**JULY, "limit": limit}).status_code
        == 422
    )


# --------------------------------------------------------------------------- #
# GET /analytics/departments/{department}/trend
# --------------------------------------------------------------------------- #
def test_trend_unknown_department_is_404(client):
    assert (
        client.get(
            "/analytics/departments/Nope/trend",
            params={"from": "2026-07-06", "to": "2026-07-07"},
        ).status_code
        == 404
    )


def test_trend_has_one_row_per_calendar_day_including_weekends_and_empty_days(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06")

    body = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-13"},
    ).json()
    assert body["department"] == "Engineering"
    dates = [row["date"] for row in body["items"]]
    assert dates == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
        "2026-07-11",
        "2026-07-12",
        "2026-07-13",
    ]
    by_date = {row["date"]: row for row in body["items"]}

    assert by_date["2026-07-06"]["present_count"] == 1
    assert by_date["2026-07-06"]["attendance_rate"] == 0.5  # 1 of 2
    # A working day with no records has rate 0.0, not null.
    assert by_date["2026-07-07"]["attendance_rate"] == 0.0
    # Weekends are present as rows but have no rate.
    assert by_date["2026-07-11"]["is_working_day"] is False
    assert by_date["2026-07-11"]["attendance_rate"] is None
    assert by_date["2026-07-11"]["headcount"] == 2
    assert by_date["2026-07-13"]["is_working_day"] is True


def test_trend_headcount_grows_with_a_mid_period_joiner(client):
    seed_employee(emp_code="EMP0001", department="Support", joined_on="2026-07-01")
    seed_employee(emp_code="EMP0002", department="Support", joined_on="2026-07-08")
    seed_employee(emp_code="EMP0003", department="Support", joined_on="2026-08-01")

    rows = client.get(
        "/analytics/departments/Support/trend",
        params={"from": "2026-07-06", "to": "2026-07-09"},
    ).json()["items"]
    assert [row["headcount"] for row in rows] == [1, 1, 2, 2]


def test_trend_rate_is_four_decimals_half_up(client):
    for code in ("EMP0001", "EMP0002", "EMP0003"):
        seed_employee(emp_code=code, department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06")
    present("EMP0001", "2026-07-07")
    present("EMP0002", "2026-07-07")

    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-07"},
    ).json()["items"]
    assert rows[0]["attendance_rate"] == 0.3333  # 1/3
    assert rows[1]["attendance_rate"] == 0.6667  # 2/3


def test_trend_moving_average_window(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    # Monday 6 July: both present -> rate 1.0; Tuesday: one present -> 0.5.
    present("EMP0001", "2026-07-06")
    present("EMP0002", "2026-07-06")
    present("EMP0001", "2026-07-07")

    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-14"},
    ).json()["items"]
    by_date = {row["date"]: row for row in rows}
    # First row: a window of one.
    assert by_date["2026-07-06"]["moving_avg_7d"] == 1.0
    # Second row: (1.0 + 0.5) / 2.
    assert by_date["2026-07-07"]["moving_avg_7d"] == 0.75
    # Third row: (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert by_date["2026-07-08"]["moving_avg_7d"] == 0.5
    # Weekend rows have no rate of their own but keep the running average.
    assert by_date["2026-07-11"]["attendance_rate"] is None
    assert by_date["2026-07-11"]["moving_avg_7d"] == 0.3  # (1.0+0.5+0+0+0)/5
    # Monday 13 July: the 7-row window is 7-13 July, so 6 July has dropped out.
    assert by_date["2026-07-13"]["moving_avg_7d"] == 0.1  # (0.5+0+0+0+0)/5
    # Tuesday 14 July: the window is 8-14 July, so no rate above 0 is left.
    assert by_date["2026-07-14"]["moving_avg_7d"] == 0.0


def test_trend_moving_average_is_null_when_the_window_has_no_rate(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-11", "to": "2026-07-12"},  # Saturday and Sunday
    ).json()["items"]
    assert [row["attendance_rate"] for row in rows] == [None, None]
    assert [row["moving_avg_7d"] for row in rows] == [None, None]


def test_trend_rate_is_null_when_headcount_is_zero(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-08-01")
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-07"},
    ).json()["items"]
    assert [row["headcount"] for row in rows] == [0, 0]
    assert [row["attendance_rate"] for row in rows] == [None, None]


def test_trend_counts_weekend_records_in_present_count(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-11")  # Saturday
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-11", "to": "2026-07-11"},
    ).json()["items"]
    assert rows[0]["present_count"] == 1
    assert rows[0]["is_working_day"] is False
    assert rows[0]["attendance_rate"] is None  # no rate on a non-working day


def test_trend_single_day_range(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-06"},
    ).json()["items"]
    assert len(rows) == 1


def test_trend_other_departments_do_not_leak_in(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Sales", joined_on="2026-01-05")
    present("EMP0002", "2026-07-06")
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-06"},
    ).json()["items"]
    assert rows[0]["present_count"] == 0
    assert rows[0]["headcount"] == 1


def test_trend_half_day_counts_as_half(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    seed_employee(emp_code="EMP0002", department="Engineering", joined_on="2026-01-05")
    present("EMP0001", "2026-07-06", work_hours=3.5, half_day=True)
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-06", "to": "2026-07-06"},
    ).json()["items"]
    assert rows[0]["present_count"] == 0.5
    assert rows[0]["attendance_rate"] == 0.25


def test_trend_range_validation(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    base = "/analytics/departments/Engineering/trend"
    assert client.get(base, params={"from": "2026-07-10", "to": "2026-07-06"}).status_code == 422
    assert client.get(base, params={"from": "2026-01-01", "to": "2026-12-31"}).status_code == 422
    assert client.get(base, params={"from": "2026-07-06"}).status_code == 422
    assert client.get(base, params={"from": "06-07-2026", "to": "2026-07-07"}).status_code == 422
    # Exactly 92 days is allowed.
    assert client.get(base, params={"from": "2026-07-01", "to": "2026-09-30"}).status_code == 200
    # 93 days is not.
    assert client.get(base, params={"from": "2026-07-01", "to": "2026-10-01"}).status_code == 422


def test_trend_full_92_day_range_has_92_rows(client):
    seed_employee(emp_code="EMP0001", department="Engineering", joined_on="2026-01-05")
    rows = client.get(
        "/analytics/departments/Engineering/trend",
        params={"from": "2026-07-01", "to": "2026-09-30"},
    ).json()["items"]
    assert len(rows) == 92
    assert rows[0]["date"] == "2026-07-01"
    assert rows[-1]["date"] == "2026-09-30"
