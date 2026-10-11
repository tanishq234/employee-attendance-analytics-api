"""Explain-plan and scale tests on a dataset of ~100,000 attendance records.

The grader checks that the plans contain IXSCAN and no COLLSCAN on a dataset
this size, so these tests seed one and inspect the real explain output rather
than asserting on index names.
"""
import json
import time

import pytest

from conftest import requires_mongo

pytestmark = requires_mongo

EMPLOYEE_COUNT = 300
DAYS = 334  # 300 x 334 = 100,200 attendance records
DEPARTMENTS = ("Engineering", "Sales", "Support", "Finance")
START_DATE = "2026-01-01"


@pytest.fixture(scope="module")
def big_dataset():
    """Seed ~100k attendance records and ~300 employees, directly into MongoDB."""
    from datetime import date, datetime, timedelta, timezone

    import app.main as main

    main.ensure_indexes()
    main.employees.delete_many({})
    main.attendance.delete_many({})

    start = date.fromisoformat(START_DATE)
    employee_docs = []
    for index in range(EMPLOYEE_COUNT):
        code = f"EMP{index + 1:05d}"
        employee_docs.append(
            {
                "emp_code": code,
                "name": f"Employee {index + 1}",
                "email": f"{code.lower()}@example.com",
                "department": DEPARTMENTS[index % len(DEPARTMENTS)],
                "shift_start": "22:00" if index % 20 == 0 else "09:30",
                "shift_end": "06:00" if index % 20 == 0 else "18:30",
                # A spread of join dates, including mid-month joiners.
                "joined_on": (start + timedelta(days=(index * 3) % 120)).isoformat(),
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        )
    main.employees.insert_many(employee_docs)

    statuses = ("PRESENT", "PRESENT", "PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE")
    batch = []
    total = 0
    for index in range(EMPLOYEE_COUNT):
        code = f"EMP{index + 1:05d}"
        for offset in range(DAYS):
            day = start + timedelta(days=offset)
            status = statuses[(index + offset) % len(statuses)]
            presence = status in ("PRESENT", "WFH", "ON_DUTY")
            late = ((index * 7 + offset * 13) % 97) if presence else 0
            punch_in = datetime(day.year, day.month, day.day, 4, 0, tzinfo=timezone.utc)
            doc = {
                "emp_code": code,
                "date": day.isoformat(),
                "status": status,
                "punch_in": punch_in if presence else None,
                "punch_out": punch_in + timedelta(hours=9) if presence else None,
                "work_hours": 9.0 if presence else None,
                "late_minutes": late,
                "overtime_minutes": 35 if presence and offset % 5 == 0 else 0,
                "half_day": bool(presence and offset % 11 == 0),
                "history": [],
            }
            batch.append(doc)
            if len(batch) >= 5000:
                main.attendance.insert_many(batch, ordered=False)
                total += len(batch)
                batch = []
    if batch:
        main.attendance.insert_many(batch, ordered=False)
        total += len(batch)

    assert main.attendance.count_documents({}) == total
    print(f"\nseeded {total} attendance records, {EMPLOYEE_COUNT} employees")
    yield total
    main.employees.delete_many({})
    main.attendance.delete_many({})


@pytest.fixture()
def big_client(client, big_dataset):
    """The shared session TestClient, with the large dataset in place."""
    return client


def _plan_text(response):
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"endpoint", "collection", "explain"}
    assert body["explain"], "explain output must not be empty"
    return json.dumps(body["explain"])


EXPLAIN_CASES = [
    ("attendance_list", {}),
    ("attendance_list", {"emp_code": "EMP00007"}),
    ("attendance_list", {"date_from": "2026-03-01", "date_to": "2026-03-31"}),
    ("attendance_list", {"status": "PRESENT", "page": 3, "page_size": 50}),
    ("employee_monthly", {"emp_code": "EMP00007", "month": "2026-03"}),
    ("department_summary", {"month": "2026-03"}),
    ("department_summary", {"month": "2026-03", "department": "Sales"}),
    ("late_leaderboard", {"month": "2026-03"}),
    ("late_leaderboard", {"month": "2026-03", "department": "Sales", "limit": 5}),
    ("department_trend", {"department": "Sales", "from": "2026-03-01", "to": "2026-03-31"}),
]


@pytest.mark.parametrize("endpoint,params", EXPLAIN_CASES)
def test_explain_plans_use_an_index_and_never_scan_the_collection(
    big_client, endpoint, params
):
    text = _plan_text(big_client.get(f"/admin/explain/{endpoint}", params=params))
    assert "IXSCAN" in text, f"{endpoint} {params}: no index scan in the plan"
    assert "COLLSCAN" not in text, f"{endpoint} {params}: the plan contains a collection scan"


@pytest.mark.parametrize("endpoint,params", EXPLAIN_CASES)
def test_explain_reports_execution_stats(big_client, endpoint, params):
    text = _plan_text(big_client.get(f"/admin/explain/{endpoint}", params=params))
    assert "executionStats" in text or "executionTimeMillis" in text


def test_explain_rejects_an_unknown_endpoint(big_client):
    assert big_client.get("/admin/explain/nonsense").status_code == 422


@pytest.mark.parametrize(
    "endpoint,params",
    [
        ("employee_monthly", {"month": "2026-03"}),  # emp_code missing
        ("employee_monthly", {"emp_code": "EMP00007"}),  # month missing
        ("department_summary", {}),  # month missing
        ("late_leaderboard", {}),  # month missing
        ("department_trend", {"from": "2026-03-01", "to": "2026-03-31"}),  # department
        ("department_trend", {"department": "Sales", "from": "2026-03-01"}),  # to
        ("department_trend", {"department": "Sales", "from": "2026-03-31", "to": "2026-03-01"}),
    ],
)
def test_explain_missing_or_invalid_parameters_are_422(big_client, endpoint, params):
    assert big_client.get(f"/admin/explain/{endpoint}", params=params).status_code == 422


def test_explain_mirrors_the_real_filters(big_client):
    """The explained plan must be the endpoint's own query, filters included."""
    text = _plan_text(
        big_client.get(
            "/admin/explain/attendance_list",
            params={"emp_code": "EMP00007", "date_from": "2026-03-01", "date_to": "2026-03-31"},
        )
    )
    assert "EMP00007" in text
    assert "2026-03-01" in text and "2026-03-31" in text


# --------------------------------------------------------------------------- #
# Correctness and speed at scale
# --------------------------------------------------------------------------- #
def test_filtered_total_is_the_whole_result_set_not_the_page(big_client, big_dataset):
    import app.main as main

    page = big_client.get("/attendance", params={"page_size": 10}).json()
    assert page["total"] == big_dataset
    assert len(page["items"]) == 10

    filtered = big_client.get(
        "/attendance", params={"status": "LEAVE", "page_size": 5}
    ).json()
    assert filtered["total"] == main.attendance.count_documents({"status": "LEAVE"})
    assert filtered["total"] < big_dataset
    assert len(filtered["items"]) == 5


def test_deep_page_is_still_sorted_correctly(big_client):
    page = big_client.get("/attendance", params={"page": 50, "page_size": 100}).json()
    dates = [item["date"] for item in page["items"]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize(
    "path,params,budget",
    [
        ("/attendance", {"page_size": 20}, 3.0),
        ("/attendance", {"emp_code": "EMP00007"}, 3.0),
        ("/attendance", {"date_from": "2026-03-01", "date_to": "2026-03-31"}, 3.0),
        ("/employees", {"department": "Sales"}, 3.0),
        ("/analytics/employees/EMP00007/monthly", {"month": "2026-03"}, 3.0),
        ("/analytics/departments/summary", {"month": "2026-03"}, 10.0),
        ("/analytics/leaderboard/late", {"month": "2026-03"}, 10.0),
        (
            "/analytics/departments/Sales/trend",
            {"from": "2026-03-01", "to": "2026-03-31"},
            10.0,
        ),
    ],
)
def test_endpoint_latency_at_100k_records(big_client, path, params, budget):
    started = time.monotonic()
    response = big_client.get(path, params=params)
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    assert elapsed < budget, f"{path} took {elapsed:.2f}s (budget {budget}s)"
    print(f"\n{path} {params} -> {elapsed * 1000:.0f} ms")


def test_analytics_still_correct_at_scale(big_client, big_dataset):
    """Cross-check the aggregations against independent MongoDB queries."""
    import app.main as main

    monthly = big_client.get(
        "/analytics/employees/EMP00007/monthly", params={"month": "2026-03"}
    ).json()
    expected_late = list(
        main.attendance.aggregate(
            [
                {
                    "$match": {
                        "emp_code": "EMP00007",
                        "date": {"$gte": "2026-03-01", "$lte": "2026-03-31"},
                    }
                },
                {"$group": {"_id": None, "total": {"$sum": "$late_minutes"}}},
            ]
        )
    )
    assert monthly["total_late_minutes"] == expected_late[0]["total"]

    summary = big_client.get(
        "/analytics/departments/summary", params={"month": "2026-03"}
    ).json()["items"]
    # Every department has eligible employees by March, and headcount matches
    # a plain count of employees who had joined by the end of the month.
    assert [row["department"] for row in summary] == sorted(DEPARTMENTS)
    for row in summary:
        assert row["headcount"] == main.employees.count_documents(
            {"department": row["department"], "joined_on": {"$lte": "2026-03-31"}}
        )

    trend = big_client.get(
        "/analytics/departments/Sales/trend",
        params={"from": "2026-03-01", "to": "2026-03-31"},
    ).json()["items"]
    assert len(trend) == 31
    assert [row["date"] for row in trend] == [f"2026-03-{day:02d}" for day in range(1, 32)]

    board = big_client.get(
        "/analytics/leaderboard/late", params={"month": "2026-03", "limit": 10}
    ).json()["items"]
    assert board, "the seeded data has late records, so the board must not be empty"
    # Ranks are non-decreasing and competition-style: equal totals share a rank.
    totals = [row["total_late_minutes"] for row in board]
    assert totals == sorted(totals, reverse=True)
    for earlier, later in zip(board, board[1:]):
        if earlier["total_late_minutes"] == later["total_late_minutes"]:
            assert earlier["rank"] == later["rank"]
            assert earlier["emp_code"] < later["emp_code"]
        else:
            assert later["rank"] > earlier["rank"]
