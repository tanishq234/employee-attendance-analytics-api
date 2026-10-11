"""Integration tests for the CRUD endpoints against a real MongoDB.

Covers validation, status codes, R1-R5 end to end, pagination (R10) and the
regularization audit trail.
"""
import pytest

from conftest import ist_ms, make_employee, requires_mongo, seed_employee, seed_log

pytestmark = requires_mongo


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_is_ok_when_mongodb_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# POST /employees
# --------------------------------------------------------------------------- #
def test_create_employee_returns_201_and_epoch_created_at(client):
    created = make_employee(client)
    assert created["emp_code"] == "EMP0001"
    assert isinstance(created["created_at"], int)
    assert created["created_at"] > 100_000_000_000
    # _id must never leak.
    assert "_id" not in created and "id" not in created


def test_duplicate_emp_code_is_409(client):
    make_employee(client)
    response = client.post(
        "/employees",
        json={
            "emp_code": "EMP0001",
            "name": "Someone Else",
            "email": "other@example.com",
            "department": "Sales",
            "joined_on": "2026-01-05",
        },
    )
    assert response.status_code == 409


def test_employee_defaults_are_the_contract_defaults(client):
    response = client.post(
        "/employees",
        json={
            "emp_code": "EMP0009",
            "name": "Default Shift",
            "email": "d@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-05",
        },
    )
    assert response.status_code == 201
    assert response.json()["shift_start"] == "09:30"
    assert response.json()["shift_end"] == "18:30"


@pytest.mark.parametrize(
    "overrides",
    [
        {"emp_code": "E1"},  # wrong pattern
        {"emp_code": "EMP01"},  # too few digits
        {"emp_code": "EMP01234567"},  # too many digits
        {"name": ""},
        {"email": "not-an-email"},
        {"department": ""},
        {"shift_start": "9:30"},
        {"shift_start": "25:00"},
        {"shift_end": "18:60"},
        {"joined_on": "06-07-2026"},
        {"joined_on": "2026-02-30"},  # not a real date
        {"shift_start": "09:30", "shift_end": "09:30"},  # must differ
    ],
)
def test_invalid_employee_is_422(client, overrides):
    body = {
        "emp_code": "EMP0001",
        "name": "Asha Rao",
        "email": "asha@example.com",
        "department": "Engineering",
        "joined_on": "2026-01-05",
    }
    body.update(overrides)
    assert client.post("/employees", json=body).status_code == 422


def test_missing_required_employee_field_is_422(client):
    assert client.post("/employees", json={"emp_code": "EMP0001"}).status_code == 422


# --------------------------------------------------------------------------- #
# GET /employees
# --------------------------------------------------------------------------- #
def test_list_employees_sorted_filtered_and_paginated(client):
    make_employee(client, "EMP0003", department="Sales")
    make_employee(client, "EMP0001", department="Engineering")
    make_employee(client, "EMP0002", department="Engineering")

    page = client.get("/employees").json()
    assert [item["emp_code"] for item in page["items"]] == ["EMP0001", "EMP0002", "EMP0003"]
    assert page["total"] == 3
    assert page["page"] == 1 and page["page_size"] == 20

    # total reflects the filter, not the collection (R10).
    filtered = client.get("/employees", params={"department": "Engineering"}).json()
    assert filtered["total"] == 2
    assert [item["emp_code"] for item in filtered["items"]] == ["EMP0001", "EMP0002"]

    # page 1 is the first page, not the second.
    first = client.get("/employees", params={"page": 1, "page_size": 1}).json()
    assert [item["emp_code"] for item in first["items"]] == ["EMP0001"]
    second = client.get("/employees", params={"page": 2, "page_size": 1}).json()
    assert [item["emp_code"] for item in second["items"]] == ["EMP0002"]


def test_department_filter_is_case_sensitive(client):
    make_employee(client, "EMP0001", department="Engineering")
    assert client.get("/employees", params={"department": "engineering"}).json()["total"] == 0


@pytest.mark.parametrize("params", [{"page": 0}, {"page": -1}, {"page_size": 0}, {"page_size": 101}])
def test_invalid_pagination_is_422(client, params):
    assert client.get("/employees", params=params).status_code == 422


# --------------------------------------------------------------------------- #
# POST /attendance/punch-in
# --------------------------------------------------------------------------- #
def test_punch_in_creates_the_record_with_derived_defaults(client):
    make_employee(client)
    punched_at = ist_ms(2026, 7, 6, 9, 28)
    response = client.post(
        "/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": punched_at}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["date"] == "2026-07-06"
    assert body["punch_in"] == punched_at
    assert body["punch_out"] is None
    assert body["work_hours"] is None
    assert body["late_minutes"] == 0
    assert body["overtime_minutes"] == 0
    assert body["half_day"] is False
    assert body["history"] == []
    assert "id" not in body and "_id" not in body


def test_punch_in_truncates_sub_second_precision(client):
    make_employee(client)
    base = ist_ms(2026, 7, 6, 9, 40, 0)
    response = client.post(
        "/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": base + 900}
    )
    # R1: the truncated value is what is stored and returned.
    assert response.json()["punch_in"] == base
    assert response.json()["late_minutes"] == 0


def test_punch_in_late_boundary_one_second_past_grace(client):
    make_employee(client)
    response = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 40, 1)},
    )
    assert response.json()["late_minutes"] == 10


def test_punch_in_overnight_shift_after_midnight_belongs_to_previous_day(client):
    make_employee(client, shift_start="22:00", shift_end="06:00")
    response = client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 0, 5)},
    )
    assert response.status_code == 201
    assert response.json()["date"] == "2026-07-06"


def test_punch_in_unknown_employee_is_404(client):
    response = client.post("/attendance/punch-in", json={"emp_code": "EMP9999"})
    assert response.status_code == 404


def test_duplicate_punch_in_for_the_same_date_is_409(client):
    make_employee(client)
    payload = {"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 28)}
    assert client.post("/attendance/punch-in", json=payload).status_code == 201
    assert client.post("/attendance/punch-in", json=payload).status_code == 409


def test_punch_in_accepts_only_presence_statuses(client):
    make_employee(client)
    for status in ("PRESENT", "WFH", "ON_DUTY"):
        response = client.post(
            "/attendance/punch-in",
            json={
                "emp_code": "EMP0001",
                "status": status,
                "punched_at": ist_ms(2026, 7, {"PRESENT": 6, "WFH": 7, "ON_DUTY": 8}[status], 9, 28),
            },
        )
        assert response.status_code == 201, status
        assert response.json()["status"] == status

    for status in ("ABSENT", "LEAVE", "present", "SOMETHING"):
        response = client.post(
            "/attendance/punch-in",
            json={
                "emp_code": "EMP0001",
                "status": status,
                "punched_at": ist_ms(2026, 7, 9, 9, 28),
            },
        )
        assert response.status_code == 422, status


@pytest.mark.parametrize(
    "punched_at",
    [
        1783312500,  # seconds, not milliseconds
        1783312500000.0,  # float
        "1783312500000",  # string
        0,
        -1,
        99_999_999_999,  # just below the minimum
        4_102_444_800_001,  # just above the maximum
    ],
)
def test_bad_timestamp_is_422(client, punched_at):
    make_employee(client)
    response = client.post(
        "/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": punched_at}
    )
    assert response.status_code == 422


def test_punch_in_without_punched_at_uses_now(client):
    make_employee(client)
    response = client.post("/attendance/punch-in", json={"emp_code": "EMP0001"})
    assert response.status_code == 201
    assert response.json()["punch_in"] % 1000 == 0  # truncated to whole seconds


def test_client_supplied_derived_fields_are_ignored(client):
    make_employee(client)
    response = client.post(
        "/attendance/punch-in",
        json={
            "emp_code": "EMP0001",
            "punched_at": ist_ms(2026, 7, 6, 10, 5),
            "work_hours": 99.0,
            "late_minutes": 0,
            "half_day": True,
        },
    )
    assert response.status_code == 201
    assert response.json()["late_minutes"] == 35
    assert response.json()["work_hours"] is None
    assert response.json()["half_day"] is False


# --------------------------------------------------------------------------- #
# POST /attendance/punch-out
# --------------------------------------------------------------------------- #
def test_punch_out_computes_work_hours_overtime_and_half_day(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 10, 5)},
    )
    response = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 19, 10)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["work_hours"] == 9.08
    assert body["late_minutes"] == 35
    assert body["overtime_minutes"] == 40
    assert body["half_day"] is False
    assert body["history"] == []  # punch-out never writes history


def test_punch_out_half_day(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 9, 30)},
    )
    body = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 13, 0)},
    ).json()
    assert body["work_hours"] == 3.5
    assert body["half_day"] is True


def test_punch_out_exactly_four_and_a_half_hours_is_not_a_half_day(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 9, 30)},
    )
    body = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 14, 0)},
    ).json()
    assert body["work_hours"] == 4.5
    assert body["half_day"] is False


def test_punch_out_across_midnight_on_an_overnight_shift(client):
    make_employee(client, shift_start="22:00", shift_end="06:00")
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 21, 55)},
    )
    body = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 6, 40)},
    ).json()
    assert body["date"] == "2026-07-06"  # still the day the shift started
    assert body["work_hours"] == 8.75
    assert body["overtime_minutes"] == 40


def test_punch_out_unknown_employee_is_404(client):
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP9999"}).status_code == 404


def test_punch_out_without_a_punch_in_is_404(client):
    make_employee(client)
    response = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 18, 35)},
    )
    assert response.status_code == 404


def test_second_punch_out_is_409(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 28)},
    )
    payload = {"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 18, 35)}
    assert client.post("/attendance/punch-out", json=payload).status_code == 200
    assert client.post("/attendance/punch-out", json=payload).status_code == 409


def test_punch_out_before_or_equal_to_punch_in_is_422(client):
    make_employee(client)
    punch_in_at = ist_ms(2026, 7, 6, 9, 28)
    client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": punch_in_at})
    # Equal is rejected; earlier never matches the record at all (404).
    assert (
        client.post(
            "/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": punch_in_at}
        ).status_code
        == 422
    )


def test_punch_out_more_than_24_hours_later_is_422(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 28)},
    )
    response = client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 9, 29)},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /attendance
# --------------------------------------------------------------------------- #
def test_list_attendance_sorting_filters_and_total(client):
    seed_employee(emp_code="EMP0001")
    seed_employee(emp_code="EMP0002", email="b@example.com")
    seed_log(emp_code="EMP0001", date="2026-07-06")
    seed_log(emp_code="EMP0002", date="2026-07-06", email=None)
    seed_log(emp_code="EMP0001", date="2026-07-07", status="LEAVE")

    page = client.get("/attendance").json()
    # date descending, then emp_code ascending
    assert [(item["date"], item["emp_code"]) for item in page["items"]] == [
        ("2026-07-07", "EMP0001"),
        ("2026-07-06", "EMP0001"),
        ("2026-07-06", "EMP0002"),
    ]
    assert page["total"] == 3

    assert client.get("/attendance", params={"emp_code": "EMP0001"}).json()["total"] == 2
    assert client.get("/attendance", params={"status": "LEAVE"}).json()["total"] == 1
    assert (
        client.get(
            "/attendance", params={"date_from": "2026-07-07", "date_to": "2026-07-07"}
        ).json()["total"]
        == 1
    )
    # total is the filtered total even when the page is smaller
    small = client.get("/attendance", params={"page_size": 1}).json()
    assert small["total"] == 3 and len(small["items"]) == 1


def test_attendance_date_range_is_inclusive(client):
    seed_employee()
    seed_log(date="2026-07-06")
    seed_log(date="2026-07-10")
    result = client.get(
        "/attendance", params={"date_from": "2026-07-06", "date_to": "2026-07-10"}
    ).json()
    assert result["total"] == 2


def test_attendance_reversed_date_range_is_422(client):
    response = client.get(
        "/attendance", params={"date_from": "2026-07-10", "date_to": "2026-07-06"}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"date_from": "06-07-2026"},
        {"date_to": "2026-7-6"},
        {"status": "NOPE"},
        {"date_from": "2026-02-30"},
    ],
)
def test_attendance_invalid_filters_are_422(client, params):
    assert client.get("/attendance", params=params).status_code == 422


def test_legacy_record_without_history_or_half_day(client):
    """DATA_MODEL section 9: older records omit both fields entirely."""
    seed_employee()
    import app.main as main

    main.attendance.insert_one(
        {
            "emp_code": "EMP0001",
            "date": "2026-07-14",
            "status": "PRESENT",
            "punch_in": None,
            "punch_out": None,
            "work_hours": 8.98,
            "late_minutes": 0,
            "overtime_minutes": 0,
        }
    )
    item = client.get("/attendance", params={"emp_code": "EMP0001"}).json()["items"][0]
    assert item["half_day"] is False
    assert item["history"] == []


# --------------------------------------------------------------------------- #
# PATCH /attendance/{emp_code}/{date}
# --------------------------------------------------------------------------- #
def _late_record(client):
    make_employee(client)
    client.post(
        "/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 10, 5)},
    )
    client.post(
        "/attendance/punch-out",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 7, 19, 0)},
    )


def test_regularize_recomputes_and_appends_one_history_entry(client):
    _late_record(client)
    corrected_in = ist_ms(2026, 7, 7, 9, 28)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_in": corrected_in,
            "reason": "biometric glitch",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["punch_in"] == corrected_in
    assert body["late_minutes"] == 0
    assert body["work_hours"] == 9.53
    assert len(body["history"]) == 1

    entry = body["history"][0]
    assert entry["by"] == "hr.admin"
    assert entry["reason"] == "biometric glitch"
    assert isinstance(entry["at"], int)
    # Only the fields that actually changed, with nested instants in epoch ms.
    assert set(entry["changes"]) == {"punch_in", "late_minutes", "work_hours"}
    assert entry["changes"]["punch_in"] == {
        "from": ist_ms(2026, 7, 7, 10, 5),
        "to": corrected_in,
    }
    assert entry["changes"]["late_minutes"] == {"from": 35, "to": 0}
    assert "status" not in entry["changes"]
    assert "overtime_minutes" not in entry["changes"]


def test_regularize_to_absent_clears_the_punch_times(client):
    _late_record(client)
    body = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={"status": "ABSENT", "reason": "no show confirmed", "regularized_by": "hr.admin"},
    ).json()
    assert body["status"] == "ABSENT"
    assert body["punch_in"] is None and body["punch_out"] is None
    assert body["work_hours"] is None
    assert body["late_minutes"] == 0 and body["overtime_minutes"] == 0
    assert body["half_day"] is False
    changes = body["history"][0]["changes"]
    assert changes["status"] == {"from": "PRESENT", "to": "ABSENT"}
    assert changes["punch_in"]["to"] is None


def test_regularize_absent_with_punch_times_is_422(client):
    _late_record(client)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "status": "LEAVE",
            "punch_in": ist_ms(2026, 7, 7, 9, 30),
            "reason": "approved leave",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 422


def test_regularize_presence_without_any_punch_in_is_422(client):
    make_employee(client)
    seed_log(emp_code="EMP0001", date="2026-07-08", status="ABSENT")
    response = client.patch(
        "/attendance/EMP0001/2026-07-08",
        json={"status": "PRESENT", "reason": "was actually here", "regularized_by": "hr.admin"},
    )
    assert response.status_code == 422


def test_regularize_absent_to_present_with_a_punch_in_works(client):
    make_employee(client)
    seed_log(emp_code="EMP0001", date="2026-07-08", status="ABSENT")
    body = client.patch(
        "/attendance/EMP0001/2026-07-08",
        json={
            "status": "PRESENT",
            "punch_in": ist_ms(2026, 7, 8, 9, 28),
            "reason": "was actually here",
            "regularized_by": "hr.admin",
        },
    ).json()
    assert body["status"] == "PRESENT"
    assert body["punch_in"] == ist_ms(2026, 7, 8, 9, 28)
    assert body["work_hours"] is None


def test_regularize_punch_in_must_stay_on_the_records_date(client):
    _late_record(client)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_in": ist_ms(2026, 7, 8, 9, 30),  # next day
            "reason": "wrong day correction",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 422


def test_regularize_punch_out_must_be_after_punch_in(client):
    _late_record(client)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_out": ist_ms(2026, 7, 7, 9, 0),
            "reason": "bad punch out",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 422


def test_regularize_punch_out_within_24_hours(client):
    _late_record(client)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_out": ist_ms(2026, 7, 8, 11, 0),
            "reason": "far too late",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 422


def test_regularize_that_changes_nothing_is_422(client):
    _late_record(client)
    response = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_in": ist_ms(2026, 7, 7, 10, 5),  # the stored value
            "reason": "no actual change",
            "regularized_by": "hr.admin",
        },
    )
    assert response.status_code == 422


def test_regularize_unknown_employee_or_date_is_404(client):
    make_employee(client)
    body = {"status": "LEAVE", "reason": "some reason", "regularized_by": "hr.admin"}
    assert client.patch("/attendance/EMP9999/2026-07-07", json=body).status_code == 404
    assert client.patch("/attendance/EMP0001/2026-07-07", json=body).status_code == 404


def test_regularize_malformed_date_is_422_not_404(client):
    body = {"status": "LEAVE", "reason": "some reason", "regularized_by": "hr.admin"}
    assert client.patch("/attendance/EMP9999/2026-02-30", json=body).status_code == 422
    assert client.patch("/attendance/EMP9999/06-07-2026", json=body).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"reason": "tiny", "regularized_by": "hr"},  # reason too short
        {"reason": "valid reason", "regularized_by": ""},  # actor empty
        {"status": "PRESENT"},  # reason and actor missing
        {"reason": "valid reason", "regularized_by": "hr"},  # nothing to change
        {"status": "NOPE", "reason": "valid reason", "regularized_by": "hr"},
        {"punch_in": 1783312500, "reason": "valid reason", "regularized_by": "hr"},
    ],
)
def test_regularize_invalid_body_is_422(client, body):
    _late_record(client)
    assert client.patch("/attendance/EMP0001/2026-07-07", json=body).status_code == 422


def test_sequential_regularizations_keep_every_history_entry(client):
    _late_record(client)
    first = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={
            "punch_in": ist_ms(2026, 7, 7, 9, 28),
            "reason": "first correction",
            "regularized_by": "hr.one",
        },
    )
    assert first.status_code == 200
    second = client.patch(
        "/attendance/EMP0001/2026-07-07",
        json={"status": "WFH", "reason": "second correction", "regularized_by": "hr.two"},
    )
    assert second.status_code == 200
    history = second.json()["history"]
    assert len(history) == 2
    assert [entry["by"] for entry in history] == ["hr.one", "hr.two"]
    assert set(history[1]["changes"]) == {"status"}


def test_regularizing_a_legacy_record_without_history(client):
    make_employee(client)
    import app.main as main

    main.attendance.insert_one(
        {
            "emp_code": "EMP0001",
            "date": "2026-07-14",
            "status": "PRESENT",
            "punch_in": None,
            "punch_out": None,
            "work_hours": 8.98,
            "late_minutes": 0,
            "overtime_minutes": 0,
        }
    )
    response = client.patch(
        "/attendance/EMP0001/2026-07-14",
        json={"status": "LEAVE", "reason": "retro leave approval", "regularized_by": "hr.admin"},
    )
    assert response.status_code == 200
    assert len(response.json()["history"]) == 1
