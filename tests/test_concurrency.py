"""Concurrency tests against a real uvicorn process and a real MongoDB.

These start the application exactly as the grader does
(`uvicorn app.main:app --port ...`) and fire genuinely parallel HTTP requests
from a thread pool, so the guarantees are exercised end to end rather than
simulated.
"""
import os
import pathlib
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from conftest import ist_ms, requires_mongo

pytestmark = requires_mongo

PARALLEL = 8
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    """A live uvicorn server on its own port, torn down at the end."""
    port = _free_port()
    env = {
        **os.environ,
        "MONGO_URI": os.environ["MONGO_URI"],
        "MONGO_DB": os.environ["MONGO_DB"],
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20  # the contract's readiness budget
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                ready = True
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    if not ready:
        process.kill()
        output = process.stdout.read().decode(errors="replace") if process.stdout else ""
        pytest.fail(f"server did not become ready within 20s\n{output}")
    yield base_url
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()


@pytest.fixture()
def wipe():
    """Clean both collections around each test in this module."""
    import app.main as main

    main.employees.delete_many({})
    main.attendance.delete_many({})
    yield
    main.employees.delete_many({})
    main.attendance.delete_many({})


def _parallel(fn, count=PARALLEL):
    """Run fn(i) count times at the same time and return the results."""
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(fn, index) for index in range(count)]
        return [future.result() for future in futures]


def _make_employee(base_url, emp_code="EMP0001", **overrides):
    body = {
        "emp_code": emp_code,
        "name": "Asha Rao",
        "email": f"{emp_code.lower()}@example.com",
        "department": "Engineering",
        "joined_on": "2026-01-05",
        **overrides,
    }
    response = httpx.post(f"{base_url}/employees", json=body, timeout=10)
    assert response.status_code == 201, response.text


# --------------------------------------------------------------------------- #
def test_parallel_employee_creates_give_exactly_one_201(server, wipe):
    def create(_):
        return httpx.post(
            f"{server}/employees",
            json={
                "emp_code": "EMP0042",
                "name": "Race Condition",
                "email": "race@example.com",
                "department": "Engineering",
                "joined_on": "2026-01-05",
            },
            timeout=15,
        ).status_code

    codes = _parallel(create)
    assert codes.count(201) == 1, codes
    assert codes.count(409) == PARALLEL - 1, codes
    assert set(codes) <= {201, 409}

    import app.main as main

    assert main.employees.count_documents({"emp_code": "EMP0042"}) == 1


def test_parallel_punch_ins_give_exactly_one_201_and_one_record(server, wipe):
    _make_employee(server)
    payload = {"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 28)}

    codes = _parallel(
        lambda _: httpx.post(f"{server}/attendance/punch-in", json=payload, timeout=15).status_code
    )
    assert codes.count(201) == 1, codes
    assert codes.count(409) == PARALLEL - 1, codes

    import app.main as main

    assert main.attendance.count_documents({"emp_code": "EMP0001", "date": "2026-07-06"}) == 1


def test_parallel_punch_outs_give_exactly_one_200(server, wipe):
    _make_employee(server)
    httpx.post(
        f"{server}/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 9, 28)},
        timeout=10,
    )
    out_at = ist_ms(2026, 7, 6, 19, 10)

    results = _parallel(
        lambda _: httpx.post(
            f"{server}/attendance/punch-out",
            json={"emp_code": "EMP0001", "punched_at": out_at},
            timeout=15,
        )
    )
    codes = [response.status_code for response in results]
    assert codes.count(200) == 1, codes
    assert codes.count(409) == PARALLEL - 1, codes

    # The winner's values are the ones that survived.
    import app.main as main

    record = main.attendance.find_one({"emp_code": "EMP0001", "date": "2026-07-06"})
    assert main.datetime_to_epoch_ms(record["punch_out"]) == out_at
    assert record["work_hours"] == 9.7
    assert record["overtime_minutes"] == 40
    assert record["history"] == []  # punch-out never writes history


def test_parallel_regularizations_never_lose_a_history_entry(server, wipe):
    """Concurrent corrections either all apply or lose with a 409, but the
    history of the ones that succeeded is never silently overwritten.
    """
    _make_employee(server)
    httpx.post(
        f"{server}/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 10, 5)},
        timeout=10,
    )

    def correct(index):
        return httpx.patch(
            f"{server}/attendance/EMP0001/2026-07-06",
            json={
                # Each actor moves the punch-in to a different minute, so every
                # request is a real change.
                "punch_in": ist_ms(2026, 7, 6, 9, 30 + index),
                "reason": f"correction number {index}",
                "regularized_by": f"hr.{index}",
            },
            timeout=15,
        ).status_code

    codes = _parallel(correct)
    successes = codes.count(200)
    assert successes >= 1, codes
    assert set(codes) <= {200, 409}, codes

    import app.main as main

    record = main.attendance.find_one({"emp_code": "EMP0001", "date": "2026-07-06"})
    # Exactly one entry per successful correction: none was lost or duplicated.
    assert len(record["history"]) == successes
    actors = [entry["by"] for entry in record["history"]]
    assert len(set(actors)) == successes


def test_sequential_corrections_under_load_keep_every_entry(server, wipe):
    """Ten corrections in a row, each retried on 409, must leave ten entries."""
    _make_employee(server)
    httpx.post(
        f"{server}/attendance/punch-in",
        json={"emp_code": "EMP0001", "punched_at": ist_ms(2026, 7, 6, 10, 0)},
        timeout=10,
    )
    for index in range(10):
        for _ in range(5):  # retry the compare-and-set a few times if it loses
            response = httpx.patch(
                f"{server}/attendance/EMP0001/2026-07-06",
                json={
                    "punch_in": ist_ms(2026, 7, 6, 9, 30 + index),
                    "reason": f"correction {index}",
                    "regularized_by": f"hr.{index}",
                },
                timeout=15,
            )
            if response.status_code == 200:
                break
        assert response.status_code == 200, response.text

    import app.main as main

    record = main.attendance.find_one({"emp_code": "EMP0001", "date": "2026-07-06"})
    assert len(record["history"]) == 10
    assert [entry["by"] for entry in record["history"]] == [f"hr.{i}" for i in range(10)]


def test_mixed_parallel_traffic_never_returns_500(server, wipe):
    """A burst of valid and invalid requests must never produce a 500."""
    _make_employee(server, "EMP0100")

    requests = [
        ("GET", "/health", None),
        ("GET", "/employees?page=1&page_size=5", None),
        ("GET", "/attendance?page=1", None),
        ("GET", "/attendance?date_from=2026-07-10&date_to=2026-07-01", None),
        ("GET", "/analytics/employees/EMP0100/monthly?month=2026-07", None),
        ("GET", "/analytics/employees/NOPE/monthly?month=2026-07", None),
        ("GET", "/analytics/departments/summary?month=2026-07", None),
        ("GET", "/analytics/leaderboard/late?month=2026-07", None),
        ("GET", "/analytics/departments/Engineering/trend?from=2026-07-01&to=2026-07-31", None),
        ("GET", "/analytics/departments/Nope/trend?from=2026-07-01&to=2026-07-31", None),
        ("GET", "/admin/explain/attendance_list", None),
        ("GET", "/admin/explain/employee_monthly?month=2026-07", None),
        ("POST", "/attendance/punch-in", {"emp_code": "EMP0100"}),
        ("POST", "/attendance/punch-in", {"emp_code": "EMP0100", "punched_at": 1.5}),
        ("POST", "/attendance/punch-out", {"emp_code": "NOPE"}),
        ("PATCH", "/attendance/EMP0100/2026-07-06", {"reason": "x", "regularized_by": "y"}),
    ]

    def fire(index):
        method, path, body = requests[index % len(requests)]
        return httpx.request(method, f"{server}{path}", json=body, timeout=20).status_code

    codes = _parallel(fire, count=len(requests) * 2)
    assert all(code < 500 for code in codes), [
        (requests[i % len(requests)][1], code) for i, code in enumerate(codes) if code >= 500
    ]
