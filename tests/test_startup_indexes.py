"""Startup index creation: it must be automatic, idempotent and resilient."""
import pytest

from conftest import requires_mongo

pytestmark = requires_mongo

EXPECTED = {
    "employees": {"uniq_emp_code", "dept_emp_code", "joined_on"},
    "attendance_logs": {"uniq_emp_date", "date_desc_emp"},
}


def _index_names(collection):
    return set(collection.index_information()) - {"_id_"}


def test_indexes_are_created_from_nothing(client):
    import app.main as main

    main.employees.drop_indexes()
    main.attendance.drop_indexes()
    assert _index_names(main.employees) == set()

    assert main.ensure_indexes() is True
    assert _index_names(main.employees) == EXPECTED["employees"]
    assert _index_names(main.attendance) == EXPECTED["attendance_logs"]


def test_index_creation_is_idempotent(client):
    import app.main as main

    assert main.ensure_indexes() is True
    before = (_index_names(main.employees), _index_names(main.attendance))
    assert main.ensure_indexes() is True  # running it again changes nothing
    assert (_index_names(main.employees), _index_names(main.attendance)) == before


@pytest.mark.parametrize(
    "collection_attr,name",
    [("employees", "uniq_emp_code"), ("attendance", "uniq_emp_date")],
)
def test_the_uniqueness_constraints_really_are_unique(client, collection_attr, name):
    import app.main as main

    main.ensure_indexes()
    info = getattr(main, collection_attr).index_information()
    assert info[name].get("unique") is True


def test_one_failing_index_does_not_block_the_others(client):
    """A unique index that cannot be built must not cost us the other four.

    Seeding two employees with the same emp_code (which only a database
    without the index could contain) makes uniq_emp_code unbuildable.
    """
    import app.main as main

    main.employees.drop_indexes()
    main.attendance.drop_indexes()
    main.employees.insert_many(
        [
            {"emp_code": "EMP0001", "name": "A", "department": "Engineering",
             "joined_on": "2026-01-05"},
            {"emp_code": "EMP0001", "name": "B", "department": "Engineering",
             "joined_on": "2026-01-05"},
        ]
    )

    assert main.ensure_indexes() is False  # reported, not swallowed
    names = _index_names(main.employees)
    assert "uniq_emp_code" not in names
    # Every other index was still created.
    assert names == EXPECTED["employees"] - {"uniq_emp_code"}
    assert _index_names(main.attendance) == EXPECTED["attendance_logs"]

    # Once the duplicate is gone, a retry completes the set.
    main.employees.delete_many({"name": "B"})
    assert main.ensure_indexes() is True
    assert _index_names(main.employees) == EXPECTED["employees"]
