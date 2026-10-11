# DECISIONS.md

**1. Indexes.** On `employees`: unique `emp_code` (identity and the duplicate
409), `{department, emp_code}` (filtered list, trend) and `{joined_on}` (R9
eligibility when the department summary is unfiltered). On `attendance_logs`:
unique `{emp_code, date}`, which both enforces one record per day and serves
every per-employee range query, and `{date: -1, emp_code: 1}`, which supplies
the list's exact sort order and the month scans. I rejected
`{emp_code, punch_in}` for punch-out: the `emp_code` prefix of the unique index
already narrows the scan to one person's rows.

**2. Punch-in race.** Both requests compute the same IST date and call
`insert_one` — no find-then-insert. The unique index lets exactly one commit;
the other raises `DuplicateKeyError`, which I turn into 409.

**3. Ties.** `$rank` is competition ranking, and `limit` is applied after it as
`$match: {rank: {$lte: limit}}`. Employees tied at the cutoff share a rank, so
both are returned and the response is longer than `limit`. Rows sort by minutes
descending, then `emp_code`.

**4. Headcount.** The pipeline starts from `employees` (`joined_on <=` month
end) and `$lookup`s the logs with `preserveNullAndEmptyArrays`, so a zero-log
employee is never dropped; headcount is the size of the distinct `emp_code` set.

**5. At 100x.** Shard `attendance_logs` on `emp_code` and precompute monthly
per-employee rollups nightly, so dashboards read one small collection instead
of rescanning raw logs.
