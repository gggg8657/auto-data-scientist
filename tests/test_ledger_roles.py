"""One ledger serves every role; the reconciliation must not mix them.

`reconcile_ledger` builds `on_disk` from the single directory it is given, but
took `started`/`completed` over **every** line of `runs/attempts.jsonl`. So an
attempt at `--role dev`, which writes to `runs/dev/`, appeared in `completed`
and not in `on_disk` — landing in `completed_but_missing_from_disk`, which that
function's own docstring and `RESULTS.md` both describe as a **deleted
result**.

The consequence was not hypothetical. The successor measurement queued on
2026-09-10 runs fifteen cells at `--role dev`. Within minutes of it starting,
the KPI's report would have asserted that results had been deleted: a false
accusation of tampering, produced by this project's own queued job, in the one
function built to detect tampering. Every ledger line already carried the
`role` that answers it.

These tests pin the two directions. A dev attempt must not pollute the
confirmatory view, and — the part that would make the fix worthless if it
regressed — a genuinely deleted *confirmatory* result must still be caught.
"""
import json
import runpy
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = runpy.run_path(str(REPO / "scripts/report.py"))
reconcile = R["reconcile_ledger"]


def _ledger(d: Path, events) -> Path:
    p = d / "attempts.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def _run(tid, seed=0, **kw):
    r = {"task_id": tid, "random_state": seed, "accuracy_pooled": 0.75,
         "complete": True, "off_registry": False, "n_interventions": 0,
         "registry_sha256": "s", "_file": f"task_{tid}_seed{seed}.json"}
    r.update(kw)
    return r


def test_a_dev_attempt_does_not_read_as_a_deleted_confirmatory_result():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [
            {"event": "started", "task_id": 3, "seed": 0, "role": "dev"},
            {"event": "completed", "task_id": 3, "seed": 0, "role": "dev"},
            {"event": "started", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
            {"event": "completed", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
        ])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["role_reconciled"] == "confirmatory"
        assert out["completed_but_missing_from_disk"] == [], (
            "a dev-role cell was reported as a deleted confirmatory result")
        assert out["reconciled"] is True, out
        assert out["n_events"] == 2 and out["n_events_in_ledger_all_roles"] == 4
    print("  a completed dev cell leaves the confirmatory view untouched")


def test_a_genuinely_deleted_confirmatory_result_is_still_caught():
    """The check the fix must not have bought its cleanliness with."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [
            {"event": "started", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
            {"event": "completed", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
            {"event": "started", "task_id": 3913, "seed": 2,
             "role": "confirmatory"},
            {"event": "completed", "task_id": 3913, "seed": 2,
             "role": "confirmatory"},
        ])
        # task 3913 seed 2 completed per the ledger but its file is gone
        out = reconcile(led, {31: [_run(31)]}, [])
        assert [3913, 2] in out["completed_but_missing_from_disk"], out
        assert out["reconciled"] is False
    print("  a deleted confirmatory result is still reported as missing")


def test_the_dev_view_is_available_and_isolated():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [
            {"event": "started", "task_id": 3, "seed": 0, "role": "dev"},
            {"event": "completed", "task_id": 3, "seed": 0, "role": "dev"},
            {"event": "started", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
        ])
        dev = reconcile(led, {}, [], role="dev")
        assert dev["role_reconciled"] == "dev"
        assert dev["n_events"] == 2
        # the confirmatory started must not appear as an unresolved dev attempt
        assert dev["attempts_started_but_unresolved"] == [], dev
        assert [3, 0] in dev["completed_but_missing_from_disk"], (
            "asked for the dev role against an empty bench, the dev cell "
            "should read as missing")
    print("  the dev view sees only dev attempts")


def test_lines_predating_the_role_field_are_treated_as_confirmatory():
    """The only role-less lines in this repo's history were confirmatory."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [
            {"event": "started", "task_id": 31, "seed": 0},
            {"event": "completed", "task_id": 31, "seed": 0},
        ])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["n_events"] == 2, (
            "role-less legacy lines were dropped from the confirmatory view, "
            "which would hide exactly the attempts the ledger was added for")
        assert out["reconciled"] is True
    print("  role-less legacy lines stay in the confirmatory view")


def test_non_attempt_events_are_not_counted_as_attempts():
    """`lock_broken` carries role None and is not a (task, seed) attempt."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [
            {"event": "lock_broken", "role": None, "reason": "stale"},
            {"event": "started", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
            {"event": "completed", "task_id": 31, "seed": 0,
             "role": "confirmatory"},
        ])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["reconciled"] is True, out
        assert out["attempts_started_but_unresolved"] == [], out
        assert out["n_events"] == 2, out
    print("  a lock_broken event does not become a phantom attempt")


def test_the_live_ledger_reconciles_or_says_exactly_why():
    """Documents the repository's actual state rather than asserting it."""
    led = REPO / "runs/attempts.jsonl"
    bench_dir = REPO / "runs/bench"
    if not led.exists() or not bench_dir.exists():
        print("  no ledger or no runs yet; skipped")
        return
    loaded = R["load_bench"](bench_dir)
    bench, failed = loaded[0], loaded[1]
    out = reconcile(led, bench, failed)
    print(f"  live: reconciled={out['reconciled']} "
          f"events(confirmatory/all)={out['n_events']}/"
          f"{out['n_events_in_ledger_all_roles']} "
          f"unresolved={out['attempts_started_but_unresolved']} "
          f"completed_missing={out['completed_but_missing_from_disk']}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
