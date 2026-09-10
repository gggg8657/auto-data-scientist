"""A set cannot count, and the ledger needs to.

`reconcile_ledger` compared **sets** of `(task, seed)`. So two `started` lines
for one cell collapsed into one, and the moment any process wrote `completed`
the evidence that two runners had touched that cell left the reconciliation
entirely. This is the same defect codex found in the seed accounting — "a set
cannot see a duplicate" — reappearing one layer down, in code written after
that lesson.

It is not hypothetical here: this repository's ledger holds two `started` lines
for task 10101 seed 6 (pids 936115 and 1493119, twenty minutes apart, the first
stopped and replaced by the other instance of this project's loop).

The second half is the `killed` event. The ledger could say `started`,
`completed`, `failed` — so an operator stopping a run left a bare `started`,
indistinguishable from the deletion the ledger exists to catch. `killed` is
counted as terminal, and the reason that is information rather than forgiveness
is pinned below: `scripts/record_kill.py` refuses to record a kill for a live
pid, refuses one for an attempt with no `started` line, refuses to double-record,
and the cell still has to be re-run before anything counts it as done.
"""
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = runpy.run_path(str(REPO / "scripts/report.py"))
reconcile = R["reconcile_ledger"]


def _ledger(d: Path, events) -> Path:
    p = d / "attempts.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def _run(tid, seed=0):
    return {"task_id": tid, "random_state": seed, "accuracy_pooled": 0.75,
            "complete": True, "off_registry": False, "n_interventions": 0,
            "registry_sha256": "s", "_file": f"task_{tid}_seed{seed}.json"}


def _ev(event, tid=31, seed=0, pid=1, role="confirmatory"):
    return {"event": event, "task_id": tid, "seed": seed, "pid": pid,
            "role": role, "out_file": f"task_{tid}_seed{seed}.json"}


def test_two_starts_and_one_completion_is_not_reconciled():
    """The exact shape the set reading hid."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [_ev("started", pid=1), _ev("started", pid=2),
                          _ev("completed", pid=2)])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["attempts_started_but_unresolved"] == [], (
            "the set reading has nothing to say here, which is the point")
        assert out["cells_with_more_starts_than_terminal_records"] == [
            [31, 0, 2, 1]], out
        assert out["cells_started_more_than_once"] == [[31, 0, 2]], out
        assert out["reconciled"] is False, (
            "an abandoned attempt was reconciled away because another attempt "
            "at the same cell finished")
    print("  2 starts / 1 terminal is visible and does not reconcile")


def test_a_recorded_kill_resolves_the_attempt_it_names():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [_ev("started", pid=1), _ev("killed", pid=1),
                          _ev("started", pid=2), _ev("completed", pid=2)])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["n_killed_events"] == 1
        assert out["cells_with_more_starts_than_terminal_records"] == [], out
        assert out["reconciled"] is True, out
        # multiplicity is still surfaced -- two runners did touch the cell
        assert out["cells_started_more_than_once"] == [[31, 0, 2]], out
    print("  a killed + a completed resolves 2 starts, and the multiplicity "
          "is still reported")


def test_multiplicity_alone_does_not_sink_the_clause():
    """A kill and re-run is legitimate; this repo has one.

    If multiplicity alone sank the reconciliation, a stopped-and-rerun cell
    could never be clean again, which would push an operator toward hiding the
    stop rather than recording it -- the opposite of what the ledger is for.
    """
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [_ev("started", pid=1), _ev("killed", pid=1),
                          _ev("started", pid=2), _ev("completed", pid=2)])
        out = reconcile(led, {31: [_run(31)]}, [])
        assert out["cells_started_more_than_once"] and out["reconciled"] is True
    print("  reported, not punished")


def test_a_killed_cell_that_was_never_rerun_still_counts_as_unrun():
    """The kill must not substitute for the result."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        led = _ledger(d, [_ev("started", pid=1), _ev("killed", pid=1)])
        # nothing on disk for it
        out = reconcile(led, {}, [])
        assert out["cells_with_more_starts_than_terminal_records"] == [], (
            "the kill did resolve the attempt")
        # and the cell is simply absent from the bench, which the seed
        # accounting in verdict() reports as a missing registered seed
        assert out["on_disk_but_not_in_ledger"] == []
        assert out["completed_but_missing_from_disk"] == [], out
    print("  a killed cell resolves the attempt but produces no result")


def test_record_kill_refuses_a_live_pid():
    """Otherwise it is a tool for writing off inconvenient results."""
    r = subprocess.run(
        [sys.executable, "scripts/record_kill.py", "--task-id", "1",
         "--seed", "0", "--pid", str(__import__("os").getpid()),
         "--reason", "test"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0, "recording a kill for a live pid succeeded"
    assert "still alive" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    print("  refuses to record a kill for a process that is still running")


def test_record_kill_refuses_an_attempt_that_never_started():
    r = subprocess.run(
        [sys.executable, "scripts/record_kill.py", "--task-id", "999999",
         "--seed", "0", "--pid", "2147483646", "--reason", "test"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "never recorded as beginning" in (r.stdout + r.stderr), r.stderr
    print("  refuses to end an attempt that was never recorded as beginning")


def test_the_live_ledger_state_is_what_it_claims():
    """Documents rather than asserts, and names the one real kill."""
    led = REPO / "runs/attempts.jsonl"
    bench_dir = REPO / "runs/bench"
    if not led.exists() or not bench_dir.exists():
        print("  no ledger yet; skipped")
        return
    loaded = R["load_bench"](bench_dir)
    out = reconcile(led, loaded[0], loaded[1])
    kills = [json.loads(x) for x in led.read_text().splitlines()
             if x.strip() and json.loads(x).get("event") == "killed"]
    for k in kills:
        assert k.get("reason"), k
        assert k.get("evidence", {}).get("holder_pid_alive") is False, k
    print(f"  live: reconciled={out['reconciled']} "
          f"killed={out['n_killed_events']} "
          f"more_starts_than_terminal="
          f"{out['cells_with_more_starts_than_terminal_records']} "
          f"started_more_than_once={out['cells_started_more_than_once']}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
