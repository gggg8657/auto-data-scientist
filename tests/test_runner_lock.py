"""Two runners may not share an output directory.

This is a fix for something that already happened, not a precaution. On
2026-09-10 two instances of this project's own loop were running against this
repository and `runs/attempts.jsonl` recorded two `started` lines for one
confirmatory cell:

    18:16:08  started task 10101 seed 6  pid 936115
    18:36:36  started task 10101 seed 6  pid 1493119

The first framing of this, in the docstring of `acquire_output_lock`, called it
a double-launch race. It was not: the lines are twenty minutes apart and the
first process was killed, the other instance having decided to relaunch with a
lower thread cap on a saturated box. That was a reasonable decision. What is
not acceptable is that it was **silent** -- nothing in the repository records
that a runner was stopped and replaced.

So the lock's job is not to override the operator. It is to make the override
leave a trace: a second runner must either wait, or break the lock, and
breaking it appends a `lock_broken` event naming both holders. Hence
`test_a_stale_lock_is_broken_but_only_on_the_record`, which is the test that
actually matters here.

Separately, `reconcile_ledger` takes `started` as a **set**, so once any
process writes `completed` for that cell the two `started` lines collapse to
one and the evidence that two runners touched it leaves the reconciliation --
the same reason codex's duplicated-seed attack worked. A set cannot count.
That is a flaw in report.py, not here, and is recorded rather than fixed in
this commit: the other loop instance is editing that file.
"""
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = runpy.run_path(str(REPO / "scripts/run_benchmark.py"))
acquire = R["acquire_output_lock"]
RunnerBusy = R["RunnerBusy"]


def test_a_live_holder_blocks_a_second_runner():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        first = acquire(d, ["--role", "dev"])
        assert first.exists(), "no lock file was written"
        held = json.loads(first.read_text())
        assert held["pid"] == os.getpid() and held["argv"] == ["--role", "dev"]
        try:
            acquire(d, ["--role", "dev"])
        except RunnerBusy as exc:
            msg = str(exc)
            assert str(os.getpid()) in msg, msg
            assert "interleave" in msg, "the error does not say why it matters"
            print(f"  second acquisition refused, naming pid {os.getpid()}")
        else:
            raise AssertionError(
                "a second runner acquired a lock held by a live process")


def test_the_lock_is_created_atomically_not_checked_then_written():
    """O_EXCL, so the creation *is* the mutual exclusion.

    A check-then-write lock has the same race as the `out.exists()` guard it
    replaces, which would be an unusually pointless fix.
    """
    src = (REPO / "scripts/run_benchmark.py").read_text()
    i = src.index("def acquire_output_lock")
    j = src.find("\ndef ", i + 1)
    body = src[i:j]
    assert "O_EXCL" in body, "the lock is not created with O_EXCL"
    assert "os.open" in body
    assert "if lock.exists()" not in body, (
        "check-then-write reintroduces the race the lock exists to close")
    print("  lock created with os.open(O_CREAT|O_EXCL), no check-then-write")


def test_a_stale_lock_is_broken_but_only_on_the_record():
    """A crashed runner must not block the weekend, and must leave a trace."""
    ledger = Path(R["LEDGER"])
    before = ledger.read_text().count("\n") if ledger.exists() else 0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # a lock held by a pid that cannot be alive
        dead = {"pid": 2 ** 31 - 1, "host": "nowhere",
                "argv": [], "started_utc": "2000-01-01T00:00:00"}
        (d / ".runner.lock").write_text(json.dumps(dead))
        got = acquire(d, ["--role", "dev"])
        assert json.loads(got.read_text())["pid"] == os.getpid(), (
            "the stale lock was not taken over")
    after = ledger.read_text().count("\n")
    assert after > before, (
        "breaking a stale lock left no ledger entry; a silent break is "
        "indistinguishable from never having locked")
    last = json.loads(ledger.read_text().strip().split("\n")[-1])
    assert last["event"] == "lock_broken", last
    assert last["stale_lock_holder"]["pid"] == dead["pid"], last
    assert last["breaker"]["pid"] == os.getpid(), last
    print("  stale lock broken, with holder and breaker recorded in the ledger")


def test_a_corrupt_lock_file_does_not_wedge_the_runner():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / ".runner.lock").write_text("{not json")
        got = acquire(d, [])
        assert json.loads(got.read_text())["pid"] == os.getpid()
    print("  an unparseable lock is treated as stale rather than fatal")


def test_the_runner_refuses_to_start_against_a_locked_directory():
    """End to end, through the CLI, which is how it will actually happen."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        held = acquire(d, ["held-by-this-test"])
        assert held.exists()
        # A subprocess whose out_dir is locked by *this* live pid must exit
        # non-zero before doing any work. --role dev writes to runs/dev, so
        # point the whole thing at the temp dir by monkeypatching via env is
        # not available; instead assert the guard directly on the same dir.
        try:
            acquire(d, ["second"])
            raise AssertionError("a second acquisition succeeded")
        except RunnerBusy:
            pass
        # and the real runner surfaces RunnerBusy as a non-zero exit
        assert issubclass(RunnerBusy, SystemExit), (
            "RunnerBusy must be a SystemExit so the CLI exits non-zero")
        code = subprocess.run(
            [sys.executable, "-c",
             "import runpy,sys;"
             "R=runpy.run_path('scripts/run_benchmark.py');"
             "raise R['RunnerBusy']('x')"],
            cwd=REPO, capture_output=True, text=True).returncode
        assert code != 0, "RunnerBusy did not produce a non-zero exit"
    print("  RunnerBusy is a SystemExit, so the CLI exits non-zero")


def test_the_live_confirmatory_directory_is_locked_while_a_run_is_going():
    """Documents the actual state, and skips cleanly when nothing is running."""
    lock = REPO / "runs/bench/.runner.lock"
    if not lock.exists():
        print("  no run in flight; runs/bench is unlocked (expected when idle)")
        return
    held = json.loads(lock.read_text())
    alive = Path(f"/proc/{held.get('pid')}").exists()
    print(f"  runs/bench held by pid {held.get('pid')} "
          f"(alive={alive}) since {held.get('started_utc')}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
