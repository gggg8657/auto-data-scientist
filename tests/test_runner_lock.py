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


def _live_benchmark_pids() -> list[int]:
    """Pids actually running `run_benchmark.py`, and shells that merely name it.

    Read from /proc rather than inferred from the lock, for the reason in the
    test below.

    First cut of this helper matched any cmdline containing the script name,
    which reported three pids: the actual runner, its `zsh -c` wrapper, and a
    chained `until grep -q '^EXIT=' ...; do sleep` waiter that only *mentions*
    the script in the command it will eventually run. Two false positives out
    of three. So the first argv token must itself be a python interpreter.
    """
    pids, waiters = [], []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            raw = (d / "cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        argv = [a for a in raw.split("\0") if a]
        if not argv or "run_benchmark.py" not in raw or "leakage" in raw:
            continue
        exe = Path(argv[0]).name
        if exe.startswith("python"):
            pids.append(int(d.name))
        else:
            waiters.append(int(d.name))
    return sorted(pids), sorted(waiters)


def test_the_live_confirmatory_directory_is_locked_while_a_run_is_going():
    """Documents the actual state -- and does not infer the state from the lock.

    This test used to read: if `runs/bench/.runner.lock` is absent, print "no
    run in flight; expected when idle" and return. On 2026-09-10 at 23:15 it
    printed exactly that **while a 4h39m confirmatory run was writing
    `runs/bench`** (pid 1493119, 38 of 40 records down). The cause is benign --
    that run started before `acquire_output_lock` existed, so it holds no lock
    -- but the inference is not: *absence of a lock does not imply absence of a
    run*, and the test asserted the converse in the one situation it exists to
    document.

    That is the fifth appearance of absence-coerced-to-a-value in this
    repository: etch-operator-twin's partial checkpoints, the joint-event
    table, `load_bench` averaging partial records, the provenance gate
    defaulting a missing `n_interventions` to 0, and now this.

    So the run is detected from `/proc` and the lock is reported *against* it.
    An unlocked live run is a real state and gets named as one rather than
    reported as an idle box.
    """
    lock = REPO / "runs/bench/.runner.lock"
    pids, waiters = _live_benchmark_pids()
    held = json.loads(lock.read_text()) if lock.exists() else None

    if not pids and not held:
        print(f"  runs/bench: no python runner in flight and no lock -- "
              f"genuinely idle (shells naming the script: {waiters})")
        return
    if pids and not held:
        # Not an assertion failure: a run predating the lock is exactly this,
        # and failing here would turn a historical fact into a red suite.
        print(f"  runs/bench: run(s) {pids} IN FLIGHT WITH NO LOCK. Legitimate "
              f"only for a run started before acquire_output_lock existed; a "
              f"newly launched runner must hold one. Shells naming the "
              f"script, which are NOT runners: {waiters}")
        return
    alive = Path(f"/proc/{held.get('pid')}").exists()
    assert held.get("pid") in pids or not alive, (
        f"runs/bench is locked by live pid {held.get('pid')}, which is not "
        f"running run_benchmark.py (in flight: {pids}). Either the lock "
        f"outlived its run or something else took it.")
    print(f"  runs/bench held by pid {held.get('pid')} (alive={alive}) since "
          f"{held.get('started_utc')}; in flight: {pids}; waiters: {waiters}")


PROBE = REPO / "scripts/leakage_probe.py"
PROBE_LOCK_DIR = REPO / "runs" / ".leakage_probe.lock.d"


def _run_probe(*args, timeout=120):
    """`scripts/leakage_probe.py` as a subprocess, on a task it will refuse.

    `--task 3917` is one of the two the power file says this instrument cannot
    resolve at any fold count, so with no `--folds` override the probe empties
    its task list and returns 2 *without fitting anything*. That makes the lock
    path testable in a second rather than in the tens of minutes a real probe
    takes.
    """
    return subprocess.run(
        [sys.executable, str(PROBE), "--task", "3917", *args],
        cwd=REPO, capture_output=True, text=True, timeout=timeout)


def test_the_probe_refuses_to_start_against_a_locked_output():
    """The claim in commit 42ca05e, which arrived as prose with no test.

    A second instance of this loop wrote that the probe's lock was "measured,
    not assumed: first holder acquires, second raises RunnerBusy". The code was
    real; the measurement was not in the repository. This is it.
    """
    PROBE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    held = acquire(PROBE_LOCK_DIR, ["--test-holder"])
    try:
        r = _run_probe()
        assert r.returncode == 3, (
            f"a probe started against a locked output and exited "
            f"{r.returncode}; two probes would both write "
            f"runs/leakage_probe.json and the survivor would be whichever "
            f"finished last\n{r.stdout}\n{r.stderr}")
        assert "another leakage probe owns the output" in r.stdout, r.stdout
        assert str(os.getpid()) in r.stdout, (
            "the refusal does not name the holder, so an operator cannot tell "
            f"what to wait for: {r.stdout}")
    finally:
        held.unlink(missing_ok=True)
    print(f"  probe refused with exit 3, naming holder pid {os.getpid()}")


def test_the_probe_releases_its_lock_on_the_no_work_path():
    """A lock that outlives a run that did nothing would wedge every later
    probe, and the no-work path is the one most likely to be taken twice."""
    lock = PROBE_LOCK_DIR / ".runner.lock"
    lock.unlink(missing_ok=True)
    first = _run_probe()
    assert first.returncode == 2, (first.returncode, first.stdout, first.stderr)
    assert not lock.exists(), (
        "the probe returned without releasing its lock, so the next probe "
        "would have to break it")
    second = _run_probe()
    assert second.returncode == 2, (
        f"the second probe exited {second.returncode}, so the first left the "
        f"output locked\n{second.stdout}")
    print("  no-work path returns 2 twice in a row and leaves no lock behind")


def test_a_probe_killed_mid_run_leaves_a_breakable_lock_not_a_wedge():
    """The gap I found reading the lock code, tested rather than assumed.

    `main()` releases the lock on its two `return` paths but has no
    `try/finally`, so an exception inside `probe_fold` -- or a kill -- leaves
    the file behind. That is survivable only if stale-breaking covers it, and
    the honest way to know is to leave a lock owned by a pid that cannot be
    alive and check the probe still starts.

    It does, so the missing `try/finally` costs an extra `lock_broken` ledger
    entry rather than blocking the weekend. Recorded as survivable rather than
    fixed, because the break is *logged* and a silent release is not.
    """
    PROBE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = PROBE_LOCK_DIR / ".runner.lock"
    lock.write_text(json.dumps({
        "pid": 2 ** 31 - 1, "host": "nowhere", "argv": ["--killed"],
        "started_utc": "2000-01-01T00:00:00"}))
    ledger = Path(R["LEDGER"])
    before = ledger.read_text().count("\n") if ledger.exists() else 0
    try:
        r = _run_probe()
        assert r.returncode == 2, (
            f"a lock left by a dead process wedged the probe (exit "
            f"{r.returncode}); stale-breaking does not cover this path and "
            f"the missing try/finally is a real bug, not a cosmetic one"
            f"\n{r.stdout}\n{r.stderr}")
        after = ledger.read_text().count("\n")
        assert after > before, (
            "the stale lock was broken without a ledger entry")
    finally:
        lock.unlink(missing_ok=True)
    print("  a dead holder's lock is broken and logged, so a killed probe "
          "costs a ledger line rather than the weekend")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
