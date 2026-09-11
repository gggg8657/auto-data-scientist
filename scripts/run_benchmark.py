"""Stage 2 — run the agent on the five tasks `runs/baselines.json` selected.

The task list comes from the frozen baselines file, never from the command line
by default: the five tasks were chosen before this script existed, and reading
them from the file is what stops a later turn from quietly swapping a task that
went badly for one that went well.  `--tasks` exists for debugging and stamps
`off_registry: true` into the output so such a run can never be mistaken for a
benchmark run.

One process per task, so a task that crashes does not take the others with it,
and so `runs/bench/task_<tid>.json` is complete or absent, never half-written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
BENCH = RUNS / "bench"


def write_json_atomic(path: Path, obj) -> None:
    """Write a run record so it is complete or absent, never half-written.

    The docstring above claimed that guarantee and did not have it.  On
    2026-09-10 the volume hit 100% (0 bytes free on a 7.0T ext4 shared with
    other work on this box) and `Path.write_text` left a 4096-byte prefix of a
    run record on disk -- valid-looking JSON up to a truncation point, which
    `json.loads` then refused and which a less strict reader would have
    accepted as a short record.  `write_text` raising is not enough: the
    partial file survives the exception.

    So: serialise fully in memory, write to `.part`, fsync (ENOSPC surfaces
    here rather than at some later flush), rename, and unlink the `.part` if
    anything failed.  Rename within a directory is atomic, so a reader sees
    either the old file or the whole new one.
    """
    text = json.dumps(obj, indent=2)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


LEDGER = RUNS / "attempts.jsonl"


def register_attempt(**fields) -> None:
    """Append-only record of every attempt, written BEFORE the work starts.

    codex, asked how it would fake this KPI, ranked this fourth: move the
    unfavourable run files out of `runs/bench/` and regenerate the documents.
    Every remaining file is genuine, every hash still checks out, and
    `test_results_md_matches_what_report_regenerates` faithfully reproduces
    the curated collection. A process killed before it wrote anything
    disappears just as cleanly.

    A report can only notice that if the attempt was recorded before its
    outcome was known. So each (task, seed) is appended here as `started`
    before `run_task` is called, and again as `completed` or `failed`
    afterwards; `report.py` reconciles the ledger against the files on disk
    and a `started` with no terminal record, or a `completed` whose run file
    is missing, is visible.

    This is append-only by convention, not by permission -- anyone able to
    delete a run file can delete a line here too. What it buys is that hiding
    a result now takes two coordinated edits instead of one `mv`, and that the
    reconciliation is printed in RESULTS.md where a reader sees it.
    """
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), **fields}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class RunnerBusy(SystemExit):
    """Another runner already owns this output directory."""


def acquire_output_lock(out_dir: Path, argv: list[str]) -> Path:
    """Refuse to start if another runner owns `out_dir`. Not a precaution.

    The runner's only guard against re-running a cell was `out.exists()`,
    checked *before* a fold loop that takes minutes to hours. Two runners
    started against the same role therefore both pass that check for the same
    (task, seed) and both proceed. `write_json_atomic` guarantees each record
    is whole; it guarantees nothing about *which* process's record survives,
    while both append `started`/`completed` lines to one shared ledger.

    This is not hypothetical. On 2026-09-10 two instances of this project's own
    loop were running against this repository and `runs/attempts.jsonl` records
    two `started` lines for one confirmatory cell:

        18:16:08  started task 10101 seed 6  pid 936115
        18:36:36  started task 10101 seed 6  pid 1493119

    An earlier version of this docstring called that a double-launch race. It
    was not, and the correction matters more than the original claim: the two
    are twenty minutes apart, and the first process was *killed* -- the other
    loop instance decided, reasonably, to relaunch with a lower thread cap on a
    box at load 360. So the real sequence is a deliberate replacement, and what
    the ledger shows is the gap named in `critique_log.md` the turn before: the
    runner cannot express an operator interruption, so a killed attempt leaves
    a bare `started` indistinguishable from the deletion the ledger exists to
    catch.

    The lock is still the right fix, for a reason the corrected story makes
    sharper. The replacement was *silent*: nothing in the repository records
    that a runner was stopped and another started in its place. With a lock,
    the second runner could not have started without either waiting or
    breaking the lock, and breaking it appends a `lock_broken` event naming
    both holders. The lock does not prevent the operator's decision; it
    prevents the decision from going unrecorded.

    Independently, and this part was right: `reconcile_ledger` takes `started`
    as a *set*, so once some process writes `completed` for that cell both
    `started` lines collapse to one and the evidence that two runners touched
    it disappears from the reconciliation -- for exactly the reason codex's
    duplicated-seed attack worked. A set cannot count. That is a flaw in
    report.py rather than here, and it is recorded as such.

    So: an exclusive lock per output directory, created with `O_EXCL` so the
    creation itself is the atomic operation rather than a check followed by a
    write. It records who holds it. A lock whose pid is gone is stale and is
    broken with the reason appended to the ledger, because a crashed runner
    must not block the weekend; a lock whose pid is alive refuses to start.
    """
    lock = out_dir / ".runner.lock"
    me = {"pid": os.getpid(), "host": platform.node(),
          "argv": argv,
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
    for attempt in (1, 2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                held = json.loads(lock.read_text())
            except (OSError, json.JSONDecodeError):
                held = {}
            pid = held.get("pid")
            alive = False
            if isinstance(pid, int):
                alive = Path(f"/proc/{pid}").exists()
            if alive and attempt == 1:
                raise RunnerBusy(
                    f"another runner owns {out_dir}: pid {pid} on "
                    f"{held.get('host')} since {held.get('started_utc')}, "
                    f"argv {held.get('argv')}. Two runners on one output "
                    "directory interleave records for the same (task, seed) "
                    "and both write to one ledger. Wait for it, or use a "
                    "different --role. If you are certain it is dead, remove "
                    f"{lock}.")
            # stale: the holder is gone. Break it, on the record.
            register_attempt(event="lock_broken", role=None,
                             stale_lock_holder=held, breaker=me,
                             reason="lock file present but holder pid is not "
                                    "alive; a crashed runner must not block "
                                    "the queue")
            lock.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(me, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            return lock
    raise RunnerBusy(f"could not acquire {lock} after breaking a stale lock")


def agent_source_digest() -> dict:
    """Pin the exact agent code a run was produced by.

    `git rev-parse HEAD` alone does not do this. codex, asked how it would
    fake this KPI, named it attack #5: look at the confirmatory scores, adjust
    a generic threshold or a search space in `ads/`, re-run, and report --
    every run is a real run with zero logged interventions, and HEAD is
    recorded but the working tree it actually ran is not. A dirty tree, or two
    runs produced by different versions of `ads/`, is then invisible.

    So each run carries a digest over the sorted contents of the package, and
    `report.py` can refuse a set of runs that do not agree on it.
    """
    h = hashlib.sha256()
    files = sorted((REPO / "ads").glob("*.py"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "ads"],
                           cwd=REPO, capture_output=True, text=True).stdout
    return {"ads_sha256": h.hexdigest(),
            "ads_files": [f.name for f in files],
            "ads_dirty_vs_head": bool(dirty.strip()),
            "ads_dirty_detail": dirty.strip() or None}


def thread_regime() -> dict:
    """What parallelism this process was actually given.

    Added 2026-09-11 turn 12, because a record could not answer the question it
    was being asked.  `runs/clean` reproduced `runs/bench` exactly on three
    cells, and the natural follow-up -- did the operator's thread change at turn
    8 move any result? -- turned out to be unanswerable from the artifacts:
    every record carried `cpu_count: 192` and nothing else, so no record in this
    repository could say what parallelism produced it.  `ADS_N_JOBS` is not the
    answer either; it reaches only joblib, while `HistGradientBoostingClassifier`
    is OpenMP-only and takes no `n_jobs` at all.

    This does not make the historical records answerable -- they stay
    `[not measured]`.  It stops the next one from having the same hole.

    `affinity_count` and `loadavg_1min` are the machine's state at the moment
    the run started; they are recorded because this repository has already seen
    one fold go from 25-40s to 1341s on external CPU contention, and a run that
    cannot report the load it met cannot be compared with one that can.
    """
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ADS_N_JOBS")
    # Absent is recorded as None, never coerced to a default that was not set.
    env_vars = {n: os.environ.get(n) for n in names}
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = None
    pools = None
    try:
        import threadpoolctl
        pools = [{k: v for k, v in p.items()
                  if k in ("user_api", "internal_api", "num_threads", "prefix")}
                 for p in threadpoolctl.threadpool_info()]
    except Exception:
        pools = None      # not installed here; absent, not zero
    return {
        "thread_env": env_vars,
        "thread_env_all_unset": all(v is None for v in env_vars.values()),
        "affinity_count": affinity,
        "threadpools": pools,
        "loadavg_1min_at_start": load1,
    }


def environment() -> dict:
    import numpy, pandas, sklearn, openml
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "openml": openml.__version__,
        "cpu_count": os.cpu_count(),
        **thread_regime(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip() or "uncommitted",
        **agent_source_digest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--role",
                    choices=("confirmatory", "dev", "successor", "clean"),
                    default="confirmatory",
                    help="confirmatory = the five registered tasks (ranks 1-5) "
                         "-> runs/bench; dev = ranks 6-15, where the agent may "
                         "be iterated on and where debug runs belong -> "
                         "runs/dev; successor = a labelled second measurement "
                         "on a different task set -> runs/successor, kept "
                         "apart from dev so a debug run cannot be mistaken "
                         "for it")
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="debug only; stamps off_registry into the output")
    ap.add_argument("--seed", type=int, default=None,
                    help="one seed; kept for debugging a single cell")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="seeds to run. Defaults to the registry's "
                         "run_protocol.seeds_screen, so the pre-registered "
                         "screen is what runs when nobody passes anything. "
                         "Escalating to run_protocol.seeds_verdict is an "
                         "explicit act: --seeds 0 1 2 3 4 5 6 7")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from ads.evaluate import run_task

    raw = Path(args.baselines).read_text()
    base = json.loads(raw)
    # The digest of the exact registry this run was measured against. Editing a
    # baseline afterwards changes this, so the run no longer matches its target
    # instead of the target silently moving underneath it.
    registry_sha = hashlib.sha256(raw.encode()).hexdigest()

    registry = list(base["selected_task_ids"])
    dev_tasks = [t["task_id"] for t in
                 sorted(base["tasks"], key=lambda t: t["rank_by_n_runs"])
                 if not t["selected"]][:10]          # ranks 6..15
    default = (registry if args.role in ("confirmatory", "clean")
               else dev_tasks)
    task_ids = args.tasks if args.tasks else default
    off_registry = bool(args.tasks) and sorted(task_ids) != sorted(default)

    # The seed set is the registry's, not this script's default: the screen was
    # pre-registered as run_protocol.seeds_screen and escalation to
    # seeds_verdict is a decision a human makes on the record, not a flag that
    # quietly grows the sample until a task crosses the line.
    protocol = base.get("run_protocol") or {}
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(protocol.get("seeds_screen") or [0])

    # `dev` is the protocol's development role and is where debug runs land.
    # A *labelled second measurement* must not share a directory with debug
    # runs: on 2026-09-10 a concurrent instance of this loop wrote a
    # `--max-folds 1` record for task 37 into runs/dev while a successor
    # measurement was queued to write there, and since the runner skips
    # existing files that one-fold record would have stood in for the real run.
    # report.py now excludes partial records from any average, but the cheaper
    # protection is not to mix the two roles in one directory.
    # `clean`, added 2026-09-11 turn 11. The confirmatory set at runs/bench is
    # complete and its numbers stand, but its ledger records one operator kill
    # and one twice-started cell -- (10101, seed 6) -- so under the brief's own
    # rule ("count any manual intervention as a failure of that run rather than
    # editing it out") clause 3 reads False on it. This role produces the same
    # registered measurement with no operator touching any cell. It writes to
    # its OWN directory precisely so that it cannot overwrite the set it is
    # meant to be compared against: the old records and the old ledger stay
    # exactly where they are, and both sets are reported.
    out_dir = (BENCH if args.role == "confirmatory"
               else RUNS / {"successor": "successor",
                            "clean": "clean"}.get(args.role, "dev"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to share an output directory with another runner. See
    # acquire_output_lock: this repository has a ledger recording two
    # different pids claiming one confirmatory cell.
    lock = acquire_output_lock(out_dir, sys.argv[1:])
    env = environment()
    print(f"role={args.role}  tasks={task_ids}  seeds={seeds}  "
          f"registry_sha={registry_sha[:12]}  lock={lock.name}", flush=True)
    try:
      for seed in seeds:
          for tid in task_ids:
              out = out_dir / f"task_{tid}_seed{seed}.json"
              if out.exists() and not args.force:
                  print(f"task {tid} seed {seed}: {out.name} exists, skipping",
                        flush=True)
                  continue
              print(f"=== task {tid} (seed {seed}) ===", flush=True)
              t0 = time.time()
              register_attempt(event="started", task_id=tid, seed=seed,
                               role=args.role, pid=os.getpid(),
                               registry_sha256=registry_sha,
                               ads_sha256=env.get("ads_sha256"),
                               max_folds=args.max_folds,
                               off_registry=off_registry,
                               out_file=out.name)
              try:
                  rec = run_task(tid, random_state=seed, max_folds=args.max_folds)
              except Exception as e:
                  print(f"task {tid} seed {seed} FAILED: {type(e).__name__}: {e}",
                        flush=True)
                  write_json_atomic(
                      out_dir / f"task_{tid}_seed{seed}.FAILED.json",
                      {"task_id": tid, "seed": seed,
                       "role": args.role,
                       "error": f"{type(e).__name__}: {e}",
                       "registry_sha256": registry_sha,
                       "env": env})
                  register_attempt(event="failed", task_id=tid, seed=seed,
                                   role=args.role,
                                   error=f"{type(e).__name__}: {e}"[:300],
                                   seconds=round(time.time() - t0, 1),
                                   out_file=out.name)
                  continue
              rec["env"] = env
              rec["role"] = args.role
              rec["registry_sha256"] = registry_sha
              rec["off_registry"] = off_registry
              rec["max_folds"] = args.max_folds
              # A partial run is not a benchmark run; the report must see it.
              rec["complete"] = args.max_folds is None
              write_json_atomic(out, rec)
              register_attempt(event="completed", task_id=tid, seed=seed,
                               role=args.role,
                               accuracy_pooled=rec["accuracy_pooled"],
                               n_folds_run=rec["n_folds_run"],
                               n_interventions=rec["n_interventions"],
                               seconds=round(time.time() - t0, 1),
                               out_file=out.name)
              print(f"task {tid} seed {seed}: pooled acc "
                    f"{rec['accuracy_pooled']:.4f} over {rec['n_folds_run']} "
                    f"folds, {time.time()-t0:.0f}s -> {out.name}", flush=True)
    finally:
        # Release only our own lock: if a stale-lock break handed ownership to
        # someone else mid-run, unlinking theirs would be worse than leaking.
        try:
            if json.loads(lock.read_text()).get("pid") == os.getpid():
                lock.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
