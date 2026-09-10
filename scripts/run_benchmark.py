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
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True).stdout.strip() or "uncommitted",
        **agent_source_digest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--role", choices=("confirmatory", "dev"),
                    default="confirmatory",
                    help="confirmatory = the five registered tasks (ranks 1-5); "
                         "dev = ranks 6-15, where the agent may be iterated on")
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
    default = registry if args.role == "confirmatory" else dev_tasks
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

    out_dir = BENCH if args.role == "confirmatory" else RUNS / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    print(f"role={args.role}  tasks={task_ids}  seeds={seeds}  "
          f"registry_sha={registry_sha[:12]}", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
