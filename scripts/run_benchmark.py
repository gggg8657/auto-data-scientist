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
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--tasks", type=int, nargs="*", default=None,
                    help="debug only; stamps off_registry into the output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from ads.evaluate import run_task

    base = json.loads(Path(args.baselines).read_text())
    registry = list(base["selected_task_ids"])
    task_ids = args.tasks if args.tasks else registry
    off_registry = bool(args.tasks) and sorted(task_ids) != sorted(registry)

    BENCH.mkdir(parents=True, exist_ok=True)
    env = environment()
    for tid in task_ids:
        out = BENCH / f"task_{tid}_seed{args.seed}.json"
        if out.exists() and not args.force:
            print(f"task {tid}: {out.name} exists, skipping", flush=True)
            continue
        print(f"=== task {tid} (seed {args.seed}) ===", flush=True)
        t0 = time.time()
        try:
            rec = run_task(tid, random_state=args.seed, max_folds=args.max_folds)
        except Exception as e:
            print(f"task {tid} FAILED: {type(e).__name__}: {e}", flush=True)
            (BENCH / f"task_{tid}_seed{args.seed}.FAILED.json").write_text(
                json.dumps({"task_id": tid, "seed": args.seed,
                            "error": f"{type(e).__name__}: {e}",
                            "env": env}, indent=2))
            continue
        rec["env"] = env
        rec["baselines_sha_source"] = args.baselines
        rec["off_registry"] = off_registry
        rec["max_folds"] = args.max_folds
        # A partial run is not a benchmark run; the report must be able to see it.
        rec["complete"] = args.max_folds is None
        out.write_text(json.dumps(rec, indent=2))
        print(f"task {tid}: pooled acc {rec['accuracy_pooled']:.4f} over "
              f"{rec['n_folds_run']} folds, {time.time()-t0:.0f}s -> {out.name}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
