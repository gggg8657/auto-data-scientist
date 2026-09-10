"""Is our accuracy the same statistic as a published run's?

Evaluating on the task's own splits is necessary for comparability but not
sufficient.  OpenML reports one scalar `value` per run, and we do not get to
assume whether it is the **pooling** of predictions over all folds (size-weighted)
or the **unweighted mean** of per-fold accuracies.  Where the task's folds are
unequal the two differ, and comparing one against the other while calling it one
number would invalidate every row of this benchmark.

So it is measured, on real published runs.  Two API calls per flow:

* the scalar `value` per run — already cached by `fetch_baselines.py`;
* the per-fold array for the same runs, via `per_fold=True`, which the server
  only serves under a narrow filter (548 *Per fold queries are experimental*
  above 1000 inspected records), so we ask flow by flow and pick flows small
  enough to qualify.

Joining the two by `run_id` gives, for each run, the server's scalar beside the
per-fold scores it was computed from.  Recomputing both aggregations then says
which one the server used — a measurement, not a reading of the documentation.

Output: `runs/metric_check.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads.openml_io import configure, get_task  # noqa: E402

import openml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
TOL = 1e-6
MAX_FLOW_RUNS = 400          # stay well under the server's 1000-record ceiling


def fold_sizes(task_id: int):
    """Test-fold sizes under the task's own procedure, in (repeat, fold) order."""
    task = get_task(task_id)
    n_rep, n_fold, n_samp = task.get_split_dimensions()
    sizes = []
    for r in range(n_rep):
        for f in range(n_fold):
            for s in range(n_samp):
                _, te = task.get_train_test_split_indices(fold=f, repeat=r,
                                                          sample=s)
                sizes.append(len(te))
    return sizes


def per_fold_for_flow(task_id: int, flow_id: int, size: int = 200):
    configure()
    try:
        return openml.evaluations.list_evaluations(
            "predictive_accuracy", tasks=[task_id], flows=[flow_id], size=size,
            per_fold=True, output_format="dataframe")
    except Exception as e:
        print(f"    flow {flow_id}: {type(e).__name__} "
              f"{str(e)[:90]}", flush=True)
        return None


def check_task(task_id: int, path: Path, n_target: int, seed: int) -> dict:
    scalars = pd.read_csv(path)[["run_id", "value", "flow_id"]]
    scalars = scalars.set_index("run_id")["value"].to_dict()

    sizes = fold_sizes(task_id)
    rec: dict = {
        "task_id": int(task_id),
        "n_folds": len(sizes),
        "fold_sizes_distinct": sorted(set(sizes)),
        "folds_equal": len(set(sizes)) == 1,
        "n_scalars_cached": len(scalars),
    }

    counts = pd.read_csv(path)["flow_id"].value_counts()
    eligible = [int(f) for f in counts[counts <= MAX_FLOW_RUNS].index]
    rng = np.random.RandomState(seed)
    rng.shuffle(eligible)
    rec["n_flows_eligible"] = len(eligible)

    pairs, flows_used = [], []
    for fid in eligible:
        if len(pairs) >= n_target:
            break
        df = per_fold_for_flow(task_id, fid)
        if df is None or len(df) == 0:
            continue
        flows_used.append(fid)
        for _, row in df.iterrows():
            vals = row["values"]
            rid = int(row["run_id"])
            if not isinstance(vals, (list, tuple)) or rid not in scalars:
                continue
            if len(vals) != len(sizes):
                continue
            pairs.append((float(scalars[rid]), [float(x) for x in vals]))

    rec["n_flows_queried"] = len(flows_used)
    rec["n_runs_joined"] = len(pairs)
    if not pairs:
        rec["verdict"] = ("[not measured] — no run on this task could be joined "
                          "to a per-fold array")
        rec["aggregation"] = None
        return rec

    w = np.asarray(sizes, dtype=float)
    d_mean = [abs(float(np.mean(v)) - s) for s, v in pairs]
    d_pool = [abs(float(np.average(v, weights=w)) - s) for s, v in pairs]
    rec["max_abs_err_unweighted_mean"] = float(np.max(d_mean))
    rec["median_abs_err_unweighted_mean"] = float(np.median(d_mean))
    rec["max_abs_err_size_weighted"] = float(np.max(d_pool))
    rec["median_abs_err_size_weighted"] = float(np.median(d_pool))
    # How far apart are the two aggregations on this task at all?  If they never
    # differ by more than TOL the task simply cannot discriminate, and saying so
    # is not the same as saying they agree everywhere.
    sep = [abs(float(np.mean(v)) - float(np.average(v, weights=w)))
           for _, v in pairs]
    rec["max_separation_between_aggregations"] = float(np.max(sep))
    rec["discriminating"] = bool(np.max(sep) > TOL)

    mean_ok = rec["max_abs_err_unweighted_mean"] < TOL
    pool_ok = rec["max_abs_err_size_weighted"] < TOL
    if not rec["discriminating"]:
        rec["verdict"] = ("cannot discriminate — the two aggregations agree to "
                          f"{np.max(sep):.2e} on every run of this task")
        rec["aggregation"] = "either" if (mean_ok or pool_ok) else "unknown"
    elif mean_ok and not pool_ok:
        rec["verdict"] = "server value is the UNWEIGHTED MEAN of per-fold scores"
        rec["aggregation"] = "mean"
    elif pool_ok and not mean_ok:
        rec["verdict"] = "server value is the SIZE-WEIGHTED (pooled) accuracy"
        rec["aggregation"] = "pooled"
    elif mean_ok and pool_ok:
        rec["verdict"] = "both reproduce the server value within tolerance"
        rec["aggregation"] = "either"
    else:
        rec["verdict"] = ("NEITHER aggregation reproduces the server value — "
                          "the comparison is not justified")
        rec["aggregation"] = "unknown"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--evals", default=str(RUNS / "evals"))
    ap.add_argument("--out", default=str(RUNS / "metric_check.json"))
    ap.add_argument("--n-target", type=int, default=200,
                    help="runs to join per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-ids", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.task_ids:
        tids = args.task_ids
    else:
        tids = list(json.loads(Path(args.baselines).read_text())["selected_task_ids"])

    per_task = []
    for tid in tids:
        p = Path(args.evals) / f"task_{tid}.csv.gz"
        if not p.exists():
            print(f"task {tid}: no evaluation cache, skipping")
            continue
        print(f"checking task {tid} ...", flush=True)
        rec = check_task(tid, p, args.n_target, args.seed)
        print(f"  joined {rec.get('n_runs_joined', 0)} runs; {rec['verdict']}",
              flush=True)
        per_task.append(rec)

    discriminating = [r for r in per_task if r.get("discriminating")]
    decided = {r["aggregation"] for r in discriminating} - {"either", None}
    out = {
        "tolerance": TOL,
        "n_tasks_checked": len(per_task),
        "n_tasks_discriminating": len(discriminating),
        "consistent": len(decided) <= 1,
        "server_aggregation": (sorted(decided)[0] if len(decided) == 1
                               else "unknown"),
        "note": "Only tasks with unequal fold sizes can tell the two "
                "aggregations apart; the rest are reported as "
                "non-discriminating rather than as agreement.",
        "tasks": per_task,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print(f"server aggregation: {out['server_aggregation']} "
          f"({out['n_tasks_discriminating']}/{out['n_tasks_checked']} tasks "
          f"could discriminate; consistent: {out['consistent']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
