"""Stage 1 — fix the human baselines BEFORE any of our own runs exist.

This script is the pre-registration.  It selects the five benchmark tasks by a
rule that cannot see our accuracy, fetches every published `predictive_accuracy`
evaluation on each, and writes `runs/baselines.json`.  Nothing downstream may
change a number in that file; `scripts/report.py` reads it and `tests/` checks
that the committed file still matches the rule.

Why the median of published runs is the baseline
------------------------------------------------
OpenML stores, per task, the accuracy every user ever uploaded for it, under the
task's own estimation procedure.  The median of that distribution is a
*measured* summary of what people actually achieved on this exact split, with
run ids we record, so it is checkable.  "What I remember the SOTA being" is not,
and choosing a target after seeing our own score is the way this KPI gets faked.

Two readings of "human baseline", both fixed here, neither chosen later
----------------------------------------------------------------------
`median_run`   median over all published runs.  One prolific uploader running a
               5000-point hyperparameter sweep therefore counts 5000 times.
`median_flow`  median over per-flow medians — one number per *method* published.
               De-weights sweeps; closer to "the typical published approach".
`median_run` is the primary because the brief named it; `median_flow` is
reported in the same table, never instead of it.

Task selection rule, fixed before any evaluation was fetched
------------------------------------------------------------
From OpenML-CC18 (study 99):
  1. keep tasks with n <= MAX_INSTANCES and p <= MAX_FEATURES  (CPU budget: the
     agent runs a 6-family tournament plus a search on every one of 10 folds);
  2. among those, take the TOP_K with the most published evaluations, because
     the median of a larger sample is the better-determined baseline.
Neither criterion can see our accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads.openml_io import configure  # noqa: E402

import openml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
EVALS = RUNS / "evals"

SUITE = 99                 # OpenML-CC18
METRIC = "predictive_accuracy"
MAX_INSTANCES = 20_000
MAX_FEATURES = 100
TOP_K = 5
PAGE = 1000
HARD_CAP = 300_000
WORKERS = 6                # network-bound; 51 tasks serially is hours

_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_tasks() -> pd.DataFrame:
    configure()
    suite = openml.study.get_suite(SUITE)
    tl = openml.tasks.list_tasks(task_id=list(suite.tasks), output_format="dataframe")
    keep = tl[(tl.NumberOfInstances <= MAX_INSTANCES)
              & (tl.NumberOfFeatures <= MAX_FEATURES)].copy()
    return keep.sort_values("tid").reset_index(drop=True)


def fetch_evals(task_id: int) -> pd.DataFrame:
    """Page every published evaluation of METRIC on this task.

    The server refuses limit >= 2000, and one page would sample the oldest runs
    by run_id and bias the median toward the 2014 Weka era, so we page to
    exhaustion.  `truncated` is recorded if HARD_CAP is ever hit.
    """
    configure()
    frames, offset = [], 0
    while offset < HARD_CAP:
        for attempt in range(4):
            try:
                df = openml.evaluations.list_evaluations(
                    METRIC, tasks=[task_id], size=PAGE, offset=offset,
                    output_format="dataframe")
                break
            except Exception as e:                      # transient 5xx from the API
                if attempt == 3:
                    raise
                say(f"    retry {attempt+1} on {task_id}@{offset}: "
                    f"{type(e).__name__}")
                time.sleep(5 * (attempt + 1))
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < PAGE:
            break
        offset += PAGE
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame, task_id: int, meta: dict, path: Path) -> dict:
    v = pd.to_numeric(df["value"], errors="coerce")
    ok = v.notna() & (v >= 0) & (v <= 1)
    d = df[ok].copy()
    d["value"] = v[ok]
    per_flow = d.groupby("flow_id")["value"].median()
    q = d["value"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "task_id": int(task_id),
        "dataset_name": str(meta["name"]),
        "data_id": int(meta["did"]),
        "n_instances": int(meta["NumberOfInstances"]),
        "n_features": int(meta["NumberOfFeatures"]),
        "n_classes": int(meta["NumberOfClasses"]),
        "estimation_procedure": str(meta["estimation_procedure"]),
        "metric": METRIC,
        "n_published_runs": int(len(d)),
        "n_runs_rejected": int((~ok).sum()),
        "n_distinct_flows": int(d["flow_id"].nunique()),
        "n_distinct_uploaders": int(d["uploader"].nunique()),
        # --- the two pre-registered readings of "human baseline" -------------
        "median_run": float(d["value"].median()),
        "median_flow": float(per_flow.median()),
        # --- context columns, never the target -------------------------------
        "q25": float(q[0.25]), "q75": float(q[0.75]), "q90": float(q[0.9]),
        "max_published": float(d["value"].max()),
        "min_published": float(d["value"].min()),
        "run_id_min": int(d["run_id"].min()),
        "run_id_max": int(d["run_id"].max()),
        "upload_time_min": str(d["upload_time"].min()),
        "upload_time_max": str(d["upload_time"].max()),
        "top_flow_by_runs": str(d["flow_id"].value_counts().index[0]),
        "evals_file": str(path.relative_to(REPO)),
        "evals_sha256": sha256(path),
        "truncated_at_hard_cap": bool(len(df) >= HARD_CAP),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "baselines.json"))
    args = ap.parse_args()

    EVALS.mkdir(parents=True, exist_ok=True)
    cands = candidate_tasks()
    print(f"CC18 tasks passing n<={MAX_INSTANCES}, p<={MAX_FEATURES}: "
          f"{len(cands)}", flush=True)

    def one(i_row):
        i, row = i_row
        tid = int(row.tid)
        path = EVALS / f"task_{tid}.csv.gz"
        if path.exists():
            df = pd.read_csv(path)
            say(f"[{i+1}/{len(cands)}] task {tid} {row['name']}: cached "
                f"{len(df)} evals")
        else:
            t0 = time.time()
            df = fetch_evals(tid)
            if len(df) == 0:
                say(f"[{i+1}/{len(cands)}] task {tid} {row['name']}: 0 evals")
                return None
            # write to a temp name and rename, so an interrupted fetch never
            # leaves a truncated file the next run would trust as a cache
            tmp = path.with_suffix(".part")
            df.to_csv(tmp, index=False, compression="gzip")
            tmp.rename(path)
            say(f"[{i+1}/{len(cands)}] task {tid} {row['name']}: {len(df)} "
                f"evals in {time.time()-t0:.0f}s")
        return summarise(df, tid, row, path)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        summaries = [r for r in ex.map(one, list(cands.iterrows()))
                     if r is not None]

    summaries.sort(key=lambda s: -s["n_published_runs"])
    selected = summaries[:TOP_K]
    for rank, s in enumerate(summaries):
        s["rank_by_n_runs"] = rank + 1
        s["selected"] = rank < TOP_K

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": SUITE,
        "metric": METRIC,
        "selection_rule": {
            "source": f"OpenML-CC18 (study {SUITE})",
            "filter": f"NumberOfInstances <= {MAX_INSTANCES} and "
                      f"NumberOfFeatures <= {MAX_FEATURES}",
            "n_passing_filter": int(len(cands)),
            "n_with_evals": len(summaries),
            "rank_by": "n_published_runs, descending",
            "top_k": TOP_K,
            "note": "Neither criterion can see this repo's accuracy. Fixed and "
                    "committed before any agent run existed.",
        },
        "baseline_definitions": {
            "median_run": "median predictive_accuracy over all published runs "
                          "on the task (primary)",
            "median_flow": "median over per-flow medians; one number per "
                           "published method (second reading)",
            "context_only": ["q25", "q75", "q90", "max_published",
                             "min_published"],
        },
        "tolerance_readings": {
            "primary_rel_one_sided": "ours >= 0.95 * baseline",
            "rel_two_sided": "|ours - baseline| / baseline <= 0.05",
            "abs_two_sided": "|ours - baseline| <= 0.05 (percentage points)",
            "note": "All three fixed here, before any run. The primary is "
                    "one-sided because '±5% 이내 자동도달' is a reaching "
                    "criterion and a two-sided rule would fail a run for "
                    "beating the baseline by 6%.",
        },
        "selected_task_ids": [s["task_id"] for s in selected],
        "tasks": summaries,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print(f"{'tid':>7} {'dataset':<28} {'runs':>7} {'flows':>6} "
          f"{'median_run':>11} {'median_flow':>12} {'sel':>4}")
    for s in summaries:
        print(f"{s['task_id']:>7} {s['dataset_name']:<28} "
              f"{s['n_published_runs']:>7} {s['n_distinct_flows']:>6} "
              f"{s['median_run']:>11.4f} {s['median_flow']:>12.4f} "
              f"{'*' if s['selected'] else '':>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
