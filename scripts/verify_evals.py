"""Stage 1b -- verify the committed evaluation cache against OpenML itself.

`runs/evals/task_*.csv.gz` is where every human baseline in this repository
comes from, and `runs/baselines.json` pins a sha256 per file. codex, asked how
it would fake this KPI, ranked as attack #7 the thing that pin does *not*
prove: a sha256 computed over rows I produced proves the file has not changed
since I wrote it, not that those rows are what OpenML published. Lower the
values in the cache, re-pin, regenerate, and every internal check still passes
because they all read the same file.

There are two separable questions and this script answers both, differently.

**Internal contamination** is answerable from the file alone, and these are
real routes rather than hypothetical: a duplicated `run_id` counts one human
submission twice in a median; a row whose `task_id` is not this task's is
another task's difficulty leaking in; a row whose `function` is not
`predictive_accuracy` blends a different metric into an accuracy median (the
same file layout carries AUC and f-measure rows on the server).

**Authenticity** is not answerable from the file at all, so it is answered by
asking the server. A seeded random sample of `run_id`s is drawn *before* any
fetch, the rule and the seed are written into the output, and each sampled row
is re-fetched from OpenML and compared value for value. This is the only check
in the repository whose evidence comes from outside it.

What it still does not prove: that the *unsampled* 99.99% of rows are
authentic. It is a spot check and the output calls it one -- with k rows per
task, systematic tampering of a fraction f of the file survives with
probability (1-f)^k, which is the number the JSON records so a reader can see
how weak or strong the guarantee is rather than taking "verified" on trust.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# The server rejects a `runs=` list whose implied limit is over its cap (code
# 546, "Requested result limit too high"), so the page size is explicit and the
# sample per task must not exceed it.
SERVER_PAGE = 100


def write_json_atomic(path: Path, obj) -> None:
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


def internal_checks(d: pd.DataFrame, task: dict, metric: str) -> dict:
    """Everything answerable from the committed file itself."""
    dup = int(d["run_id"].duplicated().sum())
    wrong_task = int((d["task_id"] != task["task_id"]).sum())
    wrong_data = int((d["data_id"] != task["data_id"]).sum())
    wrong_metric = int((d["function"] != metric).sum())
    v = pd.to_numeric(d["value"], errors="coerce")
    return {
        "n_rows": int(len(d)),
        "n_duplicate_run_ids": dup,
        "n_rows_from_another_task": wrong_task,
        "n_rows_from_another_dataset": wrong_data,
        "n_rows_with_another_metric": wrong_metric,
        "n_values_out_of_unit_range": int(((v < 0) | (v > 1)).sum()),
        "n_values_unparseable": int(v.isna().sum()),
        "run_id_min": int(d["run_id"].min()),
        "run_id_max": int(d["run_id"].max()),
        "upload_time_min": str(d["upload_time"].min()),
        "upload_time_max": str(d["upload_time"].max()),
        # the registry recorded these at fetch time; a later edit that adds or
        # drops rows at either end moves them
        "matches_registry_run_id_range": bool(
            int(d["run_id"].min()) == task["run_id_min"]
            and int(d["run_id"].max()) == task["run_id_max"]),
        "matches_registry_upload_window": bool(
            str(d["upload_time"].min()) == task["upload_time_min"]
            and str(d["upload_time"].max()) == task["upload_time_max"]),
        "clean": bool(dup == 0 and wrong_task == 0 and wrong_data == 0
                      and wrong_metric == 0),
    }


def probe(d: pd.DataFrame, task: dict, metric: str, k: int, seed: int) -> dict:
    """Re-fetch a seeded random sample of rows from OpenML and compare.

    The sample is drawn before the fetch and its seed is recorded, so it cannot
    be redrawn until it agrees.
    """
    import openml

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(d)), min(k, len(d))))
    sample = d.iloc[idx]
    ids = [int(r) for r in sample["run_id"].tolist()]
    t0 = time.time()
    got = openml.evaluations.list_evaluations(
        metric, runs=ids, output_format="dataframe", size=SERVER_PAGE)
    elapsed = time.time() - t0

    server = {int(r.run_id): float(r.value) for r in got.itertuples()}
    server_task = {int(r.run_id): int(r.task_id) for r in got.itertuples()}
    mism, missing = [], []
    for r in sample.itertuples():
        rid = int(r.run_id)
        if rid not in server:
            missing.append(rid)
            continue
        if abs(server[rid] - float(r.value)) > 1e-9:
            mism.append({"run_id": rid, "ours": float(r.value),
                         "server": server[rid]})
        elif server_task[rid] != int(r.task_id):
            mism.append({"run_id": rid, "ours_task": int(r.task_id),
                         "server_task": server_task[rid]})
    n = len(sample)
    return {
        "k_requested": k, "k_sampled": n, "sample_seed": seed,
        "sample_rule": "random.Random(seed).sample over row positions, drawn "
                       "before the fetch",
        "n_returned_by_server": len(server),
        "n_mismatched": len(mism),
        "mismatches": mism[:20],
        "n_not_returned_by_server": len(missing),
        "run_ids_not_returned": missing[:20],
        # a reader should see how weak a spot check is, not just that it passed
        "undetected_probability_if_1pct_tampered": round(0.99 ** n, 6),
        "undetected_probability_if_10pct_tampered": round(0.90 ** n, 6),
        "seconds": round(elapsed, 2),
        # A row the server does not return is not evidence of tampering: runs
        # get deleted, and a deleted run cannot be re-fetched. It is only a
        # mismatch when the server returns a *different* value.
        "verified": bool(len(mism) == 0 and len(server) > 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--out", default=str(RUNS / "evals_provenance.json"))
    ap.add_argument("-k", type=int, default=25,
                    help=f"rows re-fetched per task (server page is {SERVER_PAGE})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true",
                    help="internal checks only; the authenticity probe reads "
                         "[not measured] rather than being silently skipped")
    args = ap.parse_args()

    base = json.loads(Path(args.baselines).read_text())
    metric = base["metric"]
    selected = [t for t in base["tasks"] if t["selected"]]
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "baselines_file": str(Path(args.baselines).relative_to(REPO)),
           "metric": metric, "offline": bool(args.offline), "tasks": {}}

    for t in selected:
        f = REPO / t["evals_file"]
        d = pd.read_csv(f)
        rec = {"dataset_name": t["dataset_name"],
               "internal": internal_checks(d, t, metric)}
        if args.offline:
            rec["probe"] = None
        else:
            try:
                rec["probe"] = probe(d, t, metric, args.k, args.seed)
            except Exception as ex:            # network or server refusal
                rec["probe"] = {"error": f"{type(ex).__name__}: {ex}"[:300],
                                "verified": None}
        print(f"task {t['task_id']:6d} {t['dataset_name']:34s} "
              f"internal_clean={rec['internal']['clean']} "
              f"probe={(rec['probe'] or {}).get('verified')} "
              f"mismatched={(rec['probe'] or {}).get('n_mismatched')}",
              flush=True)
        out["tasks"][str(t["task_id"])] = rec

    recs = list(out["tasks"].values())
    out["all_internal_clean"] = all(r["internal"]["clean"] for r in recs)
    out["all_probes_verified"] = (
        None if any((r["probe"] or {}).get("verified") is None for r in recs)
        else all((r["probe"] or {})["verified"] for r in recs))
    out["n_rows_refetched_total"] = sum(
        (r["probe"] or {}).get("k_sampled", 0) for r in recs)
    out["n_mismatches_total"] = sum(
        (r["probe"] or {}).get("n_mismatched", 0) for r in recs)
    out["what_this_does_not_prove"] = (
        "the unsampled rows. Per task this re-fetches k of ~10^5 rows, so it "
        "detects wholesale tampering and would very likely miss a single "
        "edited row; the per-task undetected probabilities are recorded above.")
    write_json_atomic(Path(args.out), out)
    print(f"wrote {args.out}: internal_clean={out['all_internal_clean']} "
          f"probes_verified={out['all_probes_verified']} "
          f"({out['n_rows_refetched_total']} rows re-fetched, "
          f"{out['n_mismatches_total']} mismatched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
