"""Stage 1d -- what floor does a task's threshold actually have to clear?

`scripts/target_difficulty.py` asks whether `0.95 x median_run` sits above the
**majority-class rate**. That was the right first question and it is not
sufficient, which was measured rather than assumed: the criterion makes the
`prior` control fail *by construction* -- its accuracy IS the majority rate --
so it can be evidence about nothing stronger than a majority-class predictor.

On the successor five chosen by that criterion, `prior` duly cleared 0 of 5 and
the **untuned depth-3 tree cleared 3 of 5**. So H1 ("above the majority rate is
sufficient for the clause to be falsifiable") is false, and the honest floor is
a *procedure* floor rather than a class-prior floor.

This script measures that floor over the whole candidate pool: for every task in
the frozen registry, run the frozen controls through the task's own outer folds
and record whether the pre-registered threshold clears each one. It stays blind
to our accuracy the same way the controls do -- a frozen procedure's score is a
property of the dataset, and nothing here reads `runs/bench`.

Reported, not acted on. The registered five remain the KPI. The purpose is to
say honestly *how* demanding the target is, and to let a successor measurement
state a criterion that means something instead of one that is true by
construction.

Cost: `prior` is free, `stump` is three splits of a tree. Both are trivial
beside the agent, which is why this could be run over 51 tasks rather than 5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--out", default=str(RUNS / "falsifiability_floor.json"))
    ap.add_argument("--controls", nargs="*", default=["prior", "stump"])
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only; stamps partial=true into the output")
    args = ap.parse_args()

    # same frozen controls, same evaluation path, as scripts/negative_control.py
    from negative_control import run_control

    base = json.loads(Path(args.baselines).read_text())
    tasks = base["tasks"][:args.limit] if args.limit else base["tasks"]

    rows, failed = [], []
    for t in tasks:
        rec = {"task_id": t["task_id"], "dataset_name": t["dataset_name"],
               "rank_by_n_runs": t["rank_by_n_runs"],
               "selected": bool(t["selected"]),
               "median_run": t["median_run"],
               "threshold_primary": 0.95 * t["median_run"],
               "threshold_strictest": 0.95 * t["strictest_baseline_value"],
               "controls": {}}
        try:
            for name in args.controls:
                c = run_control(t["task_id"], name)
                rec["controls"][name] = {
                    "accuracy_pooled": c["accuracy_pooled"],
                    "clears_primary": bool(
                        c["accuracy_pooled"] >= rec["threshold_primary"]),
                    "clears_strictest": bool(
                        c["accuracy_pooled"] >= rec["threshold_strictest"]),
                    "seconds": c["seconds"],
                }
                if name == "prior":
                    rec["majority_class_rate"] = c["majority_class_rate"]
        except Exception as ex:
            # a task that cannot be loaded is recorded as such, never dropped:
            # silently skipping the awkward tasks would bias the counts
            rec["error"] = f"{type(ex).__name__}: {ex}"[:200]
            failed.append(t["task_id"])
        rec["falsifiable_vs_all_controls"] = bool(
            rec["controls"] and not rec.get("error")
            and all(not c["clears_primary"] for c in rec["controls"].values()))
        print(f"  rank {rec['rank_by_n_runs']:2d} task {rec['task_id']:6d} "
              f"{rec['dataset_name']:34s} thr={rec['threshold_primary']:.4f} "
              + " ".join(
                  f"{n}={rec['controls'][n]['accuracy_pooled']:.4f}"
                  f"{'*' if rec['controls'][n]['clears_primary'] else ' '}"
                  for n in args.controls if n in rec["controls"])
              + ("  ERROR" if rec.get("error") else ""),
              flush=True)
        rows.append(rec)

    ok = [r for r in rows if not r.get("error")]
    sel = [r for r in ok if r["selected"]]

    def counts(pool):
        out = {}
        for name in args.controls:
            out[name] = {
                "n_clearing_primary": sum(
                    r["controls"][name]["clears_primary"] for r in pool),
                "n_clearing_strictest": sum(
                    r["controls"][name]["clears_strictest"] for r in pool),
            }
        out["n_falsifiable_vs_all_controls"] = sum(
            r["falsifiable_vs_all_controls"] for r in pool)
        out["n_tasks"] = len(pool)
        return out

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "partial": bool(args.limit),
        "question": "For each candidate task, does 0.95 x median_run clear a "
                    "frozen incapable procedure? A threshold above the "
                    "majority-class rate is necessary and not sufficient: on "
                    "the majority-rate-selected successor five, prior cleared "
                    "0/5 and the depth-3 stump cleared 3/5.",
        "blind_to_our_accuracy": "a frozen procedure's score is a property of "
                                 "the dataset; nothing here reads runs/bench.",
        "does_not_change_the_registered_task_set": True,
        "controls": args.controls,
        "n_tasks_attempted": len(rows),
        "n_tasks_failed_to_load": len(failed),
        "task_ids_failed_to_load": failed,
        "over_all_candidates": counts(ok),
        "over_the_registered_five": counts(sel),
        # the successor criterion that would actually mean something
        "a_stump_falsifiable_five_under_a_blind_rule": [
            {"task_id": r["task_id"], "dataset_name": r["dataset_name"],
             "rank_by_n_runs": r["rank_by_n_runs"],
             "threshold_primary": round(r["threshold_primary"], 6),
             "stump": r["controls"].get("stump", {}).get("accuracy_pooled"),
             "margin_over_stump": round(
                 r["threshold_primary"]
                 - r["controls"]["stump"]["accuracy_pooled"], 6)}
            for r in sorted((r for r in ok if r["falsifiable_vs_all_controls"]),
                            key=lambda r: r["rank_by_n_runs"])[:5]]
        if "stump" in args.controls else [],
        "a_stump_falsifiable_five_rule":
            "the five most-published candidates whose threshold clears EVERY "
            "frozen control, not merely the class prior. Still blind to our "
            "accuracy. Offered as a criterion, NOT substituted for the "
            "registered task set.",
        "tasks": rows,
    }
    write_json_atomic(Path(args.out), out)
    print(f"\nwrote {args.out}")
    print(f"  attempted {out['n_tasks_attempted']}, "
          f"failed to load {out['n_tasks_failed_to_load']}")
    for label, key in (("all candidates", "over_all_candidates"),
                       ("the registered five", "over_the_registered_five")):
        c = out[key]
        print(f"  {label} (n={c['n_tasks']}): "
              + ", ".join(f"{n} clears {c[n]['n_clearing_primary']}"
                          for n in args.controls)
              + f", falsifiable vs all controls "
                f"{c['n_falsifiable_vs_all_controls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
