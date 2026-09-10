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
import math
import os
import random
import statistics
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


def _spearman(a, b) -> float:
    """Rank correlation with tie-averaged ranks. No scipy in this env."""
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    ra, rb = rk(a), rk(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra)
           * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def selection_effect(rows: list[dict], control: str, floor_key: str,
                     n_perm: int, seed: int) -> dict:
    """Does the popularity ranking select for undemanding thresholds?

    Two tests of two different things, and the difference is the reason this
    function exists rather than one number being quoted.

    **The draw** (hypergeometric over the 5 selected): asks whether *this*
    five is an unusual sample of the pool. At n=5 against a ~50% base rate it
    can reject only when 0 of 5 land -- its power is at most P(k=0) -- so a
    p-value from it is close to uninformative. It was the test quoted on
    2026-09-10 turn 6 (p=0.0222 against the majority-class floor) and it does
    not survive the stricter floor (p=0.187). Reported here so the reader can
    see both, and so the weaker test cannot be quoted alone again.

    **The rule** (Spearman over all 51 candidates, with a permutation null):
    asks whether popularity is associated with a less demanding threshold at
    all. n=51 rather than 5. This is the test the question actually calls for,
    since the question is about the selection *rule* and not about one draw.
    A positive rho means headroom grows with rank number, i.e. the
    most-published tasks have the least demanding thresholds.
    """
    ok = [r for r in rows if not r.get("error")]
    rank = [r["rank_by_n_runs"] for r in ok]
    if floor_key == "majority_class_rate":
        head = [r["threshold_primary"] - r["majority_class_rate"] for r in ok]
    else:
        head = [r["threshold_primary"]
                - r["controls"][control]["accuracy_pooled"] for r in ok]
    rho = _spearman(rank, head)

    rnd = random.Random(seed)
    h = list(head)
    hits = 0
    for _ in range(n_perm):
        rnd.shuffle(h)
        if _spearman(rank, h) >= rho:
            hits += 1
    p_perm = (hits + 1) / (n_perm + 1)          # never reports exactly 0

    # the draw test, on the same floor
    def clears_floor(r) -> bool:
        if floor_key == "majority_class_rate":
            return bool(r["threshold_primary"] > r["majority_class_rate"])
        return bool(r["falsifiable_vs_all_controls"])

    N = len(ok)
    K = sum(1 for r in ok if clears_floor(r))
    sel = [r for r in ok if r["selected"]]
    n_draw = len(sel)
    k_obs = sum(1 for r in sel if clears_floor(r))

    def hyp(k):
        if k < 0 or k > K or (n_draw - k) > (N - K):
            return 0.0
        return (math.comb(K, k) * math.comb(N - K, n_draw - k)
                / math.comb(N, n_draw))

    p_draw = sum(hyp(k) for k in range(0, k_obs + 1))
    # which draws could this test ever reject on? if only k=0, say so.
    rejectable = [k for k in range(0, n_draw + 1)
                  if sum(hyp(i) for i in range(0, k + 1)) <= 0.05]
    return {
        "floor": floor_key,
        "rule_test": {
            "what": "Spearman(popularity rank, threshold - floor) over all "
                    "candidates, one-sided permutation null. Positive rho = "
                    "the most-published tasks have the least demanding "
                    "thresholds.",
            "n": N, "spearman_rho": round(rho, 4),
            "n_permutations": n_perm,
            "p_one_sided_permutation": round(p_perm, 5),
        },
        "draw_test": {
            "what": "exact hypergeometric over the 5 selected. Underpowered "
                    "at n=5; see rejectable_draws.",
            "base_rate": f"{K}/{N}", "observed": f"{k_obs}/{n_draw}",
            "expected": round(n_draw * K / N, 3),
            "p_lower_tail": round(p_draw, 5),
            "rejectable_draws": rejectable,
            "note": "this test can reject only on the draws listed above, so "
                    "a non-significant p from it is weak evidence of absence.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--out", default=str(RUNS / "falsifiability_floor.json"))
    ap.add_argument("--controls", nargs="*", default=["prior", "stump"])
    ap.add_argument("--permutations", type=int, default=200000)
    ap.add_argument("--perm-seed", type=int, default=0)
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
        "selection_effect": {
            k: selection_effect(rows, "stump", k, args.permutations,
                                args.perm_seed)
            for k in ("stump", "majority_class_rate")
        } if "stump" in args.controls else {},
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
