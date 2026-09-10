"""Stage 1c -- how demanding is the pre-registered target, across the pool?

`scripts/negative_control.py` measures that trivial procedures clear the clause
on four of the five registered tasks. This asks the next question, which needs
no runs at all: **was that bad luck, or does the selection rule cause it?**

A task's primary threshold is `0.95 x median_run`. If that sits at or below the
dataset's majority-class rate, a majority-class predictor clears it and the
clause cannot distinguish competence from its absence on that task. Both
quantities are properties of *published* data -- the baseline is OpenML's run
history and the majority rate is a dataset quality -- so this whole analysis is
blind to our accuracy by construction. Nothing here can be tuned toward a
result of ours.

What it produces: the count over all 51 candidates that passed the registered
size filter, the same count restricted to the five that the top-5-by-published-
runs rule selected, and the rank correlation between popularity and
falsifiability. The last one is the point: if popularity anti-correlates with a
demanding threshold, then the rule that picked our five -- chosen because the
median of a larger sample is better determined, which is true -- was also
selecting for tasks whose medians sit near triviality.

This does **not** change the registered task set. The five stay the
measurement; a different five is a different measurement and would have to be
reported as one. This is a characterisation of the target, in the same spirit
as reporting four baseline readings instead of one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd

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
    ap.add_argument("--out", default=str(RUNS / "target_difficulty.json"))
    args = ap.parse_args()

    import openml

    base = json.loads(Path(args.baselines).read_text())
    dl = openml.datasets.list_datasets(output_format="dataframe").set_index("did")

    rows, missing = [], []
    for t in base["tasks"]:
        did = t["data_id"]
        if did not in dl.index:
            missing.append(t["task_id"])
            continue
        q = dl.loc[did]
        n, maj = q.get("NumberOfInstances"), q.get("MajorityClassSize")
        if not n or not maj or pd.isna(n) or pd.isna(maj):
            missing.append(t["task_id"])
            continue
        rate = float(maj) / float(n)
        thr = 0.95 * t["median_run"]
        rows.append({
            "task_id": t["task_id"], "dataset_name": t["dataset_name"],
            "rank_by_n_runs": t["rank_by_n_runs"],
            "selected": bool(t["selected"]),
            "n_published_runs": t["n_published_runs"],
            "majority_class_rate": round(rate, 6),
            "median_run": t["median_run"],
            "threshold_primary": round(thr, 6),
            # the whole question, per task
            "threshold_above_majority_rate": bool(thr > rate),
            "headroom": round(thr - rate, 6),
        })

    d = pd.DataFrame(rows)
    sel = d[d.selected]
    # Spearman between popularity rank (1 = most published) and headroom.
    # Negative would mean: the more published a task, the *less* demanding its
    # threshold is relative to triviality.
    rho = float(d["rank_by_n_runs"].corr(d["headroom"], method="spearman"))

    # Was 1-of-5 bad luck, or does the rule cause it? Exact hypergeometric,
    # lower tail: if 5 tasks were drawn at random from the same candidate pool,
    # how often would at most this many be falsifiable? No approximation and no
    # simulation -- the pool is 51 and the draw is 5.
    N = int(len(d))
    K = int(d["threshold_above_majority_rate"].sum())
    n_draw = int(len(sel))
    k_obs = int(sel["threshold_above_majority_rate"].sum())

    def _hyp(k):
        if k < 0 or k > K or (n_draw - k) > (N - K):
            return 0.0
        return (math.comb(K, k) * math.comb(N - K, n_draw - k)
                / math.comb(N, n_draw))

    p_lower = sum(_hyp(k) for k in range(0, k_obs + 1))
    expected = n_draw * K / N
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": "For how many candidate tasks is 0.95 x median_run above "
                    "the majority-class rate, i.e. not clearable by predicting "
                    "the commonest label?",
        "blind_to_our_accuracy": "Both inputs are published: the baseline is "
                                 "OpenML's run history and the majority rate is "
                                 "a dataset quality. Nothing here can see or be "
                                 "tuned toward a run of ours.",
        "does_not_change_the_registered_task_set": True,
        "n_candidates": int(len(d)),
        "n_candidates_missing_qualities": len(missing),
        "task_ids_missing_qualities": missing,
        "n_falsifiable": int(d["threshold_above_majority_rate"].sum()),
        "n_falsifiable_among_selected": int(
            sel["threshold_above_majority_rate"].sum()),
        "n_selected": int(len(sel)),
        "spearman_rank_vs_headroom": round(rho, 4),
        "expected_n_falsifiable_in_a_random_five": round(expected, 3),
        "exact_hypergeometric_p_lower_tail": round(p_lower, 5),
        "hypergeometric_h0": "the 5 registered tasks are a random draw of 5 "
                             "from the 51 candidates with respect to "
                             "falsifiability. Lower-tail p is the probability "
                             "of drawing at most as few falsifiable tasks as "
                             "were drawn.",
        "spearman_interpretation":
            "rank 1 = most published. A POSITIVE rho means headroom grows as "
            "rank number grows, i.e. the most-published tasks have the least "
            "demanding thresholds -- the selection rule is choosing them.",
        "a_falsifiable_five_under_a_blind_rule": [
            {k: r[k] for k in ("task_id", "dataset_name", "rank_by_n_runs",
                               "headroom")}
            for r in sorted(
                (r for r in rows if r["threshold_above_majority_rate"]),
                key=lambda r: r["rank_by_n_runs"])[:5]],
        "a_falsifiable_five_rule": "the five most-published candidates whose "
                                   "threshold exceeds their majority-class "
                                   "rate. Still blind to our accuracy. Offered "
                                   "as the successor measurement, NOT as this "
                                   "one.",
        "tasks": rows,
    }
    write_json_atomic(Path(args.out), out)
    print(f"wrote {args.out}")
    print(f"  candidates: {out['n_candidates']} "
          f"({out['n_candidates_missing_qualities']} missing qualities)")
    print(f"  falsifiable thresholds: {out['n_falsifiable']}/"
          f"{out['n_candidates']}")
    print(f"  falsifiable among the 5 registered: "
          f"{out['n_falsifiable_among_selected']}/{out['n_selected']}")
    print(f"  expected falsifiable in a random 5: "
          f"{out['expected_n_falsifiable_in_a_random_five']}, "
          f"exact hypergeometric p = "
          f"{out['exact_hypergeometric_p_lower_tail']}")
    print(f"  spearman(popularity rank, headroom) = "
          f"{out['spearman_rank_vs_headroom']}")
    for r in out["a_falsifiable_five_under_a_blind_rule"]:
        print(f"    would-be pick: rank {r['rank_by_n_runs']:2d} "
              f"{r['dataset_name']:24s} headroom {r['headroom']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
