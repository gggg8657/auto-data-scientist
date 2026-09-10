"""Stage 2c -- the reading that speaks to *split* variance, not seed variance.

The clause-2 gate is an exact sign test over seeds. It is the right test for
what it tests and it tests the smaller of the two noise sources. codex, asked
how to make the clause pass more defensibly:

    "Eight seeds reuse the same examples. They can estimate algorithmic
    randomness conditional on those data; they do not create eight independent
    datasets. Calling their p-value evidence about new-data performance is
    flattering overconfidence."

Correct, and the size of the gap is measurable from data already on disk --
every run record carries `accuracy_folds`, the ten outer-fold accuracies. So
this stage puts a confidence interval on the relative margin **across folds**
and reports it beside the seed test.

    per fold k:  d_k = (our accuracy on fold k  -  baseline) / baseline
    H0 (non-inferiority):  mean_k d_k  <=  -0.05
    reject  <=>  the one-sided lower confidence bound on mean_k d_k  >  -0.05

`-0.05` is the pre-registered tolerance and is not chosen here; only the
uncertainty model is new, and it is *added* rather than substituted.

**Two intervals, both reported, because the honest one is not obvious.**
The ten folds have disjoint test sets and heavily overlapping training sets, so
they are not independent and a naive `s/sqrt(K)` understates the variance --
anti-conservative, i.e. the flattering direction.

- `naive`: `SE = s / sqrt(K)`. Reported and labelled as the optimistic bound.
- `corrected`: `SE = s * sqrt(1/K + n_test/n_train)`, the Nadeau & Bengio
  (2003) variance correction, which for 10-fold is `sqrt(1/10 + 1/9)` and
  inflates the variance by 2.11x (SE by 1.45x).

The assumption, stated rather than buried: that correction is *derived* for
repeated random subsampling, not for k-fold cross-validation. It is used here
as a conservative adjustment for fold dependence, which is how it is commonly
applied, and both bounds are printed so a reader who rejects the adjustment can
read the other row. What is not done, and would be the flattering move, is
treating the 80 fold-seed scores as 80 independent observations.

Seeds are averaged per fold before the interval is taken, so seed randomness is
averaged out and what remains is fold-to-fold variation. That keeps the two
noise sources separate: the sign test owns the seed axis, this owns the fold
axis, and the ratio between them is recorded because it turns out to be large.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# One-sided 95% Student-t critical values, df = K - 1. Tabulated rather than
# pulled from scipy because this environment's scipy is shared and the values
# for the fold counts this benchmark can produce are few. Checked against
# scipy.stats.t.ppf(0.95, df) to 4 dp in tests.
T_CRIT_95_ONE_SIDED = {
    1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432,
    7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125, 11: 1.7959, 12: 1.7823,
    13: 1.7709, 14: 1.7613, 15: 1.7531, 19: 1.7291, 29: 1.6991,
}
TOLERANCE = -0.05          # pre-registered; not chosen in this file


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


def t_crit(df: int) -> float | None:
    if df in T_CRIT_95_ONE_SIDED:
        return T_CRIT_95_ONE_SIDED[df]
    return None


def fold_interval(fold_accs: list[float], baseline: float,
                  n_train: int, n_test: int) -> dict:
    """One-sided lower bounds on the mean relative margin over folds."""
    K = len(fold_accs)
    d = [(a - baseline) / baseline for a in fold_accs]
    m = statistics.mean(d)
    if K < 2:
        return {"n_folds": K, "mean_relative_margin": m,
                "note": "fewer than two folds; no interval"}
    s = statistics.stdev(d)
    tc = t_crit(K - 1)
    se_naive = s / math.sqrt(K)
    # Nadeau & Bengio: sigma^2_corrected = (1/K + n_test/n_train) * s^2
    ratio = n_test / n_train if n_train else 1 / (K - 1)
    se_corr = s * math.sqrt(1 / K + ratio)
    out = {
        "n_folds": K,
        "mean_relative_margin": m,
        "sd_relative_margin_over_folds": s,
        "n_train": n_train, "n_test": n_test,
        "n_test_over_n_train": ratio,
        "t_crit_95_one_sided": tc,
        "se_naive": se_naive,
        "se_corrected": se_corr,
        "variance_inflation": (se_corr / se_naive) ** 2 if se_naive else None,
        "tolerance": TOLERANCE,
    }
    if tc is None:
        out["note"] = f"no tabulated t critical value for df={K - 1}"
        return out
    out["lower_bound_naive"] = m - tc * se_naive
    out["lower_bound_corrected"] = m - tc * se_corr
    out["non_inferior_naive"] = bool(out["lower_bound_naive"] > TOLERANCE)
    out["non_inferior_corrected"] = bool(
        out["lower_bound_corrected"] > TOLERANCE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--bench", default=str(RUNS / "bench"))
    ap.add_argument("--out", default=str(RUNS / "fold_interval.json"))
    args = ap.parse_args()

    base = json.loads(Path(args.baselines).read_text())
    sel = {t["task_id"]: t for t in base["tasks"] if t["selected"]}
    bench = Path(args.bench)

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "what": "One-sided lower confidence bound on the mean relative margin "
                "over the task's own outer folds, against the pre-registered "
                "-0.05 tolerance. Speaks to split variance; the clause-2 gate "
                "is a seed test and speaks to algorithmic variance.",
        "reading": "ADDITIONAL. The gate is unchanged and this does not enter "
                   "any clause.",
        "correction": "Nadeau & Bengio (2003) variance correction "
                      "(1/K + n_test/n_train) * s^2, derived for repeated "
                      "random subsampling and applied here as a conservative "
                      "adjustment for k-fold dependence. The naive bound is "
                      "reported beside it and is anti-conservative.",
        "not_done": "The 80 fold-by-seed scores are NOT treated as 80 "
                    "independent observations; overlapping training folds are "
                    "dependent and pooling them that way understates variance.",
        "tasks": {},
    }

    for tid, t in sel.items():
        files = sorted(bench.glob(f"task_{tid}_seed*.json"))
        if not files:
            continue
        recs = [json.loads(f.read_text()) for f in files]
        K = min(len(r["accuracy_folds"]) for r in recs)
        # average across seeds per fold: seed randomness averaged out, so the
        # spread that remains is fold-to-fold, which is what this measures
        fold_mean = [statistics.mean(r["accuracy_folds"][k] for r in recs)
                     for k in range(K)]
        pf = recs[0]["per_fold"][0]
        rec = fold_interval(fold_mean, t["median_run"],
                            pf["n_train"], pf["n_test"])
        rec["dataset_name"] = t["dataset_name"]
        rec["n_seeds_averaged"] = len(recs)
        rec["seeds_averaged"] = sorted(r["random_state"] for r in recs)
        rec["baseline_median_run"] = t["median_run"]
        # the comparison that motivates reporting this at all
        seed_accs = [r["accuracy_pooled"] for r in recs]
        rec["seed_range_pooled"] = (max(seed_accs) - min(seed_accs)
                                    if len(seed_accs) > 1 else None)
        rec["seed_range_relative"] = (
            rec["seed_range_pooled"] / t["median_run"]
            if rec["seed_range_pooled"] is not None else None)
        rec["fold_sd_over_seed_range"] = (
            rec["sd_relative_margin_over_folds"] / rec["seed_range_relative"]
            if rec.get("seed_range_relative") else None)
        out["tasks"][str(tid)] = rec
        print(f"task {tid:6d} {t['dataset_name'][:24]:24s} "
              f"seeds={rec['n_seeds_averaged']} K={rec['n_folds']} "
              f"mean={rec['mean_relative_margin']:+.4f} "
              f"lo_naive={rec.get('lower_bound_naive', float('nan')):+.4f} "
              f"lo_corr={rec.get('lower_bound_corrected', float('nan')):+.4f} "
              f"non_inf={rec.get('non_inferior_corrected')}", flush=True)

    recs = list(out["tasks"].values())
    out["n_tasks"] = len(recs)
    out["all_non_inferior_naive"] = (
        all(r.get("non_inferior_naive") for r in recs) if recs else None)
    out["all_non_inferior_corrected"] = (
        all(r.get("non_inferior_corrected") for r in recs) if recs else None)
    ratios = [r["fold_sd_over_seed_range"] for r in recs
              if r.get("fold_sd_over_seed_range")]
    if ratios:
        out["fold_sd_over_seed_range_min"] = min(ratios)
        out["fold_sd_over_seed_range_max"] = max(ratios)
        out["why_that_ratio_matters"] = (
            "The fold-to-fold sd of the relative margin exceeds the "
            "seed-to-seed range by this factor. Where it is large, the "
            "seed-count discipline -- correct on its own terms -- was "
            "constraining the smaller of the two noise sources.")
    write_json_atomic(Path(args.out), out)
    print(f"wrote {args.out}")
    print(f"  non-inferior on all {out['n_tasks']} tasks: "
          f"naive={out['all_non_inferior_naive']} "
          f"corrected={out['all_non_inferior_corrected']}")
    if ratios:
        print(f"  fold sd / seed range: {min(ratios):.1f}x to {max(ratios):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
