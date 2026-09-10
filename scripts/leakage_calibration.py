"""What is the false-alarm rate of the leakage probe's decision rule?

The rule pre-registered in `critique_log.md` turn 9 and wired into clause 2 is

    LEAKAGE  <=>  any of the k permuted accuracies exceeds  p_maj + 2 sigma

which is a **maximum over k comparisons**, each at a one-sided nominal 2.3%.
The family was never calibrated.  codex made the general point when asked the
constructive question -- "calibrate the complete task-level decision, including
any maximum over permutations, rather than applying a nominal threshold
repeatedly" -- and this script is that calibration, from the recorded `n_test`
and `majority_rate` alone.  No new runs.

Direction, which is the reason this is a correction and not a withdrawal
---------------------------------------------------------------------------
An inflated false-alarm rate makes LEAKAGE *easier* to declare, and a LEAKAGE
verdict **sinks** clause 2.  So the pre-registered rule errs against the KPI.
The corrected rule is *easier to pass*, which is the direction a protocol may
never be moved in, so the pre-registered per-permutation rule stays primary and
the corrected reading is reported beside it, labelled, with the stricter of the
two identified.

Two ways this is an upper bound on an idealisation, not a measurement
--------------------------------------------------------------------
1. The k permutations share `X_train` and `X_test`, so their accuracies are
   positively correlated.  Positive correlation *lowers* the family-wise rate
   below the independent product, so treating them as independent is an upper
   bound.
2. A shuffled-label fit does not produce binomial accuracies.  It concentrates
   on the class prior, and the variance of that is *smaller* than
   `Binomial(n, p_maj)`, which pushes the true rate down again.

Both are stated in the output record, and neither is a reason to leave the
family uncalibrated.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "leakage_calibration.json"

K_PERMUTATIONS = 10          # the pre-registered k
NOISE_SIGMAS = 2.0           # the pre-registered band
TARGET_FWER = 0.05           # what a calibrated task-level decision should be


def _log_binom_sf(n: int, p: float, x: float) -> float:
    """P(Binomial(n, p) > x), summed in the tail that is small."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0
    k0 = math.floor(x) + 1
    if k0 > n:
        return 0.0
    if k0 <= 0:
        return 1.0
    # log-space to keep the tail exact at n in the thousands
    lp, lq = math.log(p), math.log1p(-p)
    lgn = math.lgamma(n + 1)
    total = 0.0
    for k in range(k0, n + 1):
        lt = (lgn - math.lgamma(k + 1) - math.lgamma(n - k + 1)
              + k * lp + (n - k) * lq)
        total += math.exp(lt)
    return min(1.0, total)


def per_comparison_alpha(p_maj: float, n_test: int) -> float:
    """P(one shuffled-label accuracy exceeds the 2-sigma band) under the
    idealised binomial null."""
    band = NOISE_SIGMAS * math.sqrt(p_maj * (1 - p_maj) / n_test)
    return _log_binom_sf(n_test, p_maj, (p_maj + band) * n_test)


def sigmas_for_fwer(p_maj: float, n_test: int, k: int,
                    target: float = TARGET_FWER) -> float | None:
    """How many sigmas would make the *task-level* decision hit `target`?

    Solved on the exact binomial tail rather than by a normal approximation,
    because at these n the tail is discrete enough for the two to differ.
    """
    per = 1.0 - (1.0 - target) ** (1.0 / k)          # Sidak
    sd = math.sqrt(p_maj * (1 - p_maj) / n_test)
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        a = _log_binom_sf(n_test, p_maj, (p_maj + mid * sd) * n_test)
        if a > per:
            lo = mid
        else:
            hi = mid
    return None if hi >= 11.999 else hi


def _norm_sf(z: float) -> float:
    """P(Z > z) for standard normal, via erfc."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_isf(a: float) -> float:
    """z such that P(Z > z) = a."""
    lo, hi = -12.0, 12.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if _norm_sf(mid) > a:
            lo = mid
        else:
            hi = mid
    return hi


def rank_rule_calibration(k: int, n_sigmas: float,
                          target: float = TARGET_FWER) -> dict:
    """Same calibration for the AUROC rule added in turn 10.

    The Mann-Whitney null for AUROC is asymptotically normal about 0.5, so the
    per-comparison rate at `n_sigmas` does not depend on the class prior or on
    n -- which is exactly the property that made the statistic worth adding,
    and it also means its family-wise inflation is a single number rather than
    one per task.

    This was written into the audit at the moment the audit found it: the rank
    rule is `max(aucs) > threshold`, the same shape as the accuracy rule, and
    it was written in the same turn that diagnosed the shape.  Recording that
    rather than quietly calibrating it.
    """
    a1 = _norm_sf(n_sigmas)
    fwer = 1.0 - (1.0 - a1) ** k
    per_needed = 1.0 - (1.0 - target) ** (1.0 / k)
    return {
        "statistic": "AUROC of the permuted arm vs the Mann-Whitney null",
        "rule": f"LEAKAGE if max over k={k} permuted AUROCs > 0.5 + "
                f"{n_sigmas} * SE_MannWhitney",
        "per_comparison_alpha": a1,
        "family_wise_alpha": fwer,
        "sigmas_for_5pct_family_wise": _norm_isf(per_needed),
        "prior_independent": True,
        "why_one_number_not_five": (
            "the Mann-Whitney null is centred at 0.5 with SE depending only on "
            "n1 and n2, so a rule stated in sigmas has a prior-independent "
            "per-comparison rate"),
        "direction": (
            "same as the accuracy rule: inflation makes LEAKAGE easier to "
            "declare, which sinks clause 2. The 2-sigma rule is the stricter."),
        "enters_no_clause": True,
    }


def main() -> int:
    power = json.loads((REPO / "runs" / "leakage_power.json").read_text())
    base = json.loads((REPO / "runs" / "baselines.json").read_text())
    order = list(base.get("selected_task_ids") or [])

    rows, tasks = [], {}
    for tid in order:
        t = power["tasks"].get(str(tid))
        if not t or t.get("status") != "measured":
            tasks[str(tid)] = {"task_id": tid, "status": "[not measured]"}
            continue
        q = t["pooled"]
        p_maj, n = q["majority_rate"], q["n_test_total"]
        a1 = per_comparison_alpha(p_maj, n)
        fwer = 1.0 - (1.0 - a1) ** K_PERMUTATIONS
        need = sigmas_for_fwer(p_maj, n, K_PERMUTATIONS)
        # Per-fold as well, since the probe runs fold sets and the single-fold
        # invocation is the one with the widest band.
        f0 = t["per_fold"][0]
        a1f = per_comparison_alpha(f0["majority_rate"], f0["n_test"])
        fwerf = 1.0 - (1.0 - a1f) ** K_PERMUTATIONS
        d = {
            "task_id": tid, "dataset_name": t["dataset_name"],
            "n_test_pooled": n, "majority_rate": p_maj,
            "per_comparison_alpha_pooled": a1,
            "family_wise_alpha_pooled": fwer,
            "per_comparison_alpha_fold0": a1f,
            "family_wise_alpha_fold0": fwerf,
            "sigmas_for_5pct_family_wise": need,
            "pre_registered_sigmas": NOISE_SIGMAS,
            "pre_registered_rule_is_the_stricter": (
                need is not None and need > NOISE_SIGMAS),
        }
        tasks[str(tid)] = d
        rows.append(d)

    worst = max((r["family_wise_alpha_pooled"] for r in rows), default=None)
    rank = rank_rule_calibration(K_PERMUTATIONS, NOISE_SIGMAS)
    record = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": ("family-wise false-alarm rate of the leakage probe's "
                    "max-over-k decision rule"),
        "k_permutations": K_PERMUTATIONS,
        "pre_registered_sigmas": NOISE_SIGMAS,
        "target_family_wise_alpha": TARGET_FWER,
        "no_new_runs": True,
        "idealisation": [
            "the k permutations are treated as independent; they share X_train "
            "and X_test and are positively correlated, which LOWERS the true "
            "family-wise rate, so these figures are an upper bound",
            "a shuffled-label fit concentrates on the class prior and its "
            "accuracy variance is smaller than Binomial(n, p_maj), which "
            "lowers the true rate again",
        ],
        "direction": (
            "an inflated false-alarm rate makes LEAKAGE easier to declare and "
            "a LEAKAGE verdict sinks clause 2, so the pre-registered rule errs "
            "AGAINST the KPI. The corrected rule is easier to pass, which is "
            "why the pre-registered one stays primary and this is reported "
            "beside it rather than replacing it."),
        "tasks": tasks,
        # The audit that found this, recorded with its scope so a reader knows
        # what was looked at and not only what was found.
        "class_audit": {
            "question": ("which decision rules in this repository take a "
                         "maximum, a minimum or a first-crossing over repeated "
                         "random draws, and is each one's family calibrated?"),
            "scanned": "scripts/*.py (13 files) and scripts/report.py",
            "instances_found": [
                "scripts/leakage_probe.py: accuracy rule, max over k "
                "permutations -- calibrated here, 18.7-21.4%",
                "scripts/leakage_probe.py: pooled accuracy rule, same shape -- "
                "the pooled columns here are that calibration",
                "scripts/leakage_probe.py: AUROC rule, max over k "
                "permutations -- calibrated here, written in the same turn "
                "the audit diagnosed the shape",
            ],
            "examined_and_not_an_instance": [
                "scripts/verify_metric.py:172 `bool(np.max(sep) > TOL)` -- a "
                "maximum over DETERMINISTIC published runs comparing two "
                "aggregation formulas, not over random draws, so there is no "
                "family to calibrate",
                "scripts/report.py clause conjunctions `all(...)` -- an "
                "intersection over tasks is conservative, not inflationary",
                "scripts/report.py:177 seed spread `max-min` -- descriptive; "
                "the gate that used it as a threshold was removed in turn 6",
            ],
            "already_closed_before_this_audit": [
                "exact_sign_test_above dropping ties (turn 6, 17.37% -> 0.85%)",
                "optional stopping at n=5,6,7 (turn 6)",
            ],
        },
        "rank_rule": rank,
        "worst_family_wise_alpha_pooled": worst,
        "pre_registered_rule_is_stricter_on_all_tasks": all(
            r["pre_registered_rule_is_the_stricter"] for r in rows) if rows else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.part")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
        fh.flush()
        import os
        os.fsync(fh.fileno())
    tmp.rename(OUT)

    print(f"{'task':>7} {'dataset':<34} {'n':>6} {'p_maj':>7} "
          f"{'a1':>9} {'FWER':>8} {'a1(f0)':>9} {'FWER(f0)':>9} {'sig@5%':>7}")
    for r in rows:
        need = r["sigmas_for_5pct_family_wise"]
        print(f"{r['task_id']:>7} {r['dataset_name']:<34} "
              f"{r['n_test_pooled']:>6} {r['majority_rate']:>7.4f} "
              f"{r['per_comparison_alpha_pooled']:>9.5f} "
              f"{r['family_wise_alpha_pooled']:>8.4f} "
              f"{r['per_comparison_alpha_fold0']:>9.5f} "
              f"{r['family_wise_alpha_fold0']:>9.4f} "
              f"{(f'{need:.2f}' if need else 'n/a'):>7}")
    print(f"\nworst family-wise false-alarm rate (pooled): {worst:.4f} "
          f"at a nominal {TARGET_FWER}")
    print(f"pre-registered 2-sigma rule is the stricter on all tasks: "
          f"{record['pre_registered_rule_is_stricter_on_all_tasks']}")
    print(f"\nrank (AUROC) rule, prior-independent: per-comparison "
          f"{rank['per_comparison_alpha']:.5f}, family-wise "
          f"{rank['family_wise_alpha']:.4f}, "
          f"{rank['sigmas_for_5pct_family_wise']:.3f} sigma would give 5%")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
