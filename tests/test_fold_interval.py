"""The fold-level interval is additional, conservative-by-default, and honest
about the assumption it borrows.

The clause-2 gate is an exact sign test over seeds. codex's objection was that
seeds re-use the same examples and so estimate algorithmic randomness
conditional on the data, not uncertainty about new data. `scripts/fold_interval.py`
adds the reading that speaks to split variance, from `accuracy_folds` already
recorded in every run.

Three things have to hold or the section is worse than not having it:

- the tabulated t critical values must actually be the t critical values;
- the correction must be conservative relative to the naive bound, and both
  must be reported, since the naive one is the flattering one;
- it must not enter any clause. A reading invented after the fact that granted
  a clause would be the exact move the amendment ledger exists to forbid.
"""
import json
import runpy
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FI = REPO / "runs/fold_interval.json"
R = runpy.run_path(str(REPO / "scripts/fold_interval.py"))


def test_the_tabulated_t_values_are_the_t_values():
    """The docstring claims these were checked against scipy. Check them."""
    try:
        from scipy.stats import t as student_t
    except Exception as exc:                       # scipy absent in some envs
        print(f"  scipy unavailable ({type(exc).__name__}); skipped")
        return
    worst = 0.0
    for df, v in R["T_CRIT_95_ONE_SIDED"].items():
        want = float(student_t.ppf(0.95, df))
        worst = max(worst, abs(v - want))
        assert abs(v - want) < 5e-5, (
            f"df={df}: tabulated {v}, scipy {want:.6f}")
    print(f"  {len(R['T_CRIT_95_ONE_SIDED'])} tabulated one-sided 95% t values "
          f"match scipy to {worst:.1e}")


def test_an_unknown_fold_count_says_so_instead_of_guessing():
    """A missing critical value must not silently become a bound."""
    out = R["fold_interval"]([0.9] * 18, 0.8, n_train=100, n_test=10)
    assert "lower_bound_naive" not in out, out
    assert "no tabulated t critical value" in out["note"], out
    print("  df=17 has no tabulated value and returns a note, not a bound")


def test_the_correction_is_always_the_conservative_one():
    """If it were ever the looser of the two, reporting both would be cover."""
    for K in (5, 10, 15):
        out = R["fold_interval"]([0.80 + 0.01 * i for i in range(K)], 0.75,
                                 n_train=900, n_test=100)
        assert out["se_corrected"] > out["se_naive"], (K, out)
        assert out["lower_bound_corrected"] < out["lower_bound_naive"], (K, out)
        assert out["variance_inflation"] > 1.0, (K, out)
    # for 10 folds with n_test/n_train = 1/9 the inflation is exactly
    # (1/10 + 1/9) / (1/10)
    out = R["fold_interval"]([0.8, 0.81, 0.79, 0.82, 0.78,
                              0.80, 0.81, 0.79, 0.82, 0.78], 0.75,
                             n_train=900, n_test=100)
    want = (1 / 10 + 100 / 900) / (1 / 10)
    assert abs(out["variance_inflation"] - want) < 1e-9, (
        out["variance_inflation"], want)
    print(f"  corrected SE exceeds naive at K=5,10,15; 10-fold inflation "
          f"{want:.4f}x as derived")


def test_the_tolerance_is_the_pre_registered_one():
    """This file may not choose the tolerance, only the uncertainty model."""
    assert R["TOLERANCE"] == -0.05, R["TOLERANCE"]
    base = json.loads((REPO / "runs/baselines.json").read_text())
    tol = base.get("tolerance_readings", {})
    assert "0.95" in json.dumps(tol), (
        "the registry's primary tolerance is not the 0.95x this file assumes")
    print("  tolerance -0.05 comes from the registry, not from this stage")


def test_it_does_not_enter_any_clause():
    """A reading invented after the fact must not grant a clause."""
    src = (REPO / "scripts/report.py").read_text()
    i = src.index("def verdict(")
    j = src.find("\ndef ", i + 1)
    body = src[i:j if j > 0 else len(src)]
    for token in ("fold_interval", "non_inferior", "lower_bound"):
        assert token not in body, (
            f"verdict() references {token!r}; the fold interval is an "
            "additional reading and must not decide a clause")
    if FI.exists():
        fi = json.loads(FI.read_text())
        assert fi["reading"].startswith("ADDITIONAL"), fi["reading"]
    print("  verdict() does not reference the fold interval")


def test_seeds_are_averaged_per_fold_not_pooled_as_independent():
    """The flattering move here is 80 'independent' observations."""
    if not FI.exists():
        print("  not run yet; skipped")
        return
    fi = json.loads(FI.read_text())
    assert "NOT treated as 80" in fi["not_done"] or "independent" in fi["not_done"]
    for tid, r in fi["tasks"].items():
        assert r["n_folds"] <= 10, (tid, r["n_folds"])
        assert r["n_seeds_averaged"] >= 1
        # df is K-1, so the interval was taken over folds, not over fold-seeds
        assert r["t_crit_95_one_sided"] == R["T_CRIT_95_ONE_SIDED"][
            r["n_folds"] - 1], (tid, r)
    print("  intervals are over folds (df = K-1), seeds averaged in")


def test_the_measured_ratio_of_the_two_noise_sources_is_recorded():
    """The reason this stage exists, kept as a number rather than a claim."""
    if not FI.exists():
        print("  not run yet; skipped")
        return
    fi = json.loads(FI.read_text())
    if "fold_sd_over_seed_range_min" not in fi:
        print("  fewer than two seeds per task; ratio not computable yet")
        return
    lo, hi = (fi["fold_sd_over_seed_range_min"],
              fi["fold_sd_over_seed_range_max"])
    assert lo > 0 and hi >= lo
    for tid, r in fi["tasks"].items():
        got = r["fold_sd_over_seed_range"]
        if got is None:
            continue
        want = (r["sd_relative_margin_over_folds"] / r["seed_range_relative"])
        assert abs(got - want) < 1e-9, (tid, got, want)
    print(f"  fold sd / seed range recomputed per task; range {lo:.2f}x-{hi:.2f}x")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
