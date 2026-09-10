"""`scripts/leakage_power.py` decides which tasks the clause-2 gate demands a
probe for, so its arithmetic and its refusals both need tests.

Two kinds here:

- unit tests on the recovery and the power arithmetic, on synthetic records;
- invariant tests on the **shipped** `runs/leakage_power.json`, because that is
  the artifact `report.py` reads and a stale or internally inconsistent one
  would silently change what the gate requires.
"""
from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LP = runpy.run_path(str(REPO / "scripts/leakage_power.py"))
fold_majority_rates = LP["fold_majority_rates"]
NOISE_SIGMAS = LP["NOISE_SIGMAS"]

POWER = REPO / "runs" / "leakage_power.json"


def _record(k=10, n_per_class=(900, 100), accs=None):
    """A synthetic k-fold record with the per-fold train class counts the real
    ones carry."""
    a, b = n_per_class
    n = a + b
    per_fold, fa, fb = [], a // k, b // k
    for f in range(k):
        train = {"A": a - fa, "B": b - fb}
        per_fold.append({
            "fold": f, "accuracy": (accs[f] if accs else 0.9),
            "n_test": n - sum(train.values()),
            "profile": {"class_counts": train},
        })
    return {"n_folds_run": k, "per_fold": per_fold}


def test_the_recovery_identity_reproduces_the_test_folds():
    r = _record()
    out = fold_majority_rates(r)
    assert out is not None and len(out) == 10
    for d in out:
        assert d["n_test"] == 100
        # 90 A / 10 B per test fold -> majority 0.9
        assert abs(d["majority_rate"] - 0.9) < 1e-12, d
    print("  10-fold recovery: n_test=100, majority 0.9 on every fold")


def test_a_record_whose_counts_do_not_describe_its_own_folds_is_refused():
    """The cross-check that must fire, in the pessimistic direction.

    codex showed the count recovery cannot be made sound from counts alone --
    ten identical recorded splits pass integrality, non-negativity and the
    n_test check while the true composition is wrong -- which is why the
    shipped script reads the rates from the task's own splits and keeps this
    only as a per-record cross-check. It still has to refuse the cases it
    *can* see.
    """
    r = _record()
    r["per_fold"][3]["n_test"] = 137          # disagrees with the recovery
    assert fold_majority_rates(r) is None, (
        "a record whose recorded n_test contradicts the recovered partition "
        "was accepted")

    r2 = _record()
    r2["per_fold"][0]["profile"]["class_counts"] = {}
    assert fold_majority_rates(r2) is None, "a record with no class counts"

    r3 = _record()
    r3["n_folds_run"] = 9                     # inconsistent with len(per_fold)
    assert fold_majority_rates(r3) is None, "an inconsistent fold count"

    r4 = _record(k=1)
    assert fold_majority_rates(r4) is None, (
        "k=1 has no (k-1) divisor and must be refused, not divided by zero")
    print("  refuses: n_test mismatch, missing counts, fold-count mismatch, k=1")


def test_phi_min_is_the_band_over_the_gap_and_its_threshold_is_one():
    """`phi_min >= 1` must mean exactly "cannot detect a complete leak"."""
    d = json.loads(POWER.read_text())
    for tid, t in d["tasks"].items():
        if t.get("status") != "measured":
            continue
        q = t["pooled"]
        band = NOISE_SIGMAS * math.sqrt(
            q["majority_rate"] * (1 - q["majority_rate"]) / q["n_test_total"])
        assert abs(band - q["noise_band_2sigma"]) < 1e-12, tid
        assert abs(q["gap"] - (q["accuracy_intact"] - q["majority_rate"])) < 1e-12
        phi = q["min_resolvable_leak_fraction"]
        if q["gap"] > 0:
            assert abs(phi - band / q["gap"]) < 1e-9, (tid, phi)
        assert q["detects_complete_leakage"] == (phi < 1.0), (
            f"task {tid}: detects={q['detects_complete_leakage']} but "
            f"phi_min={phi}; those must be the same statement")
    print("  phi_min = band/gap on every measured task, and phi_min<1 "
          "<=> detects")


def test_pooling_more_folds_narrows_the_band_or_min_folds_is_meaningless():
    d = json.loads(POWER.read_text())
    for tid, t in d["tasks"].items():
        if t.get("status") != "measured":
            continue
        rows = t["per_fold"]
        bands = []
        for F in range(1, len(rows) + 1):
            sub = rows[:F]
            n = sum(r["n_test"] for r in sub)
            p = sum(r["majority_rate"] * r["n_test"] for r in sub) / n
            bands.append(NOISE_SIGMAS * math.sqrt(p * (1 - p) / n))
        assert bands[-1] < bands[0], (
            f"task {tid}: pooling ten folds did not narrow the band, so "
            f"min_folds_for_detection buys nothing")
        mf = t.get("min_folds_for_detection")
        if mf:
            # the claimed fold count must actually clear, and one fewer must not
            def clears(F):
                sub = rows[:F]
                n = sum(r["n_test"] for r in sub)
                a = sum(r["accuracy_intact_mean_over_seeds"] * r["n_test"]
                        for r in sub) / n
                p = sum(r["majority_rate"] * r["n_test"] for r in sub) / n
                return a - p > NOISE_SIGMAS * math.sqrt(p * (1 - p) / n)
            assert clears(mf), (tid, mf)
            assert mf == 1 or not clears(mf - 1), (
                f"task {tid}: min_folds_for_detection={mf} but {mf - 1} also "
                f"clears, so it is not minimal")
    print("  pooling narrows the band; min_folds_for_detection is minimal "
          "and sufficient on every task")


def test_the_shipped_power_file_validated_every_partition():
    """The refusals must have been exercised, not merely available."""
    d = json.loads(POWER.read_text())
    assert d.get("no_new_runs") is True
    measured = [t for t in d["tasks"].values() if t.get("status") == "measured"]
    assert measured, "the shipped power file measured nothing"
    for t in measured:
        shape = t.get("split_shape") or {}
        assert shape.get("partition_validated") is True, t["task_id"]
        assert shape.get("n_repeats") == 1, t["task_id"]
        assert t.get("recovery_cross_check"), t["task_id"]
        assert not t.get("records_refused"), (t["task_id"],
                                              t["records_refused"])
    print(f"  {len(measured)} tasks, every partition validated at 1 repeat, "
          f"recovery cross-check agrees, no records refused")


def test_the_family_wise_correction_can_only_cost_power_never_add_it():
    """A wider band is a higher threshold. If the corrected reading ever looks
    *easier* than the pre-registered one, the two are not the same statistic
    and the table is misleading."""
    d = json.loads(POWER.read_text())
    for tid, t in d["tasks"].items():
        fw = t.get("family_wise_corrected")
        if not fw:
            continue
        q = t["pooled"]
        assert fw["sigmas"] >= NOISE_SIGMAS, (tid, fw["sigmas"])
        assert fw["noise_band"] >= q["noise_band_2sigma"], tid
        assert (fw["min_resolvable_leak_fraction"]
                >= q["min_resolvable_leak_fraction"]), tid
        assert not (fw["detects_complete_leakage"]
                    and not q["detects_complete_leakage"]), (
            f"task {tid} is detectable under the *corrected* band but not "
            f"under the looser pre-registered one, which is impossible")
    moved = d.get("tasks_moved_to_blind_by_the_correction") or []
    powered_2s = set(d["tasks_powered_pooled"])
    powered_fw = set(d.get("tasks_powered_family_wise_corrected") or [])
    assert powered_fw <= powered_2s, (
        "the corrected reading powered a task the pre-registered one did not")
    assert set(moved) == powered_2s - powered_fw, (
        "moved_to_blind does not equal the set difference it claims to be")
    print(f"  correction is monotone; moved to blind: {moved}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
