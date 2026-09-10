"""How much leakage could `scripts/leakage_probe.py` actually detect, per task?

A probe with no power is not evidence of cleanliness, and this is the number
that says which of the registered five it is worth running on.

The instrument in `leakage_probe.py` compares a shuffled-label fit against the
test fold's **majority-class rate**.  Under complete leakage the shuffled arm
scores at the intact accuracy; under none it scores at the majority rate.  So
the whole dynamic range of the probe on a given task is

    gap = accuracy_intact - majority_rate

and the detection threshold sits a 2-sigma binomial band above the majority
rate.  The smallest leak the probe can resolve is therefore the fraction

    phi_min = band / gap

of that gap.  `phi_min >= 1` means the probe **cannot detect even complete
leakage** on that task: an agent handed the true test labels would score inside
the noise of a majority-class predictor, because the honest agent already does.

Where the majority rates come from
----------------------------------
**Directly from the task's own splits**, which are in the repo-local
`.omlcache`, plus the label vector.  Nothing is fitted and nothing is fetched
over the network.

The first version of this script instead *recovered* them from each record's
per-fold training `class_counts`, using
`total = sum(train_counts)/(k-1)` for k-fold CV.  The algebra is right for a
single-repeat partition -- codex verified the divisor against
`.omlcache/.../tasks/{3917,10101}/task.xml`, which declare one repeat and ten
folds -- but codex also showed the validation around it was insufficient, with
an exact counterexample:

    true dataset 85 A / 15 B; ten recorded splits that all repeat the SAME
    training subset (81 A, 9 B) with ten-row test sets. The reconstruction
    yields totals of 90 A / 10 B and a test composition of 9 A / 1 B per fold.
    Integrality passes, non-negativity passes, and the recovered n_test equals
    the recorded n_test -- while the true test composition is 4 A / 6 B.

and separately that for R repeats the divisor is 9R, not 10R-1, and that keying
the aggregation on `fold` alone collapses repeats.  Both correct.  Rather than
add more checks to an inference that cannot be made sound from counts alone,
the rates are now read from the splits themselves and the recovery is kept
**only as a cross-check** on the records: it must agree, or the task is
refused.  That makes the counterexample unreachable -- it was an attack on a
premise this script no longer relies on.

What this connects to
---------------------
`runs/falsifiability_floor.json` measured that a `DummyClassifier(prior)`
clears the ±5% band on 4 of the registered 5 tasks.  That is the *same
quantity* as the one here: both are the gap between what the agent scores and
what a label-free predictor scores.  A benchmark on which a majority-class
predictor nearly reaches the human baseline cannot detect label leakage either
-- the falsifiability floor and the leakage-detectability floor are one number
read twice.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ads.openml_io import get_task, task_splits  # noqa: E402

BENCH = REPO / "runs" / "bench"
OUT = REPO / "runs" / "leakage_power.json"
NOISE_SIGMAS = 2.0          # same constant as leakage_probe.py, on purpose


def true_fold_rates(task_id: int) -> tuple[dict, dict] | None:
    """Majority rate per (repeat, fold) from the task's OWN splits.

    Also validates the partition, which is the thing the count-based recovery
    could not do: within each repeat the test index sets must be pairwise
    disjoint and cover every row exactly once.
    """
    import numpy as np
    task = get_task(task_id)
    _, y = task.get_X_and_y(dataset_format="dataframe")
    y = np.asarray(y)
    n = len(y)

    rates, seen_by_repeat = {}, defaultdict(list)
    for r, f, tr, te in task_splits(task):
        te = np.asarray(te)
        if np.intersect1d(te, np.asarray(tr)).size:
            return None                     # train and test overlap
        vals, counts = np.unique(y[te], return_counts=True)
        rates[(int(r), int(f))] = {
            "n_test": int(te.size),
            "majority_rate": float(counts.max() / counts.sum()),
            "test_class_counts": {str(v): int(c) for v, c in zip(vals, counts)},
        }
        seen_by_repeat[int(r)].append(te)

    for r, sets in seen_by_repeat.items():
        allrows = np.concatenate(sets)
        if allrows.size != n or np.unique(allrows).size != n:
            # not a partition of the dataset: overlapping or incomplete folds
            return None
    return rates, {"n_repeats": len(seen_by_repeat),
                   "n_folds_per_repeat": len(seen_by_repeat[0]),
                   "n_rows": n,
                   "partition_validated": True}


def fold_majority_rates(record: dict) -> list[dict] | None:
    """Cross-check only: recover the test-fold rates from the train counts.

    Kept because it is free and it catches a record whose per-fold profiles do
    not describe the partition the task actually has.  NOT the source of any
    number that leaves this script -- see the module docstring.
    """
    folds = record.get("per_fold") or []
    if not folds or record.get("n_folds_run") != len(folds):
        return None
    k = len(folds)
    if k < 2:
        return None
    totals: dict[str, float] = defaultdict(float)
    for f in folds:
        counts = ((f.get("profile") or {}).get("class_counts")) or {}
        if not counts:
            return None
        for cls, n in counts.items():
            totals[cls] += n
    # Each example appears in the training fold of exactly k-1 of the k folds.
    totals = {c: n / (k - 1) for c, n in totals.items()}
    if any(abs(n - round(n)) > 1e-6 for n in totals.values()):
        # Not a clean k-fold partition (a repeated or held-out procedure), so
        # the identity above does not hold and the honest answer is to refuse.
        return None
    totals = {c: int(round(n)) for c, n in totals.items()}

    out = []
    for f in folds:
        train = (f.get("profile") or {}).get("class_counts") or {}
        test = {c: totals[c] - int(train.get(c, 0)) for c in totals}
        if any(v < 0 for v in test.values()):
            return None
        n_test = sum(test.values())
        if n_test == 0 or n_test != f.get("n_test"):
            # The recovered test-fold size must equal the size the run
            # recorded, or the identity is being applied to the wrong shape.
            return None
        p_maj = max(test.values()) / n_test
        out.append({
            "fold": f.get("fold"), "n_test": n_test,
            "test_class_counts": test,
            "majority_rate": float(p_maj),
            "accuracy": float(f["accuracy"]),
        })
    return out


def main() -> int:
    base = json.loads((REPO / "runs" / "baselines.json").read_text())
    order = list(base.get("selected_task_ids") or [])
    names = {int(t["task_id"]): t["dataset_name"] for t in base["tasks"]}

    by_task: dict[int, list[dict]] = defaultdict(list)
    for p in sorted(BENCH.glob("task_*.json")):
        r = json.loads(p.read_text())
        if not r.get("complete") or r.get("role") not in (None, "confirmatory"):
            continue
        if r.get("max_folds") is not None:      # partial: excluded, as elsewhere
            continue
        r["_file"] = p.name
        by_task[int(r["task_id"])].append(r)

    tasks = {}
    for tid in order:
        recs = by_task.get(tid, [])
        if not recs:
            tasks[str(tid)] = {"task_id": tid, "dataset_name": names.get(tid),
                               "status": "[not measured]"}
            continue
        # Pool over seeds: one majority rate per fold (a property of the split,
        # identical across seeds) and the mean intact accuracy per fold.
        # The rates come from the task's own splits, validated as a partition.
        truth = true_fold_rates(tid)
        if truth is None:
            tasks[str(tid)] = {
                "task_id": tid, "dataset_name": names.get(tid),
                "status": "[not measured]",
                "reason": ("the task's splits are not a validated partition "
                           "of the dataset, so no fold majority rate is "
                           "well defined"),
            }
            continue
        rates, shape = truth
        if shape["n_repeats"] != 1:
            # The recovery cross-check below assumes a single repeat, and the
            # accuracy records key their per-fold entries on `fold`. Refuse
            # rather than silently collapse repeats -- codex's point about the
            # 9R divisor is the same defect one level up.
            tasks[str(tid)] = {
                "task_id": tid, "dataset_name": names.get(tid),
                "status": "[not measured]",
                "reason": (f"{shape['n_repeats']} repeats; this script is "
                           f"written for single-repeat k-fold"),
                "split_shape": shape}
            continue
        maj = {f: rates[(0, f)]["majority_rate"] for (_, f) in rates}
        n_test = {f: rates[(0, f)]["n_test"] for (_, f) in rates}

        per_fold_acc: dict[int, list[float]] = defaultdict(list)
        refused, disagreements = [], []
        for r in recs:
            fr = fold_majority_rates(r)
            if fr is None:
                refused.append(r["_file"])
                continue
            # Cross-check: the record's own per-fold profiles must describe the
            # partition the task actually has. A disagreement means the record
            # was produced against different splits than the ones scored here.
            for d in fr:
                t = rates.get((0, d["fold"]))
                if t is None or d["n_test"] != t["n_test"] or abs(
                        d["majority_rate"] - t["majority_rate"]) > 1e-12:
                    disagreements.append(
                        f"{r['_file']}:fold{d['fold']}")
            for d in fr:
                per_fold_acc[d["fold"]].append(d["accuracy"])
        if disagreements:
            tasks[str(tid)] = {
                "task_id": tid, "dataset_name": names.get(tid),
                "status": "[not measured]",
                "reason": ("record-recovered fold rates disagree with the "
                           "task's own splits"),
                "disagreements": disagreements[:12],
                "split_shape": shape}
            continue
        if not per_fold_acc:
            tasks[str(tid)] = {
                "task_id": tid, "dataset_name": names.get(tid),
                "status": "[not measured]",
                "reason": "no complete record carried per-fold accuracies",
                "records_refused": refused}
            continue

        rows = []
        for f in sorted(per_fold_acc):
            acc = statistics.fmean(per_fold_acc[f])
            p, n = maj[f], n_test[f]
            band = NOISE_SIGMAS * math.sqrt(p * (1 - p) / n)
            gap = acc - p
            rows.append({
                "fold": f, "n_test": n, "majority_rate": p,
                "accuracy_intact_mean_over_seeds": acc,
                "n_seeds": len(per_fold_acc[f]),
                "gap": gap, "noise_band_2sigma": band,
                # The smallest fraction of the gap a leak must recover to be
                # seen.  >=1 means complete leakage is invisible.
                "min_resolvable_leak_fraction": (
                    float("inf") if gap <= 0 else band / gap),
                "detects_complete_leakage": bool(gap > band),
            })

        # How many folds must the probe pool before it can see a complete
        # leak?  The band shrinks like 1/sqrt(n_test) while the gap does not
        # move, so this is the honest cost of a powered probe on this task --
        # and it is why the pooled and per-fold readings differ so much.
        # Folds are pooled in the order the probe runs them (fold 0 upward),
        # not best-first, because that is the order a `--folds F` run gets.
        min_folds = None
        for F in range(1, len(rows) + 1):
            sub = rows[:F]
            n_ = sum(r["n_test"] for r in sub)
            a_ = sum(r["accuracy_intact_mean_over_seeds"] * r["n_test"]
                     for r in sub) / n_
            p_ = sum(r["majority_rate"] * r["n_test"] for r in sub) / n_
            b_ = NOISE_SIGMAS * math.sqrt(p_ * (1 - p_) / n_)
            if a_ - p_ > b_:
                min_folds = F
                break

        n_det = sum(r["detects_complete_leakage"] for r in rows)
        finite = [r["min_resolvable_leak_fraction"] for r in rows
                  if math.isfinite(r["min_resolvable_leak_fraction"])]
        # Pooled reading: the probe scores a whole fold set, so pool the folds.
        n_tot = sum(r["n_test"] for r in rows)
        acc_pool = sum(r["accuracy_intact_mean_over_seeds"] * r["n_test"]
                       for r in rows) / n_tot
        p_pool = sum(r["majority_rate"] * r["n_test"] for r in rows) / n_tot
        band_pool = NOISE_SIGMAS * math.sqrt(p_pool * (1 - p_pool) / n_tot)
        gap_pool = acc_pool - p_pool
        tasks[str(tid)] = {
            "task_id": tid, "dataset_name": names.get(tid),
            "status": "measured",
            "n_records": len(recs), "records_refused": refused,
            "split_shape": shape,
            "rates_source": "the task's own splits (.omlcache), partition validated",
            "recovery_cross_check": "agrees on every fold of every record",
            "per_fold": rows,
            "n_folds_detecting_complete_leakage": n_det,
            "n_folds": len(rows),
            "min_folds_for_detection": min_folds,
            "median_min_resolvable_leak_fraction": (
                statistics.median(finite) if finite else None),
            "pooled": {
                "n_test_total": n_tot,
                "accuracy_intact": acc_pool,
                "majority_rate": p_pool,
                "gap": gap_pool,
                "noise_band_2sigma": band_pool,
                "min_resolvable_leak_fraction": (
                    float("inf") if gap_pool <= 0 else band_pool / gap_pool),
                "detects_complete_leakage": bool(gap_pool > band_pool),
            },
        }

    powered = [t for t in tasks.values()
               if t.get("status") == "measured"
               and t["pooled"]["detects_complete_leakage"]]
    record = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": ("power of the label-permutation leakage probe, per task, "
                    "computed from records already on disk"),
        "derivation": (
            "majority rates read from each task's own splits in the repo-local "
            ".omlcache, with the partition validated (test folds pairwise "
            "disjoint and covering every row exactly once per repeat). The "
            "count-based recovery total=sum(train)/(k-1) is retained only as a "
            "cross-check on each record and must agree exactly; see the module "
            "docstring for codex's counterexample against relying on it."),
        "noise_sigmas": NOISE_SIGMAS,
        "no_new_runs": True,
        "tasks": tasks,
        "n_tasks_powered_pooled": len(powered),
        "tasks_powered_pooled": sorted(t["task_id"] for t in powered),
        # What `report.py`'s clause-2 gate reads: a task is only cleared by a
        # probe run on at least this many folds.  A task absent from this map
        # cannot be cleared by this instrument at any fold count.
        "min_folds_for_detection": {
            str(t["task_id"]): t["min_folds_for_detection"]
            for t in tasks.values()
            if t.get("status") == "measured" and t.get("min_folds_for_detection")},
        "tasks_unpowered_pooled": sorted(
            t["task_id"] for t in tasks.values()
            if t.get("status") == "measured"
            and not t["pooled"]["detects_complete_leakage"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.part")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
        fh.flush()
        import os
        os.fsync(fh.fileno())
    tmp.rename(OUT)

    print(f"{'task':>7} {'dataset':<34} {'intact':>7} {'p_maj':>7} "
          f"{'gap':>8} {'band':>7} {'phi_min':>8}  detects?")
    for tid in order:
        t = tasks[str(tid)]
        if t.get("status") != "measured":
            print(f"{tid:>7} {str(t.get('dataset_name')):<34} "
                  f"{'[not measured]':>7}")
            continue
        q = t["pooled"]
        phi = q["min_resolvable_leak_fraction"]
        print(f"{tid:>7} {t['dataset_name']:<34} {q['accuracy_intact']:>7.4f} "
              f"{q['majority_rate']:>7.4f} {q['gap']:>+8.4f} "
              f"{q['noise_band_2sigma']:>7.4f} "
              f"{(f'{phi:.3f}' if math.isfinite(phi) else 'inf'):>8}  "
              f"{'YES' if q['detects_complete_leakage'] else 'NO':<4} "
              f"folds>={t['min_folds_for_detection'] or '-'} "
              f"(per-fold {t['n_folds_detecting_complete_leakage']}"
              f"/{t['n_folds']})")
    print(f"\npowered (pooled): {record['tasks_powered_pooled']}")
    print(f"unpowered      : {record['tasks_unpowered_pooled']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
