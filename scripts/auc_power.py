"""Test the registered prediction: does the rank reading have power where accuracy has none?

`critique_log.md` turn 10 registered this before any of it was run:

> phi_min under the AUROC reading will be below 1.0 on **all five** tasks, and
> the two currently-unprobeable tasks (3917 kc1, 10101 blood-transfusion) will
> become probeable at a fold count of 1 or 2.  If instead AUROC_intact on
> blood-transfusion comes out near 0.5, the hypothesis is wrong and the honest
> conclusion is that the agent has no rank information on that task either --
> which would be a statement about the benchmark, not about the instrument, and
> would be the more interesting outcome.

Why this is one fit per task and not eleven
-------------------------------------------
`phi_min` needs only the *honest* arm:

    accuracy reading:  phi_min = band(p_maj, n) / (accuracy_intact - p_maj)
    rank reading:      phi_min = band_MW(n1, n2) / (AUROC_intact      - 0.5)

`accuracy_intact` was already on disk in the 8-seed records, which is why
`leakage_power.py` needed no fits at all.  `AUROC_intact` is not, because no
record stores probabilities -- so it costs one fit per task and no permutations.
The permuted arm is what a leakage *verdict* needs, and that is
`leakage_probe.py`'s job; this is the power analysis that decides whether
running it on a given task is worth anything.

Scope, stated because it is narrower than `leakage_power.py`'s
--------------------------------------------------------------
This runs **fold 0 only**, at one seed. The band is therefore the widest the
task offers -- pooling folds shrinks it -- so a `phi_min` measured here is an
upper bound on the pooled one, and the direction is against the hypothesis.
`n_seeds=1` means no run-to-run spread is measured, so this is a **screen, not
a verdict**, in the sense the brief fixed: it decides where to point an
instrument, it decides no clause, and it is marked `enters_no_clause`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ads.openml_io import get_task, task_splits          # noqa: E402
from scripts.leakage_probe import (_auc, _brier,          # noqa: E402
                                   _fit_predict, detection_threshold,
                                   mannwhitney_auc_threshold)
from scripts.run_benchmark import (agent_source_digest,   # noqa: E402
                                   environment)

OUT = REPO / "runs" / "auc_power.json"


def main() -> int:
    power = json.loads((REPO / "runs" / "leakage_power.json").read_text())
    base = json.loads((REPO / "runs" / "baselines.json").read_text())
    order = list(base.get("selected_task_ids") or [])
    seed = int(os.environ.get("AUC_POWER_SEED", "0"))

    tasks = {}
    for tid in order:
        acc_t = power["tasks"].get(str(tid)) or {}
        task = get_task(tid)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        y = pd.Series(y)

        r, f, tr, te = next(iter(task_splits(task)))
        t0 = time.time()
        pred, family, scores = _fit_predict(X.iloc[tr], y.iloc[tr].to_numpy(),
                                            X.iloc[te], seed)
        secs = round(time.time() - t0, 1)

        y_te = y.iloc[te].to_numpy()
        auc = _auc(y_te, scores)
        thr = mannwhitney_auc_threshold(y_te)

        import numpy as np
        vals, counts = np.unique(y_te, return_counts=True)
        p_maj = float(counts.max() / counts.sum())
        n_te = int(counts.sum())
        acc = float((pred == y_te).mean())
        acc_band = detection_threshold(p_maj, n_te) - p_maj
        acc_gap = acc - p_maj

        rank = None
        if auc is not None and thr is not None:
            band, gap = thr - 0.5, auc - 0.5
            rank = {
                "auc_intact": auc,
                "brier_intact": _brier(y_te, scores),
                "noise_band_mannwhitney_2sigma": band,
                "gap": gap,
                "min_resolvable_leak_fraction": (
                    None if gap <= 0 else band / gap),
                "detects_complete_leakage": bool(gap > band),
            }
        tasks[str(tid)] = {
            "task_id": tid, "dataset_name": task.get_dataset().name,
            "fold": int(f), "repeat": int(r), "random_state": seed,
            "n_test": n_te, "seconds": secs, "family": family,
            "probabilities_available": scores is not None,
            "accuracy_reading_this_fold": {
                "accuracy_intact": acc, "majority_rate": p_maj,
                "noise_band_2sigma": acc_band, "gap": acc_gap,
                "min_resolvable_leak_fraction": (
                    None if acc_gap <= 0 else acc_band / acc_gap),
                "detects_complete_leakage": bool(acc_gap > acc_band),
            },
            # For context only: the pooled 8-seed accuracy reading already on
            # disk. Not comparable band-for-band with the single fold above.
            "accuracy_reading_pooled_8seed": (
                acc_t.get("pooled") if acc_t.get("status") == "measured"
                else None),
            "rank_reading_this_fold": rank,
        }
        print(f"  task {tid:>6} {tasks[str(tid)]['dataset_name']:<34} "
              f"acc {acc:.4f} (p_maj {p_maj:.4f}) | AUROC "
              f"{(f'{auc:.4f}' if auc is not None else 'n/a')} | "
              f"phi_acc "
              f"{_fmt(tasks[str(tid)]['accuracy_reading_this_fold']['min_resolvable_leak_fraction'])}"
              f" | phi_auc "
              f"{_fmt(rank['min_resolvable_leak_fraction'] if rank else None)}"
              f" | {family} ({secs}s)", flush=True)

    measured = [t for t in tasks.values() if t.get("rank_reading_this_fold")]
    powered_rank = sorted(t["task_id"] for t in measured
                          if t["rank_reading_this_fold"]["detects_complete_leakage"])
    powered_acc = sorted(t["task_id"] for t in tasks.values()
                         if t["accuracy_reading_this_fold"]["detects_complete_leakage"])
    record = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": ("test the turn-10 registered prediction that the rank "
                    "reading has power where the accuracy reading has none"),
        "registered_prediction": (
            "phi_min < 1 under the AUROC reading on all five tasks, and the "
            "two unprobeable tasks (3917, 10101) become probeable at 1-2 folds"),
        "scope": ("fold 0 only, one seed, so the band is the widest the task "
                  "offers and phi_min here is an upper bound on the pooled "
                  "value. A screen, not a verdict."),
        "enters_no_clause": True,
        "n_fits": len(tasks),
        "random_state": seed,
        "env": {**agent_source_digest(), **environment()},
        "tasks": tasks,
        "tasks_powered_under_rank_reading_fold0": powered_rank,
        "tasks_powered_under_accuracy_reading_fold0": powered_acc,
        "prediction_holds_on_all_five": (
            len(powered_rank) == len(order) and len(measured) == len(order)),
        "tasks_rescued_by_the_rank_reading": sorted(
            set(powered_rank) - set(powered_acc)),
    }
    tmp = OUT.with_suffix(".json.part")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.rename(OUT)

    print(f"\npowered under accuracy (fold 0): {powered_acc}")
    print(f"powered under rank     (fold 0): {powered_rank}")
    print(f"rescued by the rank reading    : "
          f"{record['tasks_rescued_by_the_rank_reading']}")
    print(f"registered prediction holds on all five: "
          f"{record['prediction_holds_on_all_five']}")
    print(f"-> {OUT}")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    return "  inf" if not math.isfinite(v) else f"{v:5.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
