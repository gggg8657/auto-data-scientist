"""Run the agent over a task's own estimation procedure.

The accuracy this produces is `predictive_accuracy` on the *same folds* the
published OpenML runs used, which is the only reason our number and the human
baseline can appear in the same table.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .agent import AutoDataScientist
from .openml_io import get_task, task_splits


def run_task(task_id: int, random_state: int = 0, max_folds: int | None = None,
             verbose: bool = True) -> dict:
    t_all = time.time()
    task = get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    y = pd.Series(y)

    folds, per_fold = [], []
    n_correct_total = n_pred_total = 0
    for r, f, tr, te in task_splits(task):
        if max_folds is not None and len(per_fold) >= max_folds:
            break
        t0 = time.time()
        agent = AutoDataScientist(random_state=random_state)
        agent.fit(X.iloc[tr], y.iloc[tr].to_numpy())
        pred = agent.predict(X.iloc[te])
        hit = (np.asarray(pred) == y.iloc[te].to_numpy())
        n_correct_total += int(hit.sum()); n_pred_total += int(hit.size)
        acc = float(hit.mean())
        per_fold.append({
            "repeat": int(r), "fold": int(f), "accuracy": acc,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "family": agent.best_name_,
            "seconds": round(time.time() - t0, 2),
            "tournament": {k: v.get("mean") for k, v in agent.tournament_.items()},
            "decisions": agent.log.to_dict(),
            "profile": agent.profile_.to_dict(),
        })
        folds.append(acc)
        if verbose:
            print(f"  task {task_id} fold {f}: acc={acc:.4f} "
                  f"family={agent.best_name_} ({per_fold[-1]['seconds']}s)",
                  flush=True)

    accs = np.array(folds)
    n_int = sum(p["decisions"]["n_interventions"] for p in per_fold)
    return {
        "task_id": int(task_id),
        "dataset_name": task.get_dataset().name,
        "data_id": int(task.dataset_id),
        "estimation_procedure": task.estimation_procedure["type"],
        "metric": "predictive_accuracy",
        "random_state": int(random_state),
        # OpenML computes predictive_accuracy over the pooled predictions of
        # every fold, so `accuracy_pooled` is the number comparable to the
        # published runs.  `accuracy_mean` (unweighted mean over folds) is
        # reported beside it; they differ only when folds are unequal.
        "accuracy_pooled": float(n_correct_total / n_pred_total),
        "n_correct": int(n_correct_total),
        "n_predictions": int(n_pred_total),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "accuracy_folds": [float(a) for a in accs],
        "n_folds_run": len(accs),
        "n_interventions": int(n_int),
        "families_chosen": sorted({p["family"] for p in per_fold}),
        "seconds_total": round(time.time() - t_all, 1),
        "per_fold": per_fold,
    }
