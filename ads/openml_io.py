"""All OpenML access lives here.

Two rules this module exists to enforce:

1. Every run is evaluated on the *task's own* estimation procedure — the same
   splits the published runs used — so our accuracy is comparable to theirs.
   If we invented our own CV we would be comparing to a different number and
   the KPI would be meaningless.
2. The cache is repo-local, so a run is reproducible from a clean checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

import openml

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / ".omlcache"


def configure() -> None:
    CACHE.mkdir(exist_ok=True)
    openml.config.set_root_cache_directory(str(CACHE))
    openml.config.retry_policy = "robot"


def get_task(task_id: int):
    configure()
    return openml.tasks.get_task(task_id)


def list_all_evaluations(task_id: int, metric: str = "predictive_accuracy",
                         page: int = 1000, hard_cap: int = 200_000):
    """Page through *every* published evaluation of `metric` on this task.

    `list_evaluations` refuses a limit above 10000 and silently truncates at
    `size`, so taking one page would sample the oldest runs by run_id and bias
    the median toward the 2014 Weka era.  We page to exhaustion instead.
    """
    configure()
    frames, offset = [], 0
    while offset < hard_cap:
        df = openml.evaluations.list_evaluations(
            metric, tasks=[task_id], size=page, offset=offset,
            output_format="dataframe")
        if df is None or len(df) == 0:
            break
        frames.append(df)
        if len(df) < page:
            break
        offset += page
    if not frames:
        return None
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def task_splits(task):
    """Yield (repeat, fold, train_idx, test_idx) for the task's own procedure."""
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    for r in range(n_repeats):
        for f in range(n_folds):
            for s in range(n_samples):
                tr, te = task.get_train_test_split_indices(fold=f, repeat=r, sample=s)
                yield r, f, tr, te
