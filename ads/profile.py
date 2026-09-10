"""Dataset profiling.

Everything the agent is allowed to know before it chooses anything.  Computed
from the **training fold only** — a profile that touched the outer test fold
would leak the thing we are measuring.  `assert_no_test_leak` in the tests
holds us to that.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Profile:
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_classes: int
    class_counts: dict
    minority_ratio: float
    frac_missing_cells: float
    cols_with_missing: int
    max_cardinality: int
    median_cardinality: float
    p_over_n: float
    numeric_scale_ratio: float = field(default=float("nan"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_frame(X: pd.DataFrame, y: pd.Series) -> Profile:
    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])
                or isinstance(X[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X.columns if c not in cat_cols]

    counts = pd.Series(y).value_counts()
    n = len(X)
    cards = [int(X[c].nunique(dropna=True)) for c in cat_cols] or [0]

    # spread of numeric magnitudes: decides whether scaling can matter at all
    scale_ratio = float("nan")
    if num_cols:
        sd = X[num_cols].astype("float64").std(numeric_only=True)
        sd = sd[np.isfinite(sd) & (sd > 0)]
        if len(sd) >= 2:
            scale_ratio = float(sd.max() / sd.min())

    return Profile(
        n_samples=int(n),
        n_features=int(X.shape[1]),
        n_numeric=len(num_cols),
        n_categorical=len(cat_cols),
        n_classes=int(len(counts)),
        class_counts={str(k): int(v) for k, v in counts.items()},
        minority_ratio=float(counts.min() / n) if len(counts) else float("nan"),
        frac_missing_cells=float(X.isna().to_numpy().mean()),
        cols_with_missing=int((X.isna().sum() > 0).sum()),
        max_cardinality=int(max(cards)),
        median_cardinality=float(np.median(cards)),
        p_over_n=float(X.shape[1] / n),
        numeric_scale_ratio=scale_ratio,
    )


def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns
           if not pd.api.types.is_numeric_dtype(X[c])
           or isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    return num, cat
