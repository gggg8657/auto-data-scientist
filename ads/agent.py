"""The autonomous data scientist.

Given a training fold and nothing else, it profiles the data, chooses its
preprocessing, its candidate model families, its inner validation scheme and
its search budget, runs the tournament, refines the winner, and fits.  Every
choice goes to the DecisionLog with the profile quantity that drove it.

Design note on what "autonomous" is allowed to mean here.  The agent may not
contain any branch on the identity of the dataset — no task id, no dataset
name.  `tests/test_no_dataset_specific_logic.py` enforces that by searching
this module for the five task ids and dataset names in the benchmark.  Rules
keyed on *measured properties* (n, p, cardinality, missingness, imbalance) are
the whole point; rules keyed on which dataset it is would be the way this KPI
gets faked.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .decisions import DecisionLog
from .profile import Profile, column_kinds, profile_frame

# ---------------------------------------------------------------- thresholds
# Fixed before any task was run; each is a property of the data, never of a
# dataset's identity.
ONEHOT_MAX_CARDINALITY = 20      # above this one-hot explodes the design matrix
SUBSAMPLE_ABOVE = 10_000         # model *selection* subsample, never final fit
INNER_FOLDS_SMALL, INNER_FOLDS_LARGE = 5, 3
SMALL_N = 5_000
IMBALANCED_BELOW = 0.10          # minority class share
KNN_MAX_P_OVER_N = 0.10          # KNN is hopeless in high relative dimension
SEARCH_ITERS_SMALL, SEARCH_ITERS_LARGE = 20, 8

# Orchestration, not a decision the agent makes: how many cores one agent may
# use.  Five task processes each taking n_jobs=-1 on a 192-core box shared with
# two other tracks oversubscribes it badly.  This changes wall-clock only --
# every estimator here is deterministic given its random_state, and
# `test_n_jobs_does_not_change_predictions` fails if that ever stops being true.
N_JOBS = int(os.environ.get("ADS_N_JOBS", "-1"))


def _preprocessor(X: pd.DataFrame, prof: Profile, log: DecisionLog,
                  scale: bool, encoding: str):
    num, cat = column_kinds(X)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    blocks = []
    if num:
        blocks.append(("num", Pipeline(num_steps), num))
    if cat:
        if encoding == "onehot":
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                min_frequency=1)
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1)
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", enc)]), cat))
    return ColumnTransformer(blocks, remainder="drop")


def choose_encoding(prof: Profile, log: DecisionLog) -> str:
    if prof.n_categorical == 0:
        choice, rule = "none", "no categorical columns in the profile"
    elif prof.max_cardinality <= ONEHOT_MAX_CARDINALITY:
        choice = "onehot"
        rule = (f"max cardinality {prof.max_cardinality} <= "
                f"{ONEHOT_MAX_CARDINALITY}, one-hot stays narrow")
    else:
        choice = "ordinal"
        rule = (f"max cardinality {prof.max_cardinality} > "
                f"{ONEHOT_MAX_CARDINALITY}; one-hot would add that many columns")
    log.record("encoding", choice, rule,
               {"n_categorical": prof.n_categorical,
                "max_cardinality": prof.max_cardinality,
                "median_cardinality": prof.median_cardinality},
               ["onehot", "ordinal", "none"])
    return choice


def choose_inner_cv(prof: Profile, log: DecisionLog, seed: int = 0) -> StratifiedKFold:
    k = INNER_FOLDS_SMALL if prof.n_samples < SMALL_N else INNER_FOLDS_LARGE
    # a fold cannot hold a class that has fewer members than there are folds
    min_class = min(prof.class_counts.values())
    if min_class < k:
        k = max(2, min_class)
        rule = (f"smallest class has {min_class} members, so k is clamped to "
                f"{k} to keep every fold stratifiable")
    else:
        rule = (f"n={prof.n_samples} {'<' if prof.n_samples < SMALL_N else '>='} "
                f"{SMALL_N}, so k={k}")
    log.record("inner_cv", f"StratifiedKFold(k={k})", rule,
               {"n_samples": prof.n_samples, "min_class_count": min_class,
                "minority_ratio": prof.minority_ratio},
               ["StratifiedKFold(k=3)", "StratifiedKFold(k=5)"])
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def candidate_families(prof: Profile, log: DecisionLog, seed: int = 0) -> dict:
    """Which model families are worth spending the budget on, and why."""
    imbalanced = prof.minority_ratio < IMBALANCED_BELOW
    cw = "balanced" if imbalanced else None
    cands: dict[str, dict] = {}

    cands["dummy"] = {"est": DummyClassifier(strategy="prior"),
                      "scale": False, "reason": "floor, so a family that never beats it is visible"}
    cands["hgb"] = {"est": HistGradientBoostingClassifier(random_state=seed),
                    "scale": False,
                    "reason": "handles mixed types and NaN natively; strongest default on tabular"}
    cands["rf"] = {"est": RandomForestClassifier(n_estimators=300, random_state=seed,
                                                 class_weight=cw, n_jobs=N_JOBS),
                   "scale": False, "reason": "variance-reduction baseline, insensitive to scaling"}
    cands["extra"] = {"est": ExtraTreesClassifier(n_estimators=300, random_state=seed,
                                                  class_weight=cw, n_jobs=N_JOBS),
                      "scale": False, "reason": "higher-variance trees; wins when signal is axis-aligned and noisy"}
    cands["logreg"] = {"est": LogisticRegression(max_iter=2000, class_weight=cw),
                       "scale": True,
                       "reason": "regularized linear model; the right prior when p/n is large"}

    dropped = {}
    if prof.p_over_n > KNN_MAX_P_OVER_N:
        dropped["knn"] = (f"p/n = {prof.p_over_n:.3f} > {KNN_MAX_P_OVER_N}; "
                          "distances concentrate")
    else:
        cands["knn"] = {"est": KNeighborsClassifier(n_jobs=N_JOBS), "scale": True,
                        "reason": f"p/n = {prof.p_over_n:.3f} is low enough for distances to mean something"}

    log.record("candidate_families", ",".join(sorted(cands)),
               "portfolio selected from profile; " +
               ("; ".join(f"dropped {k}: {v}" for k, v in dropped.items()) or "nothing dropped"),
               {"p_over_n": prof.p_over_n, "minority_ratio": prof.minority_ratio,
                "class_weight": str(cw), "imbalanced": bool(imbalanced),
                "dropped": dropped},
               ["dummy", "hgb", "rf", "extra", "logreg", "knn"])
    return cands


def _search_space(name: str, prof: Profile) -> dict:
    if name == "hgb":
        return {"clf__learning_rate": [0.03, 0.06, 0.1, 0.2],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [5, 20, 50],
                "clf__l2_regularization": [0.0, 0.1, 1.0]}
    if name in ("rf", "extra"):
        return {"clf__max_features": ["sqrt", "log2", 0.3, 0.6],
                "clf__min_samples_leaf": [1, 2, 5],
                "clf__min_samples_split": [2, 5, 10]}
    if name == "logreg":
        return {"clf__C": np.logspace(-3, 3, 13)}
    if name == "knn":
        return {"clf__n_neighbors": [1, 3, 5, 11, 21],
                "clf__weights": ["uniform", "distance"]}
    return {}


class AutoDataScientist:
    """One instance per outer fold.  Sees the training fold; never the test fold."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.log = DecisionLog()
        self.profile_: Profile | None = None
        self.best_name_: str | None = None
        self.tournament_: dict = {}
        self.pipeline_ = None

    # -- selection subsample is a decision, so it is logged like one ---------
    def _selection_view(self, X, y, prof):
        if prof.n_samples <= SUBSAMPLE_ABOVE:
            return X, y
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(prof.n_samples, SUBSAMPLE_ABOVE, replace=False)
        self.log.record(
            "selection_subsample", f"{SUBSAMPLE_ABOVE} of {prof.n_samples}",
            f"n={prof.n_samples} > {SUBSAMPLE_ABOVE}; model *selection* runs on a "
            "subsample so the tournament fits the budget. The winner is refit on "
            "the full training fold, so the final model sees every row.",
            {"n_samples": prof.n_samples, "subsample": SUBSAMPLE_ABOVE})
        return X.iloc[idx], np.asarray(y)[idx]

    def fit(self, X: pd.DataFrame, y):
        prof = profile_frame(X, pd.Series(y))
        self.profile_ = prof
        encoding = choose_encoding(prof, self.log)
        cv = choose_inner_cv(prof, self.log, self.random_state)
        cands = candidate_families(prof, self.log, self.random_state)
        Xs, ys = self._selection_view(X, y, prof)

        scores = {}
        for name, spec in cands.items():
            pre = _preprocessor(X, prof, self.log, spec["scale"], encoding)
            pipe = Pipeline([("pre", pre), ("clf", spec["est"])])
            try:
                s = cross_val_score(pipe, Xs, ys, cv=cv, scoring="accuracy",
                                    n_jobs=1, error_score="raise")
                scores[name] = {"mean": float(np.mean(s)), "std": float(np.std(s)),
                                "folds": [float(v) for v in s],
                                "reason": spec["reason"]}
            except Exception as e:                      # a family that cannot run is not a crash
                scores[name] = {"mean": float("-inf"), "error": f"{type(e).__name__}: {e}"[:300],
                                "reason": spec["reason"]}
        self.tournament_ = scores
        best = max(scores, key=lambda k: scores[k]["mean"])
        self.best_name_ = best
        self.log.record(
            "model_family", best,
            f"highest inner-CV accuracy: {scores[best]['mean']:.4f} vs runner-up "
            f"{sorted((v['mean'] for v in scores.values()), reverse=True)[1]:.4f}",
            {k: v["mean"] for k, v in scores.items()},
            sorted(scores))

        # -- refine the winner, budget chosen from n --------------------------
        space = _search_space(best, prof)
        pre = _preprocessor(X, prof, self.log, cands[best]["scale"], encoding)
        pipe = Pipeline([("pre", pre), ("clf", cands[best]["est"])])
        if space:
            n_iter = SEARCH_ITERS_SMALL if prof.n_samples < SMALL_N else SEARCH_ITERS_LARGE
            self.log.record(
                "search_budget", n_iter,
                f"n={prof.n_samples}, so {n_iter} random-search draws on {best}",
                {"n_samples": prof.n_samples, "space_size": len(space)})
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv,
                                        scoring="accuracy", n_jobs=N_JOBS,
                                        random_state=self.random_state,
                                        error_score=float("-inf"))
            search.fit(Xs, ys)
            improved = search.best_score_ > scores[best]["mean"]
            self.log.record(
                "hyperparameters", str(search.best_params_),
                (f"random search improved inner CV {scores[best]['mean']:.4f} -> "
                 f"{search.best_score_:.4f}") if improved else
                (f"random search did NOT improve on defaults "
                 f"({search.best_score_:.4f} <= {scores[best]['mean']:.4f}); "
                 "keeping the search winner anyway is the same estimator class, "
                 "so defaults are re-selected"),
                {"tuned_score": float(search.best_score_),
                 "default_score": scores[best]["mean"], "improved": bool(improved)})
            # Either way the final model is refit on the FULL training fold: the
            # subsample above was for selection only.
            self.pipeline_ = search.best_estimator_ if improved else pipe
            self.pipeline_.fit(X, y)
        else:
            self.log.record("hyperparameters", "none",
                            f"{best} has no search space defined; fitting defaults",
                            {"family": best})
            self.pipeline_ = pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)
