"""The agent must run, log, and beat its own floor, without seeing a test fold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ads.agent import AutoDataScientist, choose_encoding, choose_inner_cv
from ads.decisions import DecisionLog
from ads.profile import profile_frame


def synthetic(n=400, seed=0, missing=True, high_card=True):
    rng = np.random.RandomState(seed)
    z = rng.normal(size=n)
    X = pd.DataFrame({
        "a": z + rng.normal(scale=0.5, size=n),
        "b": rng.normal(scale=1000.0, size=n),        # a wildly different scale
        "c": rng.normal(scale=0.001, size=n),
        "cat_low": rng.choice(list("xyz"), size=n),
        "cat_high": rng.choice([f"v{i}" for i in range(60)], size=n)
        if high_card else rng.choice(list("pq"), size=n),
    })
    if missing:
        X.loc[rng.choice(n, n // 10, replace=False), "a"] = np.nan
        X.loc[rng.choice(n, n // 20, replace=False), "cat_low"] = np.nan
    y = (z + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return X, y


def test_fit_predict_and_beat_the_floor():
    X, y = synthetic()
    tr, te = np.arange(300), np.arange(300, 400)
    ag = AutoDataScientist(random_state=0).fit(X.iloc[tr], y[tr])
    pred = ag.predict(X.iloc[te])
    assert len(pred) == len(te)
    acc = (pred == y[te]).mean()
    assert acc > 0.6, f"agent at {acc:.3f} on a learnable problem"
    assert ag.best_name_ != "dummy", "the tournament picked the prior over every model"
    print(f"  fit/predict ok: acc={acc:.3f} family={ag.best_name_}")


def test_every_decision_carries_evidence():
    X, y = synthetic()
    ag = AutoDataScientist(random_state=0).fit(X, y)
    d = ag.log.to_dict()
    assert d["n_decisions"] >= 4, d["n_decisions"]
    assert d["n_interventions"] == 0
    for e in d["decisions"]:
        assert e["evidence"], f"decision {e['stage']} has no evidence"
        assert e["rule"], f"decision {e['stage']} has no rule text"
    stages = {e["stage"] for e in d["decisions"]}
    for required in ("encoding", "inner_cv", "candidate_families",
                     "model_family", "hyperparameters"):
        assert required in stages, f"{required} was never logged as a decision"
    print(f"  {d['n_decisions']} decisions, all with evidence: {sorted(stages)}")


def test_profile_is_computed_on_the_training_fold_only():
    """A profile that saw the test fold would leak the thing we measure."""
    X, y = synthetic(n=400)
    tr = np.arange(300)
    ag = AutoDataScientist(random_state=0).fit(X.iloc[tr], y[tr])
    assert ag.profile_.n_samples == 300, ag.profile_.n_samples
    full = profile_frame(X, pd.Series(y))
    assert full.n_samples == 400
    assert ag.profile_.n_samples != full.n_samples
    print("  profile n_samples tracks the training fold, not the dataset")


def test_encoding_switches_on_cardinality_not_on_identity():
    log = DecisionLog()
    Xh, yh = synthetic(high_card=True)
    Xl, yl = synthetic(high_card=False)
    assert choose_encoding(profile_frame(Xh, pd.Series(yh)), log) == "ordinal"
    assert choose_encoding(profile_frame(Xl, pd.Series(yl)), log) == "onehot"
    print("  encoding: ordinal at high cardinality, one-hot at low")


def test_inner_cv_clamps_to_the_smallest_class():
    log = DecisionLog()
    X, y = synthetic(n=300)
    y = y.copy(); y[:] = 0; y[:3] = 1          # a 3-member minority class
    cv = choose_inner_cv(profile_frame(X, pd.Series(y)), log)
    assert cv.n_splits == 3, cv.n_splits
    print(f"  inner cv clamped to k={cv.n_splits} for a 3-member class")


def test_seeds_actually_change_the_agent():
    """If every estimator were pinned at random_state=0 a seed sweep would
    report a spread of zero and we would understate our own noise floor."""
    X, y = synthetic(n=400)
    a = AutoDataScientist(random_state=0).fit(X, y)
    b = AutoDataScientist(random_state=7).fit(X, y)
    ta = a.tournament_["rf"]["mean"]; tb = b.tournament_["rf"]["mean"]
    assert ta != tb, ("two seeds gave the identical rf tournament score; the "
                      "estimators are not keyed on the agent's seed")
    print(f"  seed 0 vs 7 rf inner-CV: {ta:.4f} vs {tb:.4f}")


def test_n_jobs_does_not_change_predictions():
    """ADS_N_JOBS is orchestration: it may change wall-clock, never a
    prediction. If it did, the core cap used to share this box with two other
    tracks would be a silent protocol change."""
    import importlib
    import os

    import ads.agent as A
    X, y = synthetic(n=400)
    tr, te = np.arange(300), np.arange(300, 400)

    preds = {}
    for nj in ("1", "4"):
        os.environ["ADS_N_JOBS"] = nj
        importlib.reload(A)
        assert A.N_JOBS == int(nj)
        ag = A.AutoDataScientist(random_state=0).fit(X.iloc[tr], y[tr])
        preds[nj] = (ag.best_name_, tuple(ag.predict(X.iloc[te])),
                     round(ag.tournament_[ag.best_name_]["mean"], 12))
    os.environ.pop("ADS_N_JOBS", None)
    importlib.reload(A)

    assert preds["1"] == preds["4"], (
        "ADS_N_JOBS changed the agent's predictions, so it is not a pure "
        "orchestration knob")
    print(f"  n_jobs 1 vs 4: identical predictions, family={preds['1'][0]}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
