"""Open item #8, in-process label leakage, as tests rather than as a worry.

`scripts/leakage_probe.py` measures this on a real task once.  A measurement
that is not a test decays: the next edit to `ads/agent.py` can open the route
again and nothing goes red.  So the same three claims are asserted here, on
synthetic data, fast enough for CI:

1. the agent cannot *acquire* data -- its import closure contains nothing that
   reaches a network, a file or the OpenML cache;
2. severing `y_train` (permuting it) destroys the agent's accuracy, so
   `y_train` is carrying the signal and nothing else is;
3. neither array `ads/evaluate.py` hands the agent aliases anything holding a
   held-out label, and the parent frame is not reachable from them;
4. and the probe **fires on a leak it is told about** -- a positive control.

Claim 3 replaces a worse test I wrote earlier in the same turn.  Open item #8
was framed as "a pandas view keeps a reference to its parent, so the test
labels are reachable from the object the agent receives".  codex, asked the
constructive question, pointed out that this reachability was *assumed and not
demonstrated*, and it was right: on pandas 3.0.2 Copy-on-Write is mandatory,
`X.iloc[tr]` shares no memory with `X`, `y.iloc[tr].to_numpy()` shares no
memory with `y`, and no `gc` walk of four hops from a column selection reaches
the parent.  So the original view-vs-copy comparison could only ever pass: the
route it looked for does not exist here.  What is asserted instead is the
structural fact itself, which is exact, cheap, and goes red if a pandas upgrade
reintroduces aliasing.

Claim 4 is the one that makes the others mean anything.  codex again:
"an invariant result could simply mean the intervention never reached the
alleged channel".  So the probe's own decision rule is run against an agent
that deliberately reads the held-out labels and is required to flag it.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ads.agent import AutoDataScientist  # noqa: E402
from scripts.leakage_probe import (detection_threshold,  # noqa: E402
                                   flags_leakage,
                                   mannwhitney_auc_threshold, probe_fold)

# The agent's whole job is to reason over arrays it was handed.  Anything that
# can open a socket, a file or the OpenML cache is out of scope for it, and a
# new import landing here should have to argue for itself in a diff.
ALLOWED_TOP_LEVEL = {
    "__future__", "os", "math", "time", "typing", "dataclasses", "warnings",
    "numpy", "pandas", "scipy", "sklearn",
}
# Relative imports the agent may make inside this package.  `openml_io` and
# `evaluate` are deliberately absent: the first acquires data, the second holds
# the labelled frame.
ALLOWED_RELATIVE = {"decisions", "profile"}

AGENT_MODULES = ["ads/agent.py", "ads/profile.py", "ads/decisions.py"]


def _imports(path: Path):
    """(top_level_absolute, relative) module names imported by a source file."""
    tree = ast.parse(path.read_text())
    absolute, relative = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                absolute.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # from .x import y
                relative.add((node.module or "").split(".")[0])
            elif node.module:
                absolute.add(node.module.split(".")[0])
    return absolute, relative


@pytest.mark.parametrize("rel", AGENT_MODULES)
def test_agent_import_closure_cannot_acquire_data(rel):
    absolute, relative = _imports(REPO / rel)
    assert absolute <= ALLOWED_TOP_LEVEL, (
        f"{rel} imports {sorted(absolute - ALLOWED_TOP_LEVEL)}, which is outside "
        f"the allowlist. If this import is legitimate, add it to "
        f"ALLOWED_TOP_LEVEL in this test and say why in the diff -- the point "
        f"of the list is that widening it is visible.")
    assert relative <= ALLOWED_RELATIVE, (
        f"{rel} imports .{sorted(relative - ALLOWED_RELATIVE)}. The agent must "
        f"not be able to reach openml_io (acquires data) or evaluate (holds the "
        f"full labelled frame).")


@pytest.mark.parametrize("rel", AGENT_MODULES)
def test_agent_does_not_read_or_open_anything(rel):
    """No `open(...)`, no `read_csv`, no `__import__`.  A dynamic import would
    walk straight around the closure test above."""
    tree = ast.parse((REPO / rel).read_text())
    banned = {"open", "__import__", "eval", "exec", "compile"}
    banned_attrs = {"read_csv", "read_parquet", "read_json", "import_module",
                    "urlopen", "get_task"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                assert f.id not in banned, f"{rel} calls {f.id}()"
            elif isinstance(f, ast.Attribute):
                assert f.attr not in banned_attrs, f"{rel} calls .{f.attr}()"


def _dataset(n=400, p=6, seed=0):
    """A learnable but not trivially separable binary problem, imbalanced so
    the majority rate and 1/C are different numbers -- the whole point of the
    pre-registered rule is that the reference is the majority rate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logit = 2.0 * X[:, 0] + 1.2 * X[:, 1] - 0.8 * X[:, 2] + 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"f{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def _split(n, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def test_permuting_train_labels_destroys_accuracy():
    """Claim 2, and the synthetic mirror of `scripts/leakage_probe.py`.

    Same pre-registered rule as the script: the permuted arm must not exceed
    the test fold's majority rate plus a 2-sigma binomial band.
    """
    X, y = _dataset()
    tr, te = _split(len(y))
    y_te = y.iloc[te].to_numpy()
    y_tr = y.iloc[tr].to_numpy()

    p_maj = float(np.bincount(y_te).max() / len(y_te))
    band = 2.0 * math.sqrt(p_maj * (1 - p_maj) / len(y_te))

    intact = AutoDataScientist(random_state=0)
    intact.fit(X.iloc[tr], y_tr)
    acc_intact = float((np.asarray(intact.predict(X.iloc[te])) == y_te).mean())
    assert acc_intact > p_maj + band, (
        "the intact arm has to actually learn something, or this test cannot "
        f"distinguish anything: acc={acc_intact:.4f} vs majority {p_maj:.4f}")

    # Three permutations, not ten: this is the CI mirror, and the script is the
    # measurement.  Any one of them exceeding the band is the finding.
    for j in range(3):
        rng = np.random.default_rng(10_000 + j)
        perm = AutoDataScientist(random_state=0)
        perm.fit(X.iloc[tr], y_tr[rng.permutation(len(y_tr))])
        acc = float((np.asarray(perm.predict(X.iloc[te])) == y_te).mean())
        assert acc <= p_maj + band, (
            f"permutation {j}: fitting on shuffled labels still scored "
            f"{acc:.4f} against the true test labels, above the majority rate "
            f"{p_maj:.4f} + {band:.4f}. y_train is not the only channel to the "
            f"labels -- see critique_log.md turn 9, open item #8.")


MAX_GC_HOPS = 4


def _reaches(obj, target, hops=MAX_GC_HOPS) -> bool:
    """Is `target` reachable from `obj` by following object references?"""
    import gc
    seen, frontier = {id(obj)}, [obj]
    for _ in range(hops):
        nxt = []
        for o in frontier:
            for r in gc.get_referents(o):
                if id(r) in seen:
                    continue
                seen.add(id(r))
                if r is target:
                    return True
                nxt.append(r)
        frontier = nxt
    return False


def test_arrays_handed_to_the_agent_do_not_alias_the_held_out_labels():
    """Claim 3, using the exact expressions `ads/evaluate.py` evaluates.

    Not `assert isinstance(..., copy)` -- the question is whether the *bytes*
    the agent can address include a held-out label, so it is asked as a
    memory-sharing question.
    """
    X, y = _dataset()
    tr, te = _split(len(y))

    X_tr, X_te = X.iloc[tr], X.iloc[te]      # exactly evaluate.py's slices
    y_tr = y.iloc[tr].to_numpy()

    y_full = y.to_numpy()
    y_te_true = y.iloc[te].to_numpy()

    assert not np.shares_memory(y_tr, y_full), (
        "the label array handed to fit() shares a buffer with the full label "
        "vector, so an out-of-bounds or wrong-length read off it lands on "
        "held-out labels")
    assert not np.shares_memory(y_tr, y_te_true), (
        "the training labels alias the test labels")
    assert len(y_tr) == len(tr), "the training label array is the wrong length"

    for name, frame in (("X.iloc[tr]", X_tr), ("X.iloc[te]", X_te)):
        assert not any(np.shares_memory(a, y_full) for a in frame._mgr.arrays), (
            f"{name} shares memory with the labels")
        assert not _reaches(frame, y), (
            f"the label Series is reachable from {name} within "
            f"{MAX_GC_HOPS} reference hops")

    # And the framing that motivated #8 in the first place: labels sitting in
    # the same frame as the features.  Under Copy-on-Write a column selection
    # neither aliases nor reaches the parent.  If this ever fails, the
    # view-vs-copy question is live again and #8 reopens.
    full = X.copy()
    full["target"] = y.to_numpy()
    sel = full[list(X.columns)].iloc[tr]
    parent_labels = full["target"].to_numpy()
    assert not any(np.shares_memory(a, parent_labels) for a in sel._mgr.arrays), (
        f"pandas {pd.__version__}: a column selection of a labelled frame "
        f"aliases the label column again. Copy-on-Write no longer holds and "
        f"open item #8 reopens -- see critique_log.md turn 9.")
    assert not _reaches(sel, full), (
        f"pandas {pd.__version__}: the labelled parent frame is reachable from "
        f"a column selection of it. Open item #8 reopens.")


# --------------------------------------------------------------- claim 4
class _LeakyAgent:
    """An agent that cheats, for use as a positive control only.

    It ignores the `y` it is given and predicts from a lookup keyed on the
    feature rows, built over the *whole* dataset including the held-out fold.
    That is the leak the probe exists to catch: permuting `y_train` cannot
    touch it, so a probe that stays quiet here is measuring nothing.
    """

    def __init__(self, random_state: int = 0, table=None, with_proba=False):
        self.random_state = random_state
        self._table = table
        self.best_name_ = "leaky"
        # The probe reads probabilities off `pipeline_`, so a control for the
        # AUROC reading has to leak through that attribute too. Without this
        # the rank rule would never have been seen to fire -- the same
        # unvalidated-instrument problem one level down.
        self.pipeline_ = self if with_proba else None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def _lookup(self, X):
        return np.asarray([self._table[tuple(r)] for r in np.asarray(X)])

    def predict(self, X):
        return self._lookup(X)

    def predict_proba(self, X):
        y = self._lookup(X).astype(float)
        return np.column_stack([1.0 - y, y])


def test_the_probe_fires_on_a_leak_it_is_told_about():
    """The positive control. Same `probe_fold`, same rule, cheating agent."""
    X, y = _dataset()
    tr, te = _split(len(y))
    table = {tuple(r): int(v) for r, v in zip(X.to_numpy(), y.to_numpy())}

    d = probe_fold(X, y, tr, te, random_state=0, k=3,
                   make_agent=lambda random_state: _LeakyAgent(
                       random_state, table),
                   verbose=False)

    assert d["accuracy_intact"] == 1.0, (
        "the control agent should reproduce the labels exactly; it scored "
        f"{d['accuracy_intact']}")
    assert d["verdict_fold"] == "LEAKAGE", (
        "the probe did not flag an agent that reads the held-out labels "
        "directly. Every NO_LEAKAGE_DETECTED this instrument has ever emitted "
        f"is uninterpretable. permuted_max={d['permuted_max']}, "
        f"threshold={d['threshold']}")
    assert d["n_permutations_exceeding_threshold"] == 3, (
        "a complete leak must be flagged by every permutation, not one: "
        f"{d['n_permutations_exceeding_threshold']}/3")


def test_the_rank_reading_fires_on_the_same_leak():
    """Positive control for the AUROC reading registered in turn 10.

    Its whole justification is that the null AUROC is 0.5 whatever the prior
    is, so it has dynamic range where the accuracy reading has none. That is
    worth nothing until the rule has been seen to fire.
    """
    X, y = _dataset()
    tr, te = _split(len(y))
    table = {tuple(r): int(v) for r, v in zip(X.to_numpy(), y.to_numpy())}

    d = probe_fold(X, y, tr, te, random_state=0, k=3,
                   make_agent=lambda random_state: _LeakyAgent(
                       random_state, table, with_proba=True),
                   verbose=False)

    r = d["auc_reading"]
    assert r is not None, (
        "the rank reading was not computed at all, so it cannot have been "
        "validated; the control agent exposes predict_proba via pipeline_")
    assert r["auc_intact"] == 1.0, r["auc_intact"]
    assert r["null_value"] == 0.5
    assert r["verdict_fold"] == "LEAKAGE", (
        "the rank rule did not flag an agent whose probabilities ARE the "
        f"held-out labels: permuted_auc_max={r['permuted_auc_max']}, "
        f"threshold={r['threshold']}")
    # And the property that motivates it: the threshold does not move with the
    # class prior, unlike the accuracy band.
    balanced = np.r_[np.zeros(200, int), np.ones(200, int)]
    skewed = np.r_[np.zeros(360, int), np.ones(40, int)]
    t_bal = mannwhitney_auc_threshold(balanced)
    t_skew = mannwhitney_auc_threshold(skewed)
    assert t_bal > 0.5 and t_skew > 0.5
    assert abs((t_bal - 0.5) - (t_skew - 0.5)) < 0.05, (
        "the Mann-Whitney band should depend on n and balance only weakly; "
        f"balanced {t_bal:.4f} vs 9:1 skewed {t_skew:.4f}")
    # The accuracy band's REFERENCE, by contrast, is the prior itself: 0.5 vs
    # 0.9. That difference is the entire finding of runs/leakage_power.json.
    assert abs(detection_threshold(0.5, 400) - detection_threshold(0.9, 400)) > 0.3
    assert mannwhitney_auc_threshold(np.zeros(50, int)) is None, (
        "a fold with one class present has no defined AUROC and must not "
        "silently produce a threshold")
    print("  rank reading flags the leak; its null stays at 0.5 across a "
          "1:1 and a 9:1 prior while the accuracy reference moves 0.5 -> 0.9")


def test_the_decision_rule_is_one_sided_and_uses_the_majority_rate():
    """The rule itself, without fitting anything.

    Two properties that a two-sided or chance-referenced version would break:
    a permuted arm *below* the majority rate is not evidence, and the reference
    is the majority rate rather than 1/n_classes -- on an 85/15 problem those
    are 0.85 and 0.5, and referencing chance would flag every honest run.
    """
    p_maj, n = 0.85, 1000
    thr = detection_threshold(p_maj, n)
    assert p_maj < thr < 1.0
    assert not flags_leakage([p_maj, p_maj - 0.30, 0.0], p_maj, n), (
        "a shuffled-label fit that scores at or below the majority rate is an "
        "overfitting classifier, not a leak")
    assert flags_leakage([p_maj, thr + 1e-9], p_maj, n), (
        "a single permutation above the threshold is the finding")
    # The band must shrink with n, or pooling folds would buy no power and
    # `leakage_power.py`'s min_folds_for_detection would be meaningless.
    assert detection_threshold(p_maj, 10 * n) - p_maj < (thr - p_maj) / 3


if __name__ == "__main__":
    # CI runs each test file as `python <file>`, so a file with no __main__
    # block is a file CI reports green after running zero assertions. That
    # exact shape of omission is already in this repo's history twice, and
    # `test_every_test_file_is_actually_executable_by_ci` in
    # tests/test_registry_is_committed.py now asserts every file has one.
    # Parametrized cases are expanded by hand here rather than depending on
    # pytest being installed on the runner.
    ran = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m.args[1] for m in marks if m.name == "parametrize"]
        cases = params[0] if params else [None]
        for c in cases:
            print(f"{name}{'[' + str(c) + ']' if c is not None else ''} ...")
            fn() if c is None else fn(c)
            ran += 1
    print(f"\n{ran} tests passed")
