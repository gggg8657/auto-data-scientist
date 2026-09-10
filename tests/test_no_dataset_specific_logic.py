"""Clause 3 ('무개입') is faked by branching on which dataset it is.

The agent is allowed to key on *measured properties* — n, p/n, cardinality,
missingness, imbalance. It is not allowed to key on identity. This test reads
the registry in `runs/baselines.json` and searches the package for every task id
and dataset name in it, so it tightens automatically if the registry changes.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PKG = REPO / "ads"


def _registry():
    p = REPO / "runs/baselines.json"
    if not p.exists():
        return [], []
    b = json.loads(p.read_text())
    sel = [t for t in b["tasks"] if t["selected"]]
    return [t["task_id"] for t in sel], [t["dataset_name"] for t in sel]


def _code_only(src: str) -> str:
    """Prose may discuss a dataset or the registry; executable code may not."""
    code = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)
    return re.sub(r"#.*", "", code)


def test_no_task_id_or_dataset_name_in_the_package():
    tids, names = _registry()
    if not tids:
        print("  runs/baselines.json absent; registry check skipped "
              "(the test is vacuous until stage 1 has run)")
        return
    bad = []
    # Task ids below 1000 cannot be scanned this way and pretending otherwise
    # is worse than not scanning them. Two of the five registered ids are 31
    # and 3, and `ads/agent.py` legitimately contains `max_leaf_nodes` of
    # [15, 31, 63] (sklearn's default is 31) and `INNER_FOLDS_LARGE = 3`. A
    # bare-integer search flags those forever, so the check was failing on
    # hyperparameters rather than on identity -- a guard that is always red
    # tells you nothing.
    #
    # The scan is therefore restricted to ids of four digits or more, where a
    # literal match is not plausibly a hyperparameter, and the property the
    # scan was standing in for is tested directly by
    # `test_decisions_are_invariant_to_column_names_and_order` below. Dataset
    # *names* are still scanned in full: a string like "credit-g" in the
    # package has no innocent reading.
    scannable = [t for t in tids if abs(int(t)) >= 1000]
    unscannable = [t for t in tids if abs(int(t)) < 1000]
    for f in sorted(PKG.glob("*.py")):
        code = _code_only(f.read_text())
        for tid in scannable:
            if re.search(rf"\b{tid}\b", code):
                bad.append(f"{f.name}: task id {tid}")
        for nm in names:
            if nm.lower() in code.lower():
                bad.append(f"{f.name}: dataset name {nm!r}")
    assert not bad, "dataset-specific logic in the agent: " + "; ".join(bad)
    print(f"  task ids not scanned as literals (ambiguous with "
          f"hyperparameters): {unscannable}")
    print(f"  none of {len(tids)} task ids or {len(names)} dataset names "
          f"appears in ads/*.py code")


def test_decisions_are_invariant_to_column_names_and_order():
    """The property the literal scan was a proxy for.

    An agent that keys on identity has to recognise the dataset, and the
    cheapest handle on identity is the schema: column names, their order, or a
    fingerprint built from them. codex made this point directly -- a string
    scan "cannot distinguish a general rule from an identity lookup disguised
    as a rule". This tests the invariance instead of the spelling: rename every
    column to an opaque label, permute the column order, and the agent's
    decisions and predictions must not move.

    It cannot prove the absence of an identity lookup keyed on *values*, and
    the docstring says so rather than implying a stronger guarantee. It does
    close the schema-fingerprint route, which is the one a rule keyed on
    "measured properties" can hide inside.
    """
    import numpy as np
    import pandas as pd
    from ads.agent import AutoDataScientist

    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.normal(50_000, 12_000, n),
        "grade": rng.choice(["a", "b", "c"], n),
        "region": rng.choice(["north", "south"], n),
    })
    y = ((X.income > 50_000).astype(int)
         ^ (X.grade == "a").astype(int)).to_numpy()

    a1 = AutoDataScientist(random_state=0).fit(X, y)
    p1 = a1.predict(X)

    # same data, opaque names, reversed column order
    order = list(X.columns)[::-1]
    X2 = X[order].rename(columns={c: f"f{i}" for i, c in enumerate(order)})
    a2 = AutoDataScientist(random_state=0).fit(X2, y)
    p2 = a2.predict(X2)

    assert a1.best_name_ == a2.best_name_, (
        f"the chosen family changed with column names/order: "
        f"{a1.best_name_} vs {a2.best_name_}")
    d1 = {d["stage"]: d["choice"] for d in a1.log.to_dict()["decisions"]}
    d2 = {d["stage"]: d["choice"] for d in a2.log.to_dict()["decisions"]}
    for k in ("encoding", "inner_cv", "candidate_families"):
        assert d1.get(k) == d2.get(k), (
            f"decision {k!r} changed with column names/order: "
            f"{d1.get(k)!r} vs {d2.get(k)!r}")
    agree = float((np.asarray(p1) == np.asarray(p2)).mean())
    assert agree == 1.0, (
        f"predictions changed under a pure relabelling of columns "
        f"({agree:.4f} agreement), so something is keying on the schema")
    print(f"  family, decisions and 100% of predictions invariant to "
          f"renaming and reordering columns")


def test_agent_never_imports_the_benchmark_registry():
    """If the agent could read baselines.json it could read the answer."""
    bad = [f.name for f in PKG.glob("*.py")
           if "baselines" in _code_only(f.read_text())]
    assert not bad, f"package modules referencing the registry: {bad}"
    print("  no package module reads runs/baselines.json")


def test_fit_signature_takes_no_task_identity():
    from ads.agent import AutoDataScientist
    import inspect
    params = set(inspect.signature(AutoDataScientist.fit).parameters)
    assert params == {"self", "X", "y"}, params
    init = set(inspect.signature(AutoDataScientist.__init__).parameters)
    assert init == {"self", "random_state"}, init
    print(f"  fit{tuple(sorted(params - {'self'}))} and "
          f"__init__{tuple(sorted(init - {'self'}))} carry no task identity")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
