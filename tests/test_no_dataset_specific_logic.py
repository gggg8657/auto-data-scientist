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
    for f in sorted(PKG.glob("*.py")):
        code = _code_only(f.read_text())
        for tid in tids:
            if re.search(rf"\b{tid}\b", code):
                bad.append(f"{f.name}: task id {tid}")
        for nm in names:
            if nm.lower() in code.lower():
                bad.append(f"{f.name}: dataset name {nm!r}")
    assert not bad, "dataset-specific logic in the agent: " + "; ".join(bad)
    print(f"  none of {len(tids)} task ids or {len(names)} dataset names "
          f"appears in ads/*.py code")


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
