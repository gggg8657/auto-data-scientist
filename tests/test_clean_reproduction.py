"""The reproduction digest must not be blind to the things it exists to catch.

`scripts/clean_reproduction.py` decides whether the untouched `clean` re-run
reproduced a cell of the confirmatory set. It does that by digesting the whole
record with volatile fields removed -- and every field removed is a field the
comparison can no longer see. The first version of this script stripped too
little (all three cells came back non-identical purely because each logged
decision carries a wall-clock `t`); the failure mode in the other direction is
worse and silent, because stripping too much makes every cell reproduce.

So these tests pin both edges: the timestamp must be ignored, and a change to
anything the agent actually *decided* -- the accuracy, the model family, the
tournament, the preprocessing profile -- must break the digest. The "accuracy
agrees but the record does not" bucket exists so that a cell landing there is
reported rather than counted, and that too is pinned.
"""
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(REPO / "scripts/clean_reproduction.py"))
digest = M["digest"]
canonical = M["canonical"]


def _rec(acc=0.75, family="logreg", t=1000.0, profile="std", complete=True):
    return {
        "task_id": 31, "seed": 0, "complete": complete,
        "accuracy_pooled": acc, "accuracy_folds": [acc, acc],
        "families_chosen": [family, family],
        "n_correct": 75, "n_predictions": 100,
        "registry_sha256": "abc", "seconds_total": 12.5,
        "role": "clean", "off_registry": False,
        "env": {"ads_sha256": "same", "ads_dirty_vs_head": False},
        "per_fold": [
            {"fold": 0, "accuracy": acc, "family": family, "seconds": 1.0,
             "profile": profile,
             "tournament": [{"family": family, "score": acc}],
             "decisions": {"decisions": [{"what": "pick", "t": t}]}},
        ],
    }


def test_timestamps_do_not_break_the_digest():
    """Two runs of one cell can never agree on wall-clock; that is not a diff."""
    a, _ = digest(_rec(t=1000.0))
    b, _ = digest(_rec(t=9999.0))
    assert a == b


def test_timestamp_stripping_is_counted_and_narrow():
    """If the strip rule silently broadens, the count changes and this goes red."""
    _, n = digest(_rec())
    assert n == 1, f"expected exactly one `t` stripped, got {n}"


def test_volatile_fields_are_the_only_ones_dropped():
    """Everything the agent decided survives into the digested record."""
    c = canonical(_rec())
    assert "seconds_total" not in c and "env" not in c and "role" not in c
    assert "seconds" not in c["per_fold"][0]
    for k in ("accuracy", "family", "profile", "tournament", "decisions"):
        assert k in c["per_fold"][0], f"{k} was stripped and must not be"
    for k in ("accuracy_pooled", "accuracy_folds", "families_chosen",
              "n_correct", "n_predictions", "registry_sha256"):
        assert k in c, f"{k} was stripped and must not be"


def test_a_changed_decision_breaks_the_digest():
    """The whole point: a different choice is not a reproduction."""
    base, _ = digest(_rec())
    for label, other in [
        ("accuracy", _rec(acc=0.76)),
        ("family", _rec(family="hgb")),
        ("profile", _rec(profile="quantile")),
    ]:
        d, _ = digest(other)
        assert d != base, f"digest is blind to a changed {label}"


def test_accuracy_equal_but_record_differs_is_not_a_reproduction(tmp_path):
    """A cell that agrees on the score but not on how it got there is reported,
    not counted -- that bucket is the whole reason the strong test exists."""
    bench, clean = tmp_path / "bench", tmp_path / "clean"
    bench.mkdir(); clean.mkdir()
    # Same pooled accuracy, different model family.
    (bench / "task_31_seed0.json").write_text(json.dumps(_rec(family="logreg")))
    (clean / "task_31_seed0.json").write_text(json.dumps(_rec(family="hgb")))
    M["main"](bench_dir=bench, clean_dir=clean, out_path=tmp_path / "out.json")
    got = json.loads((tmp_path / "out.json").read_text())
    assert got["n_identical"] == 0
    assert got["n_accuracy_equal_digest_differs"] == 1
    # Coverage IS complete here -- every confirmatory cell was compared -- and
    # that is exactly why `clean_set_complete` must not be the clause-3 gate.
    assert got["clean_set_complete"] is True
    assert got["all_cells_reproduced"] is False


def test_partial_records_are_not_compared(tmp_path):
    """An incomplete record reproduces nothing and must not inflate coverage."""
    bench, clean = tmp_path / "bench", tmp_path / "clean"
    bench.mkdir(); clean.mkdir()
    (bench / "task_31_seed0.json").write_text(json.dumps(_rec()))
    (clean / "task_31_seed0.json").write_text(json.dumps(_rec(complete=False)))
    M["main"](bench_dir=bench, clean_dir=clean, out_path=tmp_path / "out.json")
    got = json.loads((tmp_path / "out.json").read_text())
    assert got["n_cells_compared"] == 0
    assert got["clean_set_complete"] is False
    assert got["all_cells_reproduced"] is False


def test_a_complete_set_with_a_mismatch_does_not_read_as_reproduced(tmp_path):
    """The fail-open shape: coverage complete, one cell wrong, gate must be False."""
    bench, clean = tmp_path / "bench", tmp_path / "clean"
    bench.mkdir(); clean.mkdir()
    for seed, fam in ((0, "logreg"), (1, "logreg")):
        r = _rec(family=fam); r["seed"] = seed
        (bench / f"task_31_seed{seed}.json").write_text(json.dumps(r))
    for seed, fam in ((0, "logreg"), (1, "hgb")):
        r = _rec(family=fam); r["seed"] = seed
        (clean / f"task_31_seed{seed}.json").write_text(json.dumps(r))
    M["main"](bench_dir=bench, clean_dir=clean, out_path=tmp_path / "out.json")
    got = json.loads((tmp_path / "out.json").read_text())
    assert got["clean_set_complete"] is True
    assert got["n_identical"] == 1
    assert got["all_cells_reproduced"] is False


def test_partial_clean_set_is_never_complete(tmp_path):
    """Coverage below the confirmatory set cannot read as a settled clause."""
    bench, clean = tmp_path / "bench", tmp_path / "clean"
    bench.mkdir(); clean.mkdir()
    for seed in (0, 1):
        r = _rec(); r["seed"] = seed
        (bench / f"task_31_seed{seed}.json").write_text(json.dumps(r))
    r = _rec()
    (clean / "task_31_seed0.json").write_text(json.dumps(r))
    M["main"](bench_dir=bench, clean_dir=clean, out_path=tmp_path / "out.json")
    got = json.loads((tmp_path / "out.json").read_text())
    assert got["n_identical"] == 1
    assert got["clean_coverage_of_bench"] == 0.5
    assert got["clean_set_complete"] is False
    assert got["all_cells_reproduced"] is False


if __name__ == "__main__":
    # CI runs each test file with plain `python`, not pytest, so the `tmp_path`
    # fixture does not exist here -- supply a temporary directory to the tests
    # that ask for one and call the rest bare.
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        if "tmp_path" in inspect.signature(f).parameters:
            with tempfile.TemporaryDirectory() as d:
                f(Path(d))
        else:
            f()
    print(f"\n{len(fns)} tests passed")
