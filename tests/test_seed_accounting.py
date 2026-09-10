"""`ours` is a mean over run files, so the files must be a bijection with the
registered seeds.

This is the hole `codex` went for when it was asked *how it would fake this
KPI* rather than what was wrong with the code, and it was real. `report.py`
computed the reported accuracy as the mean over every file matching
`runs/bench/task_*.json`, but checked completeness against the **set** of
`random_state` values in those files. A set cannot see a duplicate, so:

    cp runs/bench/task_31_seed2.json runs/bench/task_31_seed2_rerun.json

reweights the mean toward whichever seed scored best, while
`seeds_present` still reads `[0, 1, 2]` and every clause still passes. On a
three-seed screen with accuracies (0.74, 0.75, 0.78) the measured shift is
0.7567 -> 0.7660: nearly a full percentage point, which is larger than several
of the gaps this benchmark has to decide, and it requires editing no number.

The symmetric hole is a seed from *outside* the registered protocol -- run
seeds 0..20, keep the good ones -- which is the same attack with more compute.

These tests pin the bijection. They are written against `verdict()` directly
because that is where the accounting lives.
"""
import runpy
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = runpy.run_path(str(REPO / "scripts/report.py"))
verdict = R["verdict"]

PROTOCOL = {"seeds_screen": [0, 1, 2], "seeds_verdict": list(range(8)),
            "development_tasks": "x", "confirmatory_tasks": "y",
            "escalation_rule": "z", "every_attempt_recorded": "w"}


def _base(n_tasks=1):
    return {"selected_task_ids": list(range(n_tasks)),
            "run_protocol": PROTOCOL,
            "tasks": [{"task_id": t, "selected": True} for t in range(n_tasks)]}


def _run(tid, seed, acc, fname=None):
    return {"task_id": tid, "random_state": seed, "accuracy_pooled": acc,
            "complete": True, "off_registry": False, "n_interventions": 0,
            "registry_sha256": "same", "families_chosen": ["hgb"],
            "_file": fname or f"task_{tid}_seed{seed}.json"}


def _rows(n_tasks=1, ours=0.8):
    return [{"task_id": t, "ours": ours, "primary_pass": True}
            for t in range(n_tasks)]


def test_the_registered_screen_passes():
    bench = {0: [_run(0, s, a) for s, a in zip((0, 1, 2), (0.74, 0.75, 0.78))]}
    v = verdict(_rows(), _base(), bench)
    assert v["clauses"]["3_end_to_end_no_intervention"] is True, v
    assert not v["seeds_duplicated"] and not v["seeds_not_in_registered_protocol"]
    print("  the exact registered screen [0,1,2] passes clause 3")


def test_a_duplicated_seed_sinks_clause_3():
    runs = [_run(0, s, a) for s, a in zip((0, 1, 2), (0.74, 0.75, 0.78))]
    honest = statistics.mean([r["accuracy_pooled"] for r in runs])
    runs.append(_run(0, 2, 0.78, "task_0_seed2_rerun.json"))
    inflated = statistics.mean([r["accuracy_pooled"] for r in runs])
    assert inflated > honest, "the duplicate did not move the mean; bad fixture"
    v = verdict(_rows(), _base(), {0: runs})
    assert v["clauses"]["3_end_to_end_no_intervention"] is False, (
        f"a duplicated seed still passed clause 3; the mean moved "
        f"{honest:.4f} -> {inflated:.4f} unchallenged")
    assert v["seeds_duplicated"] == {"0": {"2": 2}}, v["seeds_duplicated"]
    print(f"  duplicating the best seed moves the mean {honest:.4f} -> "
          f"{inflated:.4f} and is now caught")


def test_a_seed_outside_the_registered_protocol_sinks_clause_3():
    runs = [_run(0, s, 0.80) for s in (0, 1, 2)] + [_run(0, 99, 0.95)]
    v = verdict(_rows(), _base(), {0: runs})
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["seeds_not_in_registered_protocol"] == {"0": [99]}
    print("  a seed outside seeds_screen/seeds_verdict is caught")


def test_escalating_to_the_registered_eight_seeds_is_allowed():
    """Escalation is in the protocol, so it must not be treated as cheating."""
    runs = [_run(0, s, 0.80) for s in range(8)]
    v = verdict(_rows(), _base(), {0: runs})
    assert v["clauses"]["3_end_to_end_no_intervention"] is True, v
    print("  the registered 8-seed verdict set passes")


def test_a_missing_registered_seed_still_sinks_clause_3():
    runs = [_run(0, s, 0.80) for s in (0, 1)]
    v = verdict(_rows(), _base(), {0: runs})
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["seeds_registered_but_missing"] == {"0": [2]}
    print("  a missing registered seed is still caught")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
