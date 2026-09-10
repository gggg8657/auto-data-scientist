"""A task nearer the 5% line than our own seed spread may not be called.

`run_protocol.escalation_rule` was registered in `runs/baselines.json` from the
start -- "a task whose |gap to the 5% line| is within the measured seed range
gets the 8-seed set and an exact test before it is called either way" -- and
nothing enforced it. `codex`, asked how it would fake this KPI rather than what
was wrong with the code, ranked that second of eight: run the three screen
seeds, and if their mean lands just above the line, generate the report. It
also pointed out that "an exact test" named no test.

Both are closed now. The test is an exact one-sided sign test of
H0: median over seeds <= 0.95 x baseline, with k = #{seeds above} out of the
non-tied seeds and p = P(Binomial(n, 1/2) >= k). At the registered 8 seeds:
8/8 gives p = 0.0039, 7/8 gives 0.0352, 6/8 gives 0.1445 -- so a task that
clears the line on a bare majority does not get called.
"""
import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = runpy.run_path(str(REPO / "scripts/report.py"))
sign_test = R["exact_sign_test_above"]
escalation_state = R["escalation_state"]
verdict = R["verdict"]

PROTOCOL = {"seeds_screen": [0, 1, 2], "seeds_verdict": list(range(8))}


def test_sign_test_p_values_are_the_exact_binomial_tail():
    assert abs(sign_test([1.0] * 8, 0.5)["p"] - 1 / 256) < 1e-15
    assert abs(sign_test([1.0] * 7 + [0.0], 0.5)["p"] - 9 / 256) < 1e-15
    assert abs(sign_test([1.0] * 6 + [0.0] * 2, 0.5)["p"] - 37 / 256) < 1e-15
    assert sign_test([1.0] * 8, 0.5)["reject_h0"] is True
    assert sign_test([1.0] * 7 + [0.0], 0.5)["reject_h0"] is True
    assert sign_test([1.0] * 6 + [0.0] * 2, 0.5)["reject_h0"] is False, (
        "6 of 8 above the line is p = 0.1445 and must not be called")
    print("  exact tail: 8/8 p=0.0039, 7/8 p=0.0352, 6/8 p=0.1445 (not called)")


def test_ties_are_excluded_rather_than_counted_as_successes():
    t = sign_test([0.5, 0.5, 1.0, 1.0], 0.5)
    assert t["n"] == 2 and t["k"] == 2, t
    print("  seeds exactly on the threshold drop out of n, not into k")


def test_a_comfortably_clear_task_is_called_on_the_screen():
    """Escalation is for borderline tasks; it must not block an easy pass."""
    accs = [0.90, 0.901, 0.902]                 # baseline 0.80 -> T = 0.76
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["near_line"] is False and e["escalation_required"] is False
    assert e["called"] is True, e
    print(f"  margin {e['margin_to_threshold']:.4f} exceeds spread "
          f"{e['seed_range']:.4f}; called on 3 seeds")


def test_a_borderline_task_cannot_be_called_on_three_seeds():
    # T = 0.95 * 0.80 = 0.76; mean 0.7620 sits 0.0020 above it, inside a
    # seed range of 0.0100 -- the decision is inside our own noise.
    accs = [0.757, 0.762, 0.767]
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["near_line"] is True, e
    assert e["escalation_required"] is True, e
    assert e["called"] is False, "a borderline 3-seed screen was called"
    print(f"  margin {e['margin_to_threshold']:.4f} < spread "
          f"{e['seed_range']:.4f}: escalation to 8 seeds required")


def test_escalation_pending_blocks_clause_2_even_when_every_task_passes():
    """The attack: three favourable screen seeds reported as a pass."""
    accs = [0.757, 0.762, 0.767]
    rows = [{"task_id": t, "ours": 0.762, "primary_pass": True,
             "strict_pass": True,
             "escalation": escalation_state(accs, 0.80, PROTOCOL)}
            for t in range(5)]
    base = {"selected_task_ids": list(range(5)), "run_protocol": PROTOCOL}
    bench = {t: [{"task_id": t, "random_state": s, "accuracy_pooled": a,
                  "complete": True, "off_registry": False,
                  "n_interventions": 0, "registry_sha256": "same",
                  "_file": f"task_{t}_seed{s}.json"}
                 for s, a in zip((0, 1, 2), accs)] for t in range(5)}
    v = verdict(rows, base, bench)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is False, (
        "every task 'passed' the primary reading on 3 borderline seeds and "
        "clause 2 was granted anyway")
    assert sorted(v["escalation_pending_tasks"]) == list(range(5))
    assert v["status"] == "NOT MET as measured"
    print("  5 borderline screens with primary_pass=True do NOT make clause 2")


def test_eight_seeds_all_above_the_line_do_get_called():
    accs = [0.762] * 8                      # all above T = 0.76, none tied
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["escalation_required"] is False
    if e["near_line"]:
        assert e["exact_test"]["reject_h0"] is True
        assert e["called"] is True
    print(f"  8 seeds, near_line={e['near_line']}, called={e['called']}")


def test_strict_verdict_is_computed_beside_the_primary_one():
    rows = [{"task_id": t, "ours": 0.90, "primary_pass": True,
             "strict_pass": False,
             "escalation": escalation_state([0.90, 0.901, 0.902], 0.80, PROTOCOL)}
            for t in range(5)]
    base = {"selected_task_ids": list(range(5)), "run_protocol": PROTOCOL}
    bench = {t: [{"task_id": t, "random_state": s, "accuracy_pooled": 0.9,
                  "complete": True, "off_registry": False,
                  "n_interventions": 0, "registry_sha256": "same",
                  "_file": f"task_{t}_seed{s}.json"} for s in (0, 1, 2)]
             for t in range(5)}
    v = verdict(rows, base, bench)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is True
    assert v["clause2_under_strictest_baseline"] is False, (
        "the strictest-baseline verdict must be reported, not just recorded")
    print("  primary clause 2 True while the strictest reading is False, "
          "both surfaced")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
