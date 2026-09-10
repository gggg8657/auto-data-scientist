"""A task nearer the 5% line than our own seed spread may not be called.

`run_protocol.escalation_rule` was registered in `runs/baselines.json` from the
start -- "a task whose |gap to the 5% line| is within the measured seed range
gets the 8-seed set and an exact test before it is called either way" -- and
nothing enforced it. `codex`, asked how it would fake this KPI rather than what
was wrong with the code, ranked that second of eight: run the three screen
seeds, and if their mean lands just above the line, generate the report. It
also pointed out that "an exact test" named no test.

Both are closed now. The test is an exact one-sided sign test of
H0: median over seeds <= 0.95 x baseline, with k = #{seeds strictly above} out
of **all** n seeds and p = P(Binomial(n, 1/2) >= k). At the registered 8 seeds:
8/8 gives p = 0.0039, 7/8 gives 0.0352, 6/8 gives 0.1445 -- so a task that
clears the line on a bare majority does not get called.

Two of the assertions in this file are the *reverse* of what they were, and
both reversals are recorded where they live rather than quietly rewritten:

- ties used to be dropped from n, which is the continuous-distribution sign
  test and is anti-conservative on a discrete metric (Type I error 17.37% at
  a nominal 5%, computed below);
- a wide-margin 3-seed screen used to be "called", which let the pre-registered
  exact test run on none of the five tasks.
"""
import math
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


def test_ties_count_as_non_wins_and_stay_in_the_denominator():
    """Reversed on 2026-09-10; the old behaviour inflated Type I error.

    This asserted that ties drop out of n. That is the sign test for a
    *continuous* distribution, and accuracy is discrete, so ties at the
    threshold have real probability and dropping them is anti-conservative.
    codex supplied the counterexample when asked how to make the clause pass;
    `test_dropping_ties_would_inflate_type_i_error_to_17_percent` computes it.
    """
    t = sign_test([0.5, 0.5, 1.0, 1.0], 0.5)
    assert t["n"] == 4 and t["k"] == 2 and t["n_ties"] == 2, t
    assert abs(t["p"] - 11 / 16) < 1e-12, t     # P(Binom(4,1/2) >= 2)
    print("  2 wins, 2 ties of 4 -> k=2, n=4, p=0.6875 (was k=2, n=2, p=0.25)")


def test_dropping_ties_would_inflate_type_i_error_to_17_percent():
    """The counterexample, computed rather than quoted.

    P(A = T) = 0.6, P(A > T) = 0.4, nothing below T. The median IS T, so
    H0: median <= T is true. A gate at alpha = 0.05 must reject at most 5% of
    the time.
    """
    def p_dropping_ties(k, n_nonties):
        if n_nonties == 0:
            return 1.0
        return sum(math.comb(n_nonties, i)
                   for i in range(k, n_nonties + 1)) / 2 ** n_nonties

    p_tie, p_above, n = 0.6, 0.4, 8
    old = new = 0.0
    for w in range(n + 1):                    # w strict wins, n-w ties
        prob = math.comb(n, w) * p_above ** w * p_tie ** (n - w)
        if p_dropping_ties(w, w) <= 0.05:
            old += prob
        # the shipped test: ties in the denominator, counted as non-wins
        if sign_test([1.0] * w + [0.5] * (n - w), 0.5)["reject_h0"]:
            new += prob
    assert old > 0.17, old
    assert new <= 0.05, new
    print(f"  H0 true: dropping ties rejects {old:.2%} of the time at a "
          f"nominal 5%; the shipped rule rejects {new:.2%}")


def test_no_three_seed_screen_can_be_called_however_wide_the_margin():
    """This assertion is the reverse of what it was, and the reversal is the
    finding of 2026-09-10 turn 6.

    It used to read: a task whose margin exceeds its own seed range "is called
    on 3 seeds", on the reasoning that escalation is for borderline tasks and
    must not block an easy pass. That let the 3-seed screen of the five
    registered tasks -- every one of them clear of its threshold by 4.5x to
    68x its seed range -- produce `status: PASS` with the pre-registered exact
    test never run on any of them.

    The gate it used, `margin > observed seed range`, is biased at small n and
    biased the flattering way: E[range] is 1.69 sigma at n=3 against 2.85
    sigma at n=8, so the quantity in the denominator is the one that shrinks
    when you run fewer seeds. A gate that is easier to clear on less evidence
    is not a gate.

    So the exact test is now the gate for every task, and since p >= 1/2^n it
    cannot reach alpha=0.05 at n=3 no matter how large the margin.
    """
    accs = [0.90, 0.901, 0.902]                 # baseline 0.80 -> T = 0.76
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["margin_exceeds_seed_range"] is True, e
    assert e["exact_test"]["k"] == 3 and e["exact_test"]["n"] == 3
    assert e["exact_test"]["p"] == 0.125
    assert e["called"] is False, "a 3-seed screen was called"
    assert e["escalation_required"] is True, e
    print(f"  margin {e['margin_to_threshold']:.4f} is {e['margin_to_threshold'] / e['seed_range']:.0f}x "
          f"the spread {e['seed_range']:.4f} and still not called: p=0.125 floor at n=3")


def test_the_p_value_floor_is_what_makes_three_seeds_uncallable():
    """Not an implementation detail -- it is why the rule is enforceable."""
    for n in (1, 2, 3, 4):
        t = sign_test([1.0] * n, 0.5)
        assert t["reject_h0"] is False, (n, t)
    assert sign_test([1.0] * 5, 0.5)["reject_h0"] is True, (
        "5 of 5 is p=0.03125 and is callable; the floor argument stops at n=5")
    print("  n<=4 cannot reject at alpha=0.05; n=5 all-above gives p=0.03125")


def test_a_favourable_prefix_of_the_verdict_set_cannot_be_called_early():
    """Optional stopping, found by writing the test above.

    The p-value floor alone stops only n <= 4. At n=5 an all-above run gives
    p=0.03125 and would reject, so a runner that watched the seeds land and
    stopped at the first rejection would be reporting a sequentially monitored
    p-value as if it were fixed-sample. The registered verdict set is 8 seeds
    and all 8 are required before a task is called either way.
    """
    T_base = 0.80                               # T = 0.76
    for n in (5, 6, 7):
        e = escalation_state([0.90] * n, T_base, PROTOCOL)
        assert e["exact_test"]["reject_h0"] is True, (n, e)
        assert e["seeds_meet_registered_verdict_count"] is False, (n, e)
        assert e["called"] is False, (
            f"{n} of the registered 8 seeds rejected H0 and the task was "
            "called on the prefix")
    e8 = escalation_state([0.90] * 8, T_base, PROTOCOL)
    assert e8["called"] is True and e8["escalation_required"] is False, e8
    print("  n=5,6,7 reject H0 but are not called; n=8 is")


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
    c2 = v["clauses"]["2_within_5pct_of_human_baseline"]
    assert c2 is not True, (
        "every task 'passed' the primary reading on 3 borderline seeds and "
        "clause 2 was granted anyway")
    # None, not False: a screen that has not reached the registered seed count
    # has not measured the clause, and "NOT MET as measured" off a screen is
    # the same error as PASS off a screen with the sign flipped.
    assert c2 is None, c2
    assert sorted(v["escalation_pending_tasks"]) == list(range(5))
    assert v["status"] == "RUNNING (not all clauses measured)", v["status"]
    print("  5 borderline screens with primary_pass=True leave clause 2 "
          "unmeasured, not failed")


def test_eight_seeds_all_above_the_line_do_get_called():
    accs = [0.762] * 8                      # all above T = 0.76, none tied
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["escalation_required"] is False
    assert e["exact_test"]["reject_h0"] is True and e["exact_test"]["n"] == 8
    assert e["called"] is True, e
    print(f"  8 seeds, p={e['exact_test']['p']:.4f}, called={e['called']}")


def test_a_task_that_fails_at_the_full_seed_count_is_a_result_not_a_pending_run():
    """The negative direction of the same gate."""
    accs = [0.70] * 8                       # all BELOW T = 0.76
    e = escalation_state(accs, 0.80, PROTOCOL)
    assert e["exact_test"]["k"] == 0 and e["exact_test"]["reject_h0"] is False
    assert e["called"] is False and e["escalation_required"] is False, e
    rows = [{"task_id": t, "ours": 0.70, "primary_pass": False,
             "strict_pass": False, "escalation": e,
             "escalation_strict": e} for t in range(5)]
    base = {"selected_task_ids": list(range(5)), "run_protocol": PROTOCOL}
    bench = {t: [{"task_id": t, "random_state": s, "accuracy_pooled": 0.70,
                  "complete": True, "off_registry": False,
                  "n_interventions": 0, "registry_sha256": "same",
                  "_file": f"task_{t}_seed{s}.json"} for s in range(8)]
             for t in range(5)}
    v = verdict(rows, base, bench)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is False
    assert v["status"] == "NOT MET as measured", v["status"]
    print("  0/8 above the line at the full seed count => NOT MET, not RUNNING")


def test_strict_verdict_is_computed_beside_the_primary_one():
    accs = [0.90] * 8
    rows = [{"task_id": t, "ours": 0.90, "primary_pass": True,
             "strict_pass": False,
             "escalation": escalation_state(accs, 0.80, PROTOCOL),
             # strict baseline 1.00 -> T = 0.95, every seed below it
             "escalation_strict": escalation_state(accs, 1.00, PROTOCOL)}
            for t in range(5)]
    base = {"selected_task_ids": list(range(5)), "run_protocol": PROTOCOL}
    bench = {t: [{"task_id": t, "random_state": s, "accuracy_pooled": 0.9,
                  "complete": True, "off_registry": False,
                  "n_interventions": 0, "registry_sha256": "same",
                  "_file": f"task_{t}_seed{s}.json"} for s in range(8)]
             for t in range(5)}
    # Amendment 7 added a leakage gate to clause 2. The subject of this test is
    # the strictest-baseline reading, so a clean probe is injected rather than
    # letting an absent one route the clause to None and hide what is asserted.
    leak = {"verdict": "NO_LEAKAGE_DETECTED", "tasks_cleared": [0, 1],
            "env": {"ads_sha256": "same"}}
    power = {"min_folds_for_detection": {"0": 1, "1": 1},
             "tasks_unpowered_pooled": [2, 3, 4]}
    v = verdict(rows, base, bench, leakage=leak, power=power)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is True
    assert v["clause2_under_strictest_baseline"] is False, (
        "the strictest-baseline verdict must be reported, not just recorded")
    print("  primary clause 2 True while the strictest reading is False, "
          "both surfaced, both at 8 seeds")


def test_the_leakage_gate_has_three_states_and_cannot_launder_a_fail():
    """Amendment 7. Absent/stale/underpowered -> None, not True and not a
    laundered False."""
    accs = [0.90] * 8
    def mkrows(primary_pass=True):
        return [{"task_id": t, "ours": 0.90, "primary_pass": primary_pass,
                 "strict_pass": True,
                 "escalation": escalation_state(accs, 0.80, PROTOCOL),
                 "escalation_strict": escalation_state(accs, 0.80, PROTOCOL)}
                for t in range(5)]
    base = {"selected_task_ids": list(range(5)), "run_protocol": PROTOCOL}
    bench = {t: [{"task_id": t, "random_state": s, "accuracy_pooled": 0.9,
                  "complete": True, "off_registry": False,
                  "n_interventions": 0, "registry_sha256": "same",
                  "env": {"ads_sha256": "same"},
                  "_file": f"task_{t}_seed{s}.json"} for s in range(8)]
             for t in range(5)}
    power = {"min_folds_for_detection": {"0": 1, "1": 1},
             "tasks_unpowered_pooled": [2, 3, 4]}
    C = "2_within_5pct_of_human_baseline"

    def c2(leakage, power=power, rows=None):
        return verdict(rows or mkrows(), base, bench,
                       leakage=leakage, power=power)["clauses"][C]

    clean = {"verdict": "NO_LEAKAGE_DETECTED", "tasks_cleared": [0, 1],
             "env": {"ads_sha256": "same"}}
    assert c2(clean) is True, "a fresh, powered, clean probe must clear it"
    assert c2(None) is None, "an absent probe is a hole, not a pass"
    assert c2({**clean, "env": {}}) is None, (
        "a probe with no agent digest cannot testify about these runs")
    assert c2({**clean, "env": {"ads_sha256": "other"}}) is None, (
        "a probe produced by a different ads/ than the runs is stale")
    assert c2(clean, power={}) is None, (
        "without the power file there is no way to know what the probe could "
        "have detected")
    assert c2({**clean, "tasks_cleared": [0]}) is None, (
        "task 1 is probeable and was not cleared; a partial probe is not a "
        "pass")
    assert c2({**clean, "verdict": "LEAKAGE"}) is False, (
        "a leak must sink the clause outright")
    # And the direction that matters: an absent probe must not turn a genuine
    # FAIL into RUNNING, which would be the flattering reading.
    assert c2(None, rows=mkrows(primary_pass=False)) is False, (
        "an absent probe laundered a failing accuracy into RUNNING")
    print("  leakage gate: clean=True, absent/stale/undigested/unpowered/"
          "partial=None, LEAKAGE=False, and it cannot launder a FAIL")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
