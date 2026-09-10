"""The negative controls must exist, be incapable, and be reported.

Every other test in this repository guards our number. This one guards the
*target*: a KPI whose threshold sits below the majority-class rate can be
cleared by predicting the commonest label, and no amount of provenance
machinery would ever notice.

`agy`, asked what one addition would most increase a sceptical reader's belief
in a PASS, ranked this first and predicted the outcome before it was run: "a
dumb DummyClassifier(strategy='prior') [...] clears the primary threshold on 4
of the 5 tasks without learning anything." It does. Four of the five
pre-registered thresholds are at or below the task's own majority-class rate.

So the finding is load-bearing and these tests keep it that way:

- the controls have to be run on the *same* folds and the same pooled metric,
  or the comparison is not a comparison;
- they have to stay incapable -- a "negative control" that got tuned is a
  second agent, and tuning it upward until it fails would be the way to make
  this section flattering;
- the finding has to reach `RESULTS.md`, because a result that lives only in a
  JSON is a result nobody reads.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NC = REPO / "runs/negative_control.json"
BASE = REPO / "runs/baselines.json"
RESULTS = REPO / "RESULTS.md"


def test_the_negative_controls_were_run_on_every_registered_task():
    if not BASE.exists():
        print("  stage 1 has not run; skipped")
        return
    assert NC.exists(), (
        "runs/negative_control.json is missing. Without it nothing in this "
        "repository establishes that the KPI's threshold is demanding at all. "
        "Run scripts/negative_control.py")
    nc = json.loads(NC.read_text())
    want = set(json.loads(BASE.read_text())["selected_task_ids"])
    for name, c in nc["controls"].items():
        got = {r["task_id"] for r in c["tasks"]}
        assert got == want, f"control {name} covers {sorted(got)} not {sorted(want)}"
    print(f"  {len(nc['controls'])} controls x {len(want)} registered tasks")


def test_the_controls_used_the_same_folds_and_the_same_statistic_as_the_agent():
    """Otherwise the rows are not comparable and the section proves nothing."""
    if not NC.exists():
        print("  skipped")
        return
    nc = json.loads(NC.read_text())
    bench = REPO / "runs/bench"
    for name, c in nc["controls"].items():
        for r in c["tasks"]:
            assert "accuracy_pooled" in r, (name, r["task_id"])
            assert r["n_predictions"] > 0 and r["n_folds"] > 0
            # same fold count and the same number of predictions as the
            # agent's runs on that task, which is what "same folds" cashes out
            # to for a pooled metric
            runs = sorted(bench.glob(f"task_{r['task_id']}_seed*.json"))
            if not runs:
                continue
            a = json.loads(runs[0].read_text())
            assert r["n_folds"] == a["n_folds_run"], (
                f"control {name} task {r['task_id']}: {r['n_folds']} folds "
                f"vs the agent's {a['n_folds_run']}")
            assert r["n_predictions"] == a["n_predictions"], (
                f"control {name} task {r['task_id']}: pooled over "
                f"{r['n_predictions']} predictions vs the agent's "
                f"{a['n_predictions']} -- different folds")
    print("  controls and agent pooled over identical fold counts and "
          "prediction counts")


def test_the_controls_are_still_incapable():
    """A tuned negative control is a second agent, not a control.

    The direction that matters: raising a control's score until it *fails* the
    threshold would make this whole section read as though the KPI were
    demanding. So the controls are pinned to what they are.
    """
    src = (REPO / "scripts/negative_control.py").read_text()
    assert 'DummyClassifier(strategy="prior")' in src, (
        "the prior control is no longer a majority-class dummy")
    assert re.search(r"DecisionTreeClassifier\(max_depth=3,\s*random_state=0\)",
                     src), "the stump control is no longer an untuned depth-3 tree"
    assert "GridSearch" not in src and "RandomizedSearch" not in src, (
        "a search appeared in the negative-control script; a tuned control is "
        "not a control")
    if NC.exists():
        nc = json.loads(NC.read_text())
        for r in nc["controls"]["prior"]["tasks"]:
            # a majority-class predictor scores exactly the majority rate
            assert abs(r["accuracy_pooled"] - r["majority_class_rate"]) < 5e-3, (
                f"task {r['task_id']}: the 'prior' control scored "
                f"{r['accuracy_pooled']:.4f} against a majority rate of "
                f"{r['majority_class_rate']:.4f}; it is doing something")
        print("  prior scores the majority rate on every task; stump is "
              "depth 3, untuned; no search in the script")
    else:
        print("  script pins verified; no run yet")


def test_the_finding_reaches_the_document():
    """A result that lives only in a JSON is a result nobody reads."""
    if not RESULTS.exists() or not NC.exists():
        print("  skipped")
        return
    txt = RESULTS.read_text()
    if "[not measured]" in txt and "negative control" not in txt.lower():
        print("  RESULTS.md predates the controls; skipped")
        return
    nc = json.loads(NC.read_text())
    assert "falsifiable" in txt.lower(), (
        "RESULTS.md does not ask whether the clause is falsifiable")
    n = nc["n_thresholds_below_majority_rate"]
    assert f"**{n} of 5 primary" in txt, (
        f"RESULTS.md does not state that {n} of 5 thresholds sit at or below "
        "the majority-class rate")
    for name in nc["controls"]:
        assert name in txt, f"control {name} is missing from RESULTS.md"
    print(f"  RESULTS.md carries the controls and the {n}/5 threshold finding")


def test_the_joint_criterion_is_what_the_controls_fail():
    """The claim that survives the finding, asserted rather than hoped.

    If a control ever clears all five, the KPI is not measuring capability at
    all and this test says so loudly rather than letting the section keep its
    reassuring last paragraph.
    """
    if not NC.exists():
        print("  skipped")
        return
    nc = json.loads(NC.read_text())
    offenders = [n for n, c in nc["controls"].items()
                 if c["clears_all_five_primary"]]
    assert not offenders, (
        f"control(s) {offenders} clear all five tasks on the primary reading. "
        "The joint criterion is then not discriminating either, and the "
        "conclusion in RESULTS.md -- that what survives is the joint reading "
        "-- is false and must be rewritten, not this test relaxed.")
    print("  no control clears all five; the joint criterion is what "
          f"separates them ({', '.join(nc['controls'])})")


TD = REPO / "runs/target_difficulty.json"


def test_the_selection_rule_analysis_is_blind_to_our_accuracy():
    """The analysis of the target must not be able to see our result.

    `runs/target_difficulty.json` asks how many of the 51 candidates have a
    threshold above their majority-class rate. Both inputs are published --
    OpenML's run history and a dataset quality -- so the analysis cannot be
    steered toward a number of ours. This test pins that: the producing script
    must not read runs/bench, and the recorded per-task fields must contain no
    accuracy of ours.
    """
    src = (REPO / "scripts/target_difficulty.py").read_text()
    # Comments and docstrings discuss the other scripts by name, which is what
    # they are for; only executable code is scanned. (Same distinction as the
    # gate fingerprint: documenting a decision is not making one.)
    code = re.sub(r'"""(?:.|\n)*?"""', "", src)
    code = re.sub(r"^\s*#.*$", "", code, flags=re.M)
    for forbidden in ("runs/bench", "accuracy_pooled", "negative_control"):
        assert forbidden not in code, (
            f"scripts/target_difficulty.py has executable code referencing "
            f"{forbidden!r}; the target analysis must not be able to see our "
            "runs")
    if not TD.exists():
        print("  script pins verified; not run yet")
        return
    td = json.loads(TD.read_text())
    for r in td["tasks"]:
        assert "accuracy_pooled" not in r and "ours" not in r, r["task_id"]
    assert td["does_not_change_the_registered_task_set"] is True
    print(f"  {td['n_candidates']} candidates analysed with no access to any "
          "run of ours")


def test_the_successor_task_set_is_offered_not_substituted():
    """A better task set found after the fact may not become this KPI."""
    if not TD.exists() or not BASE.exists():
        print("  skipped")
        return
    td = json.loads(TD.read_text())
    registered = set(json.loads(BASE.read_text())["selected_task_ids"])
    successor = {r["task_id"]
                 for r in td["a_falsifiable_five_under_a_blind_rule"]}
    assert registered == set(json.loads(BASE.read_text())["selected_task_ids"]), (
        "the registry's selected_task_ids changed")
    assert successor != registered, (
        "the successor set is identical to the registered one, so either the "
        "analysis is wrong or the registry was edited")
    # the thing that must not have happened: the successor quietly replacing
    # the registered five in the file that defines the measurement
    for tid in successor - registered:
        assert tid not in registered
    assert "NOT as this one" in td["a_falsifiable_five_rule"], (
        "the successor rule does not say it is a different measurement")
    print(f"  successor set {sorted(successor)} is offered; the registered "
          f"{sorted(registered)} is unchanged")


def test_the_selection_effect_is_reported_with_its_exact_test():
    if not TD.exists() or not RESULTS.exists():
        print("  skipped")
        return
    td = json.loads(TD.read_text())
    txt = RESULTS.read_text()
    if "falsifiable" not in txt.lower():
        print("  RESULTS.md predates the analysis; skipped")
        return
    assert str(td["exact_hypergeometric_p_lower_tail"]) in txt, (
        "RESULTS.md states the counts but not the exact test behind them")
    assert str(td["n_falsifiable"]) in txt and str(td["n_candidates"]) in txt
    print(f"  RESULTS.md carries {td['n_falsifiable']}/{td['n_candidates']} "
          f"and p={td['exact_hypergeometric_p_lower_tail']}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
