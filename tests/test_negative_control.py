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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
