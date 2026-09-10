"""The report must derive its verdict, and must say [not measured] rather than
omit a row. A previous project in this workspace shipped a hard-coded 'both
clauses are met' that outlived the numbers, and a report.py that silently
dropped a whole split from the table."""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import report as R  # noqa: E402


def test_tolerance_readings_are_the_three_that_were_pre_registered():
    # exactly on the line: must pass, and must not depend on float noise
    t = R.tolerance_readings(0.95, 1.00)
    assert t["rel_one_sided"] is True and t["rel_two_sided"] is True, (
        "a result exactly at tolerance was failed by floating point")
    t = R.tolerance_readings(0.96, 1.00)
    assert t["rel_one_sided"] is True and t["rel_two_sided"] is True
    t = R.tolerance_readings(0.9499, 1.00)
    assert t["rel_one_sided"] is False, "0.9499 is below 0.95 x baseline"
    # EPS must not widen the tolerance by anything a measurement could reach
    t = R.tolerance_readings(0.95 - 1e-6, 1.00)
    assert t["rel_one_sided"] is False, "EPS widened the tolerance"
    # the primary is one-sided: beating the baseline by 6% must not fail
    t = R.tolerance_readings(1.06, 1.00)
    assert t["rel_one_sided"] is True and t["rel_two_sided"] is False
    # relative and absolute disagree where the baseline is low
    t = R.tolerance_readings(0.46, 0.50)
    assert t["abs_two_sided"] is True and t["rel_one_sided"] is False
    print("  three readings distinct and correctly signed")


def _base(n=5):
    tasks = [{"task_id": 100 + i, "dataset_name": f"d{i}", "selected": True,
              "median_run": 0.8, "median_flow": 0.75, "q90": 0.9,
              "max_published": 0.95, "n_instances": 1, "n_features": 1,
              "n_classes": 2, "n_published_runs": 10, "n_distinct_flows": 3,
              "n_distinct_uploaders": 2} for i in range(n)]
    return {"generated": "t", "metric": "predictive_accuracy",
            "selected_task_ids": [t["task_id"] for t in tasks],
            "selection_rule": {"source": "s", "filter": "f", "top_k": n,
                               "n_with_evals": 40, "rank_by": "r"},
            "tasks": tasks}


def _run(tid, acc, **kw):
    r = {"task_id": tid, "accuracy_pooled": acc, "n_interventions": 0,
         "families_chosen": ["hgb"], "complete": True, "off_registry": False,
         "_file": f"task_{tid}.json"}
    r.update(kw)
    return r


def test_verdict_is_derived_and_a_single_failing_task_sinks_clause_2():
    base = _base()
    bench = {100 + i: [_run(100 + i, 0.80)] for i in range(5)}
    rows = [{"task_id": t, "ours": 0.80, "primary_pass": True}
            for t in base["selected_task_ids"]]
    v = R.verdict(rows, base, bench)
    assert v["status"] == "PASS", v
    rows[3]["primary_pass"] = False
    v = R.verdict(rows, base, bench)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is False
    assert v["status"] == "NOT MET as measured", v["status"]
    print("  one failing task flips clause 2 and the status")


def test_a_single_intervention_sinks_clause_3():
    base = _base()
    bench = {100 + i: [_run(100 + i, 0.80)] for i in range(5)}
    bench[102][0]["n_interventions"] = 1
    rows = [{"task_id": t, "ours": 0.80, "primary_pass": True}
            for t in base["selected_task_ids"]]
    v = R.verdict(rows, base, bench)
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["n_interventions_total"] == 1
    print("  one logged intervention flips clause 3")


def test_partial_and_off_registry_runs_cannot_pass():
    base = _base()
    rows = [{"task_id": t, "ours": 0.80, "primary_pass": True}
            for t in base["selected_task_ids"]]
    for key, val in (("complete", False), ("off_registry", True)):
        bench = {100 + i: [_run(100 + i, 0.80)] for i in range(5)}
        bench[101][0][key] = val
        v = R.verdict(rows, base, bench)
        assert v["clauses"]["3_end_to_end_no_intervention"] is False, key
    print("  a truncated run and an off-registry run both sink clause 3")


def test_four_measured_tasks_is_not_a_pass():
    base = _base()
    bench = {100 + i: [_run(100 + i, 0.80)] for i in range(4)}
    rows = [{"task_id": t, "ours": 0.80 if t < 104 else None,
             "primary_pass": True if t < 104 else None}
            for t in base["selected_task_ids"]]
    v = R.verdict(rows, base, bench)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is False
    assert v["n_tasks_measured"] == 4 and v["n_tasks_registered"] == 5
    print("  4 of 5 measured is not clause 2")


def test_report_writes_not_measured_rather_than_omitting_a_task():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "baselines.json").write_text(json.dumps(_base()))
        (d / "bench").mkdir()
        (d / "README.md").write_text(
            "x\n<!-- BASELINES:BEGIN -->\nold\n<!-- BASELINES:END -->\ny\n")
        sys.argv = ["report.py", "--baselines", str(d / "baselines.json"),
                    "--bench", str(d / "bench"), "--out", str(d / "RESULTS.md"),
                    "--readme", str(d / "README.md")]
        R.main()
        out = (d / "RESULTS.md").read_text()
        for i in range(5):
            assert f"d{i}" in out, f"task d{i} omitted from the report entirely"
        assert out.count(R.NM) >= 5, "unmeasured tasks did not say [not measured]"
        assert "RUNNING" in out
        readme = (d / "README.md").read_text()
        assert "old" not in readme and "d0" in readme
        print("  all 5 rows present with [not measured]; README block refreshed")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
