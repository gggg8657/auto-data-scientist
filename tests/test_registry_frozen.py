"""The committed registry must still be the one the rule produced, and the
committed RESULTS.md must still be what report.py generates from runs/*.json.

A previous project in this workspace had CI that ran neither stage its report
read, so the pipeline went green on a document whose numbers no longer matched
the runs underneath it.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "runs/baselines.json"


def test_selection_is_the_top_k_by_published_runs():
    if not BASE.exists():
        print("  runs/baselines.json absent; skipped (vacuous until stage 1 ran)")
        return
    b = json.loads(BASE.read_text())
    k = b["selection_rule"]["top_k"]
    ranked = sorted(b["tasks"], key=lambda t: -t["n_published_runs"])
    want = [t["task_id"] for t in ranked[:k]]
    assert b["selected_task_ids"] == want, (
        f"selected {b['selected_task_ids']} but the rule gives {want}")
    assert sum(t["selected"] for t in b["tasks"]) == k
    for t in ranked[:k]:
        assert t["selected"], f"task {t['task_id']} is top-{k} but not selected"
    print(f"  selected {want} == top-{k} by published runs, "
          f"out of {len(b['tasks'])} candidates")


def test_baseline_numbers_are_inside_the_published_range():
    if not BASE.exists():
        print("  skipped")
        return
    b = json.loads(BASE.read_text())
    for t in b["tasks"]:
        assert t["min_published"] <= t["median_run"] <= t["max_published"], t["task_id"]
        assert t["q25"] <= t["median_run"] <= t["q75"] <= t["q90"], t["task_id"]
        assert t["n_published_runs"] > 0 and t["n_distinct_flows"] > 0
        assert not t["truncated_at_hard_cap"], (
            f"task {t['task_id']} hit the paging cap, so its median is over a "
            "truncated sample and is not the median of published runs")
    print(f"  {len(b['tasks'])} baselines internally consistent, none truncated")


def test_results_md_matches_what_report_regenerates():
    results = REPO / "RESULTS.md"
    if not results.exists() or not BASE.exists():
        print("  RESULTS.md or baselines absent; skipped")
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "RESULTS.md"
        readme = Path(d) / "README.md"
        readme.write_text((REPO / "README.md").read_text())
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts/report.py"),
             "--baselines", str(BASE), "--bench", str(REPO / "runs/bench"),
             "--out", str(out), "--readme", str(readme)],
            capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        assert out.read_text() == results.read_text(), (
            "RESULTS.md is stale: regenerating it from runs/*.json gives a "
            "different document. Run scripts/report.py and commit the result.")
        assert readme.read_text() == (REPO / "README.md").read_text(), (
            "the README baseline block is stale; run scripts/report.py")
    print("  RESULTS.md and the README block match a fresh regeneration")


def test_metric_check_justifies_the_aggregation_report_uses():
    p = REPO / "runs/metric_check.json"
    if not p.exists():
        print("  runs/metric_check.json absent; skipped")
        return
    m = json.loads(p.read_text())
    assert m["consistent"], f"tasks disagree on the aggregation: {m}"
    assert m["server_aggregation"] in ("pooled", "either"), (
        "report.py compares our accuracy_pooled against the published scalar; "
        f"the measured server aggregation is {m['server_aggregation']!r}, so "
        "that comparison is not justified")
    assert m["n_tasks_discriminating"] >= 1, (
        "no checked task had unequal folds, so nothing was actually verified")
    print(f"  server aggregation measured as {m['server_aggregation']!r} on "
          f"{m['n_tasks_discriminating']}/{m['n_tasks_checked']} discriminating tasks")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
