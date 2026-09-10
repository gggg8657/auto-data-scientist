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
        assert not BASE.exists(), (
            "runs/baselines.json exists but runs/metric_check.json does not: "
            "report.py compares our accuracy_pooled against the published "
            "scalar and nothing has verified that they are the same statistic")
        print("  stage 1 has not run either; skipped")
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


def test_baselines_recompute_from_the_raw_evaluations():
    """The registry must be reproducible from the evidence beside it.

    codex, reviewing before any result existed, gave a concrete attack: lower
    `median_run` toward `q25`, keep it inside the published range, regenerate
    the documents, and nothing fails -- the target moves and the runs measured
    against it still look valid. The old tests only checked inequalities.

    So this recomputes every baseline reading from `runs/evals/*.csv.gz`, whose
    sha256 is pinned in the registry, and compares. The recomputation here is
    deliberately an independent reimplementation rather than an import of
    `fetch_baselines.summarise`, so that editing the producer does not also
    edit its own check.
    """
    if not BASE.exists():
        print("  skipped")
        return
    import hashlib

    import pandas as pd

    b = json.loads(BASE.read_text())
    checked = 0
    for t in b["tasks"]:
        f = REPO / t["evals_file"]
        assert f.exists(), f"task {t['task_id']}: {t['evals_file']} is missing"
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == t["evals_sha256"], (
            f"task {t['task_id']}: {t['evals_file']} does not match the sha256 "
            "pinned in the registry")

        d = pd.read_csv(f)
        v = pd.to_numeric(d["value"], errors="coerce")
        ok = v.notna() & (v >= 0) & (v <= 1)
        d = d[ok].copy()
        d["value"] = v[ok]
        want = {
            "n_published_runs": len(d),
            "n_distinct_flows": d["flow_id"].nunique(),
            "n_distinct_uploaders": d["uploader"].nunique(),
            "median_run": d["value"].median(),
            "median_flow": d.groupby("flow_id")["value"].median().median(),
            "median_flow_best": d.groupby("flow_id")["value"].max().median(),
            "median_uploader_best": d.groupby("uploader")["value"].max().median(),
            "max_published": d["value"].max(),
            "min_published": d["value"].min(),
        }
        for k, exp in want.items():
            got = t[k]
            assert abs(float(got) - float(exp)) < 1e-12, (
                f"task {t['task_id']}: registry says {k}={got} but the raw "
                f"evaluations give {exp}")
        checked += 1
    print(f"  {checked} registries recomputed from sha256-pinned raw "
          "evaluations; every reading matches")


def test_strictest_baseline_is_recorded_and_is_the_maximum():
    if not BASE.exists():
        print("  skipped")
        return
    b = json.loads(BASE.read_text())
    keys = ("median_run", "median_flow", "median_flow_best",
            "median_uploader_best")
    for t in b["tasks"]:
        vals = {k: t[k] for k in keys}
        assert t["strictest_baseline"] == max(vals, key=vals.get), t["task_id"]
        assert abs(t["strictest_baseline_value"] - max(vals.values())) < 1e-12
    print(f"  strictest of {len(keys)} readings recorded per task, so no row "
          "can hide behind the easiest baseline")


def test_the_run_protocol_was_registered_in_advance():
    if not BASE.exists():
        print("  skipped")
        return
    rp = json.loads(BASE.read_text())["run_protocol"]
    for k in ("development_tasks", "confirmatory_tasks", "seeds_screen",
              "seeds_verdict", "escalation_rule", "every_attempt_recorded"):
        assert rp.get(k), f"run_protocol is missing {k}"
    assert rp["seeds_screen"] == [0, 1, 2]
    assert rp["seeds_verdict"] == list(range(8))
    print(f"  run protocol registered: screen {rp['seeds_screen']}, "
          f"verdict {rp['seeds_verdict']}")


def test_every_run_is_bound_to_the_registry_it_was_measured_against():
    """A run that does not carry the digest of the registry it was scored
    against cannot be checked for a moved target."""
    import hashlib
    bench = REPO / "runs/bench"
    if not BASE.exists() or not bench.exists():
        print("  skipped")
        return
    want = hashlib.sha256(BASE.read_text().encode()).hexdigest()
    files = sorted(bench.glob("task_*.json"))
    if not files:
        print("  no benchmark runs yet; skipped")
        return
    for f in files:
        r = json.loads(f.read_text())
        assert r.get("registry_sha256") == want, (
            f"{f.name} was measured against registry "
            f"{str(r.get('registry_sha256'))[:12]} but runs/baselines.json is "
            f"now {want[:12]} -- the target moved under the run")
    print(f"  {len(files)} runs all bound to registry {want[:12]}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
