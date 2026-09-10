"""The sha256 in the registry proves consistency, not authentic acquisition.

codex, asked how it would fake this KPI, ranked this seventh and the criticism
is exact:

    "Programmatically filter genuine cached evaluation rows toward low
    scores -- or duplicate low-scoring rows -- then run the unchanged baseline
    producer and commit its generated artifacts. [...] The hash proves
    subsequent consistency, not authentic acquisition."

`test_baselines_recompute_from_the_raw_evaluations` recomputes every reading
from the pinned CSVs, which is necessary and not sufficient: it recomputes from
rows that a poisoner would control. Lowering `median_run` toward `q25` by
duplicating weak rows makes the primary threshold easier and leaves every
existing test green.

This does not close that attack -- only independent re-acquisition against a
dated, externally anchored snapshot would -- and the docstring is explicit
rather than implying more. What it does close is the *cheap* version, which is
row duplication, because OpenML's `run_id` is a server-assigned primary key: a
repeated `run_id` cannot arise from honest paging and is the signature of
either duplicated rows or an overlapping page boundary. That second reading
matters independently of any attack, because paging to exhaustion at
`offset += 1000` over a table that is being written to can genuinely double-count
a row, which would bias a median with nobody having done anything wrong.
"""
import os
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "runs/baselines.json"
EVALS = REPO / "runs/evals"


def _read_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        idx = header.index("run_id")
        return [line.rstrip("\n").split(",")[idx] for line in fh if line.strip()]


def test_run_ids_are_unique_within_every_cached_evaluation_file():
    files = sorted(EVALS.glob("task_*.csv.gz"))
    if not files:
        print("  no evaluation caches; skipped")
        return
    offenders = []
    total = 0
    for f in files:
        ids = _read_ids(f)
        total += len(ids)
        if len(ids) != len(set(ids)):
            dup = len(ids) - len(set(ids))
            offenders.append(f"{f.name}: {dup} duplicated run_id of {len(ids)}")
    assert not offenders, (
        "run_id is a server-assigned primary key, so a repeat is either "
        "duplicated rows or an overlapping page boundary; either way the "
        "median is over a population that does not exist:\n  "
        + "\n  ".join(offenders))
    print(f"  {total} evaluations across {len(files)} tasks, every run_id "
          "unique within its file")


def test_selected_tasks_have_their_evidence_on_disk():
    if not BASE.exists():
        print("  registry absent; skipped")
        return
    b = json.loads(BASE.read_text())
    missing = [t["task_id"] for t in b["tasks"] if t["selected"]
               and not (REPO / t["evals_file"]).exists()]
    assert not missing, (
        f"selected tasks with no evaluation file behind their baseline: {missing}")
    print(f"  all {len(b['selected_task_ids'])} selected tasks have their "
          "evaluation file on disk")


def test_registry_row_counts_match_the_files_they_pin():
    """A registry counting more runs than its file holds is the poisoning
    signature that costs nothing to check."""
    if not BASE.exists():
        print("  registry absent; skipped")
        return
    b = json.loads(BASE.read_text())
    bad = []
    for t in b["tasks"]:
        f = REPO / t["evals_file"]
        if not f.exists():
            continue
        n_rows = len(_read_ids(f))
        # n_published_runs counts rows surviving the 0<=value<=1 filter, so it
        # can be lower than the file's row count but never higher.
        if t["n_published_runs"] > n_rows:
            bad.append(f"task {t['task_id']}: registry says "
                       f"{t['n_published_runs']} runs, file holds {n_rows}")
    assert not bad, "\n  ".join(bad)
    print(f"  no registry row count exceeds the file it pins")


def test_no_selected_baseline_was_truncated_at_the_paging_cap():
    """The turn-2 defect, pinned so it cannot come back silently.

    `test_baseline_numbers_are_inside_the_published_range` already refuses any
    truncated task. This states the narrower thing a reader cares about: the
    five targets specifically.
    """
    if not BASE.exists():
        print("  registry absent; skipped")
        return
    b = json.loads(BASE.read_text())
    bad = [t["task_id"] for t in b["tasks"]
           if t["selected"] and t.get("truncated_at_hard_cap")]
    assert not bad, (
        f"selected tasks whose median is over a truncated prefix: {bad}. "
        "Tasks 31 and 10101 hit a 300k cap on 2026-09-10 and their medians "
        "covered ~55% and ~70% of the population, oldest-first.")
    print("  none of the five targets is a median over a truncated prefix")


# ---------------------------------------------------------------------------
# 2026-09-10 turn 6: the half of #7 that could only be answered from outside.
# `scripts/verify_evals.py` re-fetches a seeded random sample of rows from
# OpenML and compares them value for value, and also checks the contamination
# routes that are answerable from the file but were not covered above: a row
# from another task, from another dataset, or carrying another metric (the same
# layout carries AUC and f-measure rows on the server, and one blended into an
# accuracy median moves it). The tests below assert its output rather than
# re-deriving it, and assert that it publishes the weakness of a spot check.
# ---------------------------------------------------------------------------
PROV = REPO / "runs/evals_provenance.json"


def test_provenance_file_exists_once_the_registry_does():
    if not BASE.exists():
        print("  stage 1 has not run; skipped")
        return
    assert PROV.exists(), (
        "runs/baselines.json exists but runs/evals_provenance.json does not: "
        "every human baseline in this repo is read from runs/evals/*.csv.gz "
        "and nothing has checked those rows against OpenML. Run "
        "scripts/verify_evals.py")
    print("  runs/evals_provenance.json present")


def test_every_selected_task_is_covered():
    if not PROV.exists():
        print("  skipped")
        return
    p = json.loads(PROV.read_text())
    b = json.loads(BASE.read_text())
    want = {str(t) for t in b["selected_task_ids"]}
    assert set(p["tasks"]) == want, (
        f"provenance covers {sorted(p['tasks'])} but the registry selected "
        f"{sorted(want)}")
    assert p["metric"] == b["metric"]
    print(f"  all {len(want)} selected tasks covered")


def test_no_internal_contamination_of_the_evaluation_cache():
    if not PROV.exists():
        print("  skipped")
        return
    p = json.loads(PROV.read_text())
    for tid, rec in p["tasks"].items():
        i = rec["internal"]
        assert i["n_duplicate_run_ids"] == 0, (
            f"task {tid}: {i['n_duplicate_run_ids']} duplicated run_ids, each "
            "counting one human submission more than once in the median")
        assert i["n_rows_from_another_task"] == 0, tid
        assert i["n_rows_from_another_dataset"] == 0, tid
        assert i["n_rows_with_another_metric"] == 0, (
            f"task {tid}: {i['n_rows_with_another_metric']} rows carry a "
            "different metric, which would blend it into an accuracy median")
        assert i["matches_registry_run_id_range"], (
            f"task {tid}: the cache's run_id range no longer matches the one "
            "the registry recorded at fetch time, so rows were added or "
            "dropped at an end")
        assert i["matches_registry_upload_window"], tid
    assert p["all_internal_clean"] is True
    print(f"  {len(p['tasks'])} caches clean: no duplicate run_ids, no foreign "
          "task/dataset/metric rows, ranges match the registry")


def test_the_external_probe_found_no_mismatch():
    """The only check in this repo whose evidence comes from outside it."""
    if not PROV.exists():
        print("  skipped")
        return
    p = json.loads(PROV.read_text())
    if p.get("offline"):
        assert not os.environ.get("ADS_REQUIRE_EXTERNAL_PROBE"), (
            "the committed provenance run was --offline, so authenticity is "
            "[not measured]; re-run scripts/verify_evals.py with network")
        print("  DEFERRED: committed provenance run was offline")
        return
    for tid, rec in p["tasks"].items():
        pr = rec["probe"]
        assert pr is not None and "error" not in pr, (tid, pr)
        assert pr["n_mismatched"] == 0, (
            f"task {tid}: {pr['n_mismatched']} re-fetched rows disagree with "
            f"the committed cache: {pr['mismatches']}")
        assert pr["n_returned_by_server"] > 0, (
            f"task {tid}: the server returned nothing, so nothing was verified")
        # a spot check must publish its own weakness
        assert 0 < pr["undetected_probability_if_1pct_tampered"] < 1, pr
        assert pr["k_sampled"] >= 10, (
            f"task {tid}: only {pr['k_sampled']} rows re-fetched")
    assert p["all_probes_verified"] is True
    assert p["n_mismatches_total"] == 0
    print(f"  {p['n_rows_refetched_total']} rows re-fetched from OpenML across "
          f"{len(p['tasks'])} tasks, {p['n_mismatches_total']} mismatched")


def test_the_probe_sample_was_drawn_before_the_fetch():
    """A sample that can be redrawn until it agrees is not a check."""
    if not PROV.exists() or json.loads(PROV.read_text()).get("offline"):
        print("  skipped")
        return
    p = json.loads(PROV.read_text())
    for tid, rec in p["tasks"].items():
        pr = rec["probe"]
        assert "sample_seed" in pr and "sample_rule" in pr, tid
        assert "before the fetch" in pr["sample_rule"], tid
    print("  every probe records its seed and its sampling rule")


def test_the_file_says_what_it_does_not_prove():
    if not PROV.exists():
        print("  skipped")
        return
    p = json.loads(PROV.read_text())
    assert "what_this_does_not_prove" in p and p["what_this_does_not_prove"]
    print("  the output states the limit of a spot check")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
