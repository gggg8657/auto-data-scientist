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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
