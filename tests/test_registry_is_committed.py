"""The registry has to be *frozen*, and "frozen" means committed.

Turn 1's commit message (0c9dd8a) said the baselines were "fetched, committed
and frozen ... before any agent run exists". Two of those three were true.
`runs/baselines.json` was written at 09:10, thirty-three minutes *after* the
08:37 commit that described committing it, and it was not tracked by git at all
-- so there was nothing stopping the target from being regenerated later,
against a result, with no diff for a reviewer to notice.

The property that actually matters is not "a registry exists" but "the registry
the runs were scored against is in the history". So:

- before any benchmark run exists, the registry may be regenerated freely --
  that is the whole point of fixing the target early, and a fix applied with no
  result in hand cannot be result-driven;
- once even one benchmark run exists, the registry must be tracked *and*
  unmodified relative to HEAD, because from that moment on any edit to it moves
  the target under a measurement.

`test_every_run_is_bound_to_the_registry_it_was_measured_against` catches the
case where the registry changed after a run and the digests stop matching.
This catches the narrower case that one gets past: an uncommitted registry,
where there is no committed version to compare against in the first place.
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "runs/baselines.json"
BENCH = REPO / "runs/bench"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _benchmark_runs() -> list[Path]:
    if not BENCH.exists():
        return []
    return sorted(p for p in BENCH.glob("task_*.json")
                  if not p.name.endswith(".FAILED.json"))


def test_registry_is_tracked_by_git():
    if not BASE.exists():
        print("  runs/baselines.json absent; skipped (vacuous until stage 1 ran)")
        return
    r = _git("ls-files", "--error-unmatch", "runs/baselines.json")
    assert r.returncode == 0, (
        "runs/baselines.json is not tracked by git, so the pre-registered "
        "target is not in the history and an edit to it would leave no diff. "
        "git add runs/baselines.json")
    print("  runs/baselines.json is tracked")


def test_registry_is_unmodified_once_a_run_has_been_scored_against_it():
    runs = _benchmark_runs()
    if not BASE.exists() or not runs:
        print(f"  {len(runs)} benchmark runs; registry may still be "
              "regenerated freely, skipped")
        return
    r = _git("diff", "--quiet", "HEAD", "--", "runs/baselines.json")
    assert r.returncode == 0, (
        f"{len(runs)} benchmark runs exist and runs/baselines.json differs "
        "from HEAD: the target moved under a measurement. Either commit the "
        "registry before running the benchmark, or discard these runs and "
        "re-run them against the committed registry.")
    print(f"  registry matches HEAD, with {len(runs)} runs scored against it")


def test_superseded_baselines_are_kept_as_evidence_not_overwritten():
    """A target that moved must leave its old value behind.

    The 300k-paging-cap defect (turn 2) moved two of the five primary targets.
    That is legitimate only because it happened before any accuracy existed and
    because the old readings are still on disk to be compared against.
    """
    p = REPO / "runs/baselines_truncated_superseded.json"
    if not p.exists():
        print("  no superseded registry recorded; skipped")
        return
    s = json.loads(p.read_text())
    assert "note" in s and "readings" in s, (
        "a superseded-baselines file must say why it was superseded and carry "
        "the readings it replaced")
    assert s["readings"], "superseded file records no readings"
    for tid, old in s["readings"].items():
        for k in ("median_run", "median_flow", "n_published_runs"):
            assert k in old, f"task {tid} superseded reading is missing {k}"
    print(f"  {len(s['readings'])} superseded baselines retained with their "
          "reason and readings")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
