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
              "median_run": 0.8, "median_flow": 0.75,
              # the four readings and the strictest of them, as the registry
              # carries them; `median_flow_best` is deliberately BELOW
              # `median_run` on one of these because that inequality is not
              # guaranteed and task 10101 is a real counterexample (0.7500 vs
              # 0.7634)
              "median_flow_best": 0.78, "median_uploader_best": 0.82,
              "strictest_baseline": "median_uploader_best",
              "strictest_baseline_value": 0.82,
              "q25": 0.7, "q75": 0.85, "q90": 0.9,
              "max_published": 0.95, "min_published": 0.2,
              "n_instances": 1, "n_features": 1,
              "n_classes": 2, "n_published_runs": 10, "n_distinct_flows": 3,
              "n_distinct_uploaders": 2} for i in range(n)]
    return {"generated": "t", "metric": "predictive_accuracy",
            "selected_task_ids": [t["task_id"] for t in tasks],
            "selection_rule": {"source": "s", "filter": "f", "top_k": n,
                               "n_with_evals": 40, "rank_by": "r"},
            "run_protocol": {"seeds_screen": [0, 1, 2],
                             "seeds_verdict": list(range(8)),
                             "development_tasks": "ranks 6..15",
                             "confirmatory_tasks": "ranks 1..5",
                             "escalation_rule": "near the line -> 8 seeds",
                             "every_attempt_recorded": "ledger"},
            "tasks": tasks}


def _run(tid, acc, seed=0, **kw):
    r = {"task_id": tid, "accuracy_pooled": acc, "random_state": seed,
         "n_interventions": 0, "families_chosen": ["hgb"], "complete": True,
         "off_registry": False, "registry_sha256": "same",
         "env": {"ads_sha256": "same", "ads_dirty_vs_head": False},
         "_file": f"task_{tid}_seed{seed}.json"}
    r.update(kw)
    return r


VERDICT_SEEDS = tuple(range(8))

# A reconciled attempt ledger. Passed explicitly by every fixture that expects
# clause 3 to hold, because an absent ledger now sinks it: without one, "no
# attempt was started and abandoned" is a claim with no record behind it.
LEDGER_OK = {"ledger_present": True, "reconciled": True}

# Amendment 7: clause 2 now also requires a fresh, powered, clean leakage
# probe. The fixtures below supply one, so tests about the *other* gates keep
# testing those gates; the leakage gate itself is exercised directly in
# `test_the_leakage_gate_*` below.
AGENT_SHA = "same"   # matches the digest _run() stamps
POWER_OK = {"min_folds_for_detection": {"100": 1, "101": 1},
            "tasks_unpowered_pooled": [102, 103, 104]}
LEAK_OK = {"verdict": "NO_LEAKAGE_DETECTED",
           "task_ids": [100, 101],
           "tasks_cleared": [100, 101],
           "env": {"ads_sha256": AGENT_SHA}}


def _verdict(*a, leakage=LEAK_OK, power=POWER_OK, **kw):
    """`R.verdict` with a clean leakage probe injected by default."""
    return R.verdict(*a, leakage=leakage, power=power, **kw)


def _bench(n_tasks=5, acc=0.80, seeds=VERDICT_SEEDS):
    """One run per registered *verdict* seed.

    These fixtures used to carry a single seedless run per task, which passed
    only because the fixture registry had no `run_protocol` for the seed
    accounting to check against. Adding one turned them red -- correctly.

    They then carried the three screen seeds, and asserted `status == PASS` off
    them. That is no longer reachable and the change is deliberate: the
    pre-registered exact sign test is now the gate for every task and its
    p-value floor of 1/2^n cannot clear alpha=0.05 at n=3, so a 3-seed screen
    cannot produce a PASS. Fixtures that want a PASS must supply the full
    registered verdict set, which is what the protocol asked for all along.
    """
    return {100 + i: [_run(100 + i, acc, seed=s) for s in seeds]
            for i in range(n_tasks)}


def _rows(base, acc=0.80, seeds=VERDICT_SEEDS, baseline=0.8):
    """Clause rows carrying the escalation state the verdict now requires."""
    e = R.escalation_state([acc] * len(seeds), baseline,
                           base["run_protocol"])
    return [{"task_id": t, "ours": acc, "primary_pass": True,
             "strict_pass": True, "escalation": e, "escalation_strict": e}
            for t in base["selected_task_ids"]]


def test_verdict_is_derived_and_a_single_failing_task_sinks_clause_2():
    base = _base()
    bench = _bench(5)
    rows = _rows(base)
    v = _verdict(rows, base, bench, None, LEDGER_OK)
    assert v["status"] == "PASS", v
    rows[3]["primary_pass"] = False
    v = _verdict(rows, base, bench, None, LEDGER_OK)
    assert v["clauses"]["2_within_5pct_of_human_baseline"] is False
    assert v["status"] == "NOT MET as measured", v["status"]
    print("  one failing task flips clause 2 and the status")


def test_a_single_intervention_sinks_clause_3():
    base = _base()
    bench = _bench(5)
    bench[102][0]["n_interventions"] = 1
    rows = _rows(base)
    v = _verdict(rows, base, bench, None, LEDGER_OK)
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["n_interventions_total"] == 1
    print("  one logged intervention flips clause 3")


def test_partial_and_off_registry_runs_cannot_pass():
    base = _base()
    rows = _rows(base)
    for key, val in (("complete", False), ("off_registry", True)):
        bench = _bench(5)
        bench[101][0][key] = val
        v = _verdict(rows, base, bench, None, LEDGER_OK)
        assert v["clauses"]["3_end_to_end_no_intervention"] is False, key
    print("  a truncated run and an off-registry run both sink clause 3")


def test_four_measured_tasks_is_not_a_pass():
    base = _base()
    bench = _bench(4)
    rows = [{"task_id": t, "ours": 0.80 if t < 104 else None,
             "primary_pass": True if t < 104 else None}
            for t in base["selected_task_ids"]]
    v = _verdict(rows, base, bench, None, LEDGER_OK)
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


def test_the_provenance_gate_fails_closed_on_missing_evidence():
    """codex, 2026-09-10, asked how to make this clause pass legitimately:

        "The provenance gate fails open. Missing intervention counts become
        zero [...]; missing agent digests are discarded [...]; an absent
        ledger is explicitly accepted. Direction: flattering. [...] Agreement
        among available self-reports is weaker than evidence of no
        intervention."

    Correct on all three, and all three are the flattering direction: a run
    that omits a field was being read as a run that reported the field clean.
    Each route gets a case here.
    """
    base = _base()
    rows = _rows(base)
    ledger = LEDGER_OK

    # baseline: with a reconciled ledger and complete records, clause 3 holds
    v = _verdict(rows, base, _bench(5), None, ledger)
    assert v["clauses"]["3_end_to_end_no_intervention"] is True, v

    # 1. a run with no n_interventions field cannot testify that it is zero
    bench = _bench(5)
    del bench[102][0]["n_interventions"]
    v = _verdict(rows, base, bench, None, ledger)
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["runs_missing_a_required_field"], v
    assert v["n_interventions_total"] == 0, (
        "the missing field must not be summed as anything but zero; it is the "
        "*absence* that sinks the clause, not a fabricated count")

    # 2. a run with no agent digest must not be silently dropped from the
    #    'all runs came from one agent' comparison
    bench = _bench(5)
    bench[103][1]["env"] = {"ads_dirty_vs_head": False}
    v = _verdict(rows, base, bench, None, ledger)
    assert v["clauses"]["3_end_to_end_no_intervention"] is False
    assert v["runs_missing_an_agent_digest"], v
    assert v["distinct_agent_source_digests_across_runs"] == 2, (
        "an absent digest was discarded, so 39 runs agreeing outvoted the one "
        "that said nothing")

    # 3. no ledger at all means 'no failed attempts' is unfalsifiable
    for absent in (None, {"ledger_present": False}):
        v = _verdict(rows, base, _bench(5), None, absent)
        assert v["clauses"]["3_end_to_end_no_intervention"] is False, absent
        assert v["ledger_present_and_reconciled"] is False

    print("  missing intervention count, missing agent digest and absent "
          "ledger each sink clause 3 instead of passing it")


def test_a_seed_with_unrun_tasks_is_excluded_not_counted_as_a_failure():
    """A bug the interim table caught within a minute of the code existing.

    The joint-event reading asks, per seed, "did one unattended run clear all
    five tasks?" The first version treated a task with no run for that seed the
    same as a task that ran and missed, so the seed that happened to be
    mid-flight read "no, missed tasks 3 and 3917" when those two had simply not
    started. Wrong in the pessimistic direction, and it would have put a
    fabricated failure in a table of real ones.
    """
    base = _base()
    sel = {t["task_id"]: t for t in base["tasks"]}
    bench = _bench(5, acc=0.80, seeds=(0, 1))
    # seed 1 of task 104 never ran
    bench[104] = [r for r in bench[104] if r["random_state"] != 1]

    j = R.joint_seed_event(bench, sel, "median_run")
    assert j["n_seeds_seen"] == 2 and j["n_seeds_complete"] == 1
    assert j["n_seeds_incomplete"] == 1
    assert j["n_seeds_sweeping_all_five"] == 1
    by_seed = {e["seed"]: e for e in j["per_seed"]}
    assert by_seed[0]["all_five_cleared"] is True
    assert by_seed[1]["all_five_cleared"] is None, (
        "a seed with an unrun task was called a failed sweep")
    assert j["exact_test_on_the_joint_event"]["n"] == 1, (
        "the incomplete seed entered the test")

    # and a task that ran and genuinely missed IS a failure
    bench2 = _bench(5, acc=0.80, seeds=(0, 1))
    bench2[104][1]["accuracy_pooled"] = 0.10
    j2 = R.joint_seed_event(bench2, sel, "median_run")
    by_seed2 = {e["seed"]: e for e in j2["per_seed"]}
    assert by_seed2[1]["all_five_cleared"] is False
    assert j2["n_seeds_complete"] == 2 and j2["n_seeds_sweeping_all_five"] == 1
    print("  unrun task -> seed excluded (None); run-and-missed -> seed fails")


def test_a_fixture_run_cannot_write_into_the_repositorys_own_documents():
    """Found by this file poisoning WEEKEND.md, on 2026-09-10.

    `test_report_writes_not_measured_rather_than_omitting_a_task` runs main()
    over a temp registry of five fixture tasks named d0..d4. It passes an
    explicit --out and --readme into the temp dir, but --weekend had just been
    added with a default of the REAL WEEKEND.md -- so the fixture's five rows,
    with `[not measured]` accuracies for datasets that do not exist, were
    written into the headline block of the repository's handover document.

    Fixture numbers in a shipped document is the precise failure this
    repository exists to prevent, and it arrived through the generator built to
    prevent it. The rule now: an explicit --out means the caller is writing
    somewhere else and must not touch sibling documents unless it names them.
    """
    real_weekend = REPO / "WEEKEND.md"
    real_readme = REPO / "README.md"
    before = (real_weekend.read_text() if real_weekend.exists() else None,
              real_readme.read_text() if real_readme.exists() else None)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "baselines.json").write_text(json.dumps(_base()))
        (d / "bench").mkdir()
        sys.argv = ["report.py", "--baselines", str(d / "baselines.json"),
                    "--bench", str(d / "bench"),
                    "--out", str(d / "RESULTS.md")]
        R.main()
        assert (d / "RESULTS.md").exists()
    after = (real_weekend.read_text() if real_weekend.exists() else None,
             real_readme.read_text() if real_readme.exists() else None)
    assert before == after, (
        "a fixture run modified the repository's own WEEKEND.md or README.md")
    # Look for a fixture *table row*, not for the bare string "d0": both
    # WEEKEND.md and critique_log.md legitimately describe this bug by name,
    # and a substring check turns documenting a defect into a test failure --
    # the same mistake as fingerprinting comments along with code.
    for doc, name in ((after[0], "WEEKEND.md"), (after[1], "README.md")):
        if doc:
            for tid in range(100, 105):
                assert f"| {tid} | d{tid - 100} |" not in doc, (
                    f"a fixture table row leaked into {name}")
    print("  a fixture run with an explicit --out leaves WEEKEND.md and "
          "README.md untouched")


def test_a_partial_record_is_excluded_from_the_average_and_sinks_clause_3():
    """The third appearance of this bug class this weekend.

    A concurrent instance of this same loop ran
    `run_benchmark.py --role dev --tasks 37 --seeds 0 --max-folds 1` into
    `runs/dev/` while a successor measurement was queued to write there. That
    one-fold debug record would have been averaged into the task's accuracy as
    though it were the ten-fold measurement, and the runner would then have
    *skipped* the real run because the file existed.

    A one-fold accuracy is not a noisy ten-fold accuracy; it is a different
    statistic whose bias depends on which fold happened to run. So partials are
    separated from the sample, reported, and cannot coexist with a passing
    clause 3. Same rule as the joint-event fix and as etch-operator-twin's
    partial checkpoints: absence and partiality route to "no result", never to
    a value.
    """
    import tempfile as _tf
    base = _base()
    with _tf.TemporaryDirectory() as d:
        d = Path(d)
        for tid in base["selected_task_ids"]:
            for seed in VERDICT_SEEDS:
                (d / f"task_{tid}_seed{seed}.json").write_text(
                    json.dumps(_run(tid, 0.80, seed=seed)))
        full, failed, partial = R.load_bench(d)
        assert not partial and not failed
        assert all(len(v) == len(VERDICT_SEEDS) for v in full.values())

        # the twin's artifact: one fold, complete=False, max_folds=1
        bad = _run(104, 0.10, seed=0)
        bad.update({"complete": False, "max_folds": 1, "n_folds_run": 1})
        (d / "task_104_seed0.json").write_text(json.dumps(bad))
        full2, failed2, partial2 = R.load_bench(d)

        assert len(partial2) == 1, partial2
        assert partial2[0]["_file"] == "task_104_seed0.json"
        accs = [r["accuracy_pooled"] for r in full2[104]]
        assert 0.10 not in accs, (
            "the one-fold debug record was averaged into task 104's accuracy")
        assert len(full2[104]) == len(VERDICT_SEEDS) - 1, (
            "the partial replaced the real seed-0 record in the sample")

        v = _verdict(_rows(base), base, full2, failed2, LEDGER_OK, partial2)
        assert v["clauses"]["3_end_to_end_no_intervention"] is False
        assert v["n_partial_records_excluded_from_the_average"] == 1
        assert "max_folds=1" in v["partial_records"][0]
    print("  a --max-folds record is kept out of the average, reported, and "
          "sinks clause 3")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
