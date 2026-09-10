"""Regenerate RESULTS.md and the README baseline block from `runs/*.json`.

This is the only code in the repository permitted to write a number into a
document.  It reads `runs/baselines.json` (stage 1) and `runs/bench/*.json`
(stage 2) and nothing else; if a stage has not run, its rows read
`[not measured]` rather than being omitted, because an omitted row is how a
report ends up flattering.

The project verdict is **computed** by `verdict()` from the clause rows, never
typed.  A previous project in this workspace shipped a hard-coded "both clauses
are met" string that survived the numbers moving underneath it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NM = "[not measured]"


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def cell(x):
    return str(x).replace("|", "\\|")


def table(rows, header):
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, n=4):
    return NM if x is None else f"{x:.{n}f}"


# ------------------------------------------------------------------ tolerance
# A result landing exactly on the 5% line must not be decided by float noise:
# |0.95 - 1.00| / 1.00 evaluates to 0.050000000000000044 in IEEE754, which would
# fail a run that is exactly at tolerance. EPS is 1e-12 -- eleven orders of
# magnitude below any accuracy difference this benchmark can resolve, so it can
# only ever rescue an exact tie, never widen the tolerance.
EPS = 1e-12


def tolerance_readings(ours: float, base: float) -> dict:
    """The three readings of '±5% 이내', all fixed in runs/baselines.json."""
    return {
        "rel_one_sided": bool(ours >= 0.95 * base - EPS),     # primary
        "rel_two_sided": bool(abs(ours - base) / base <= 0.05 + EPS),
        "abs_two_sided": bool(abs(ours - base) <= 0.05 + EPS),
        "rel_gap": float((ours - base) / base),
        "abs_gap": float(ours - base),
    }


BASELINE_KEYS = ("median_run", "median_flow", "median_flow_best",
                 "median_uploader_best")


# ------------------------------------------------------------ the exact test
# `run_protocol.escalation_rule` registered that a task near the 5% line gets
# the 8-seed set "and an exact test", without saying which test. codex, asked
# how it would fake this KPI, named that gap as attack #2: stop at three
# favourable seeds and report, because nothing enforced the escalation and no
# test was specified. So it is specified here.
#
# The baseline is a fixed constant (it is a published statistic, not a seeded
# arm), so there is nothing to pair against and a paired sign-flip test does
# not apply. The question is one-sample: is the centre of our seed
# distribution above the threshold T = 0.95 x baseline?
#
#   H0: median over seeds of accuracy_pooled <= T
#   statistic: k = #{seeds with accuracy > T}, over n = #{seeds != T}
#   under H0 at the boundary, k ~ Binomial(n, 1/2), so the one-sided
#   p-value is P(X >= k), computed exactly rather than approximated.
#
# At the registered 8 seeds, all 8 above the line gives p = 1/256 = 0.0039;
# 7 of 8 gives 0.0352; 6 of 8 gives 0.1445 and does not clear alpha = 0.05.
# That is the intended strictness: a task that only just clears the line on a
# majority of seeds does not get called.
ALPHA = 0.05


def exact_sign_test_above(accs: list[float], threshold: float) -> dict:
    """One-sided exact sign test of H0: median(accs) <= threshold."""
    nonties = [a for a in accs if a != threshold]
    n = len(nonties)
    k = sum(1 for a in nonties if a > threshold)
    if n == 0:
        return {"n": 0, "k": 0, "p": 1.0, "reject_h0": False}
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return {"n": n, "k": k, "p": float(p), "reject_h0": bool(p <= ALPHA)}


def escalation_state(accs: list[float], baseline: float, protocol: dict) -> dict:
    """Is this task near the line, and does the protocol allow calling it?

    "Near the line" is measured against our own noise, not against a number
    picked by hand: a task is near the line when the distance from the mean to
    the threshold is smaller than the observed seed range. That is the
    weekend's rule -- an effect smaller than the run-to-run spread is not an
    effect -- applied to the pass/fail decision itself.
    """
    T = 0.95 * baseline
    n_verdict = len(protocol.get("seeds_verdict") or []) or 8
    mean = statistics.mean(accs)
    spread = (max(accs) - min(accs)) if len(accs) >= 2 else None
    margin = abs(mean - T)
    near = bool(spread is not None and margin < spread)
    test = exact_sign_test_above(accs, T) if near else None
    # A near-the-line task run on fewer than the registered verdict seeds
    # cannot be called either way: that is the escalation the protocol
    # registered, and leaving it unenforced was how three favourable screen
    # seeds could have become a headline.
    needs_more = bool(near and len(accs) < n_verdict)
    return {
        "threshold": float(T), "mean": float(mean),
        "margin_to_threshold": float(margin),
        "seed_range": None if spread is None else float(spread),
        "near_line": near, "n_seeds": len(accs),
        "n_seeds_registered_for_verdict": n_verdict,
        "escalation_required": needs_more,
        "exact_test": test,
        # A task is *called* when it is comfortably clear of the line (margin
        # exceeds our own spread), or when the exact test at the registered
        # seed count rejects H0.
        "called": bool((not near and mean >= T)
                       or (near and not needs_more and test
                           and test["reject_h0"])),
        "screen_only": bool(len(accs) <= 3),
    }


def load_bench(bench_dir: Path) -> tuple[dict[int, list[dict]], list[dict]]:
    """Successful runs by task, and every failed attempt.

    Failures are returned, not dropped: a report that silently averages
    whichever seed files happen to have survived is choosing its own sample.
    """
    by_task: dict[int, list[dict]] = {}
    failed: list[dict] = []
    for f in sorted(bench_dir.glob("task_*.json")):
        r = read_json(f)
        if r is None:
            continue
        r["_file"] = f.name
        if f.name.endswith(".FAILED.json"):
            failed.append(r)
        else:
            by_task.setdefault(int(r["task_id"]), []).append(r)
    return by_task, failed


def reconcile_ledger(ledger_path: Path, bench: dict,
                     failed: list[dict] | None = None) -> dict:
    """Compare the append-only attempt ledger against the files on disk.

    The attack this answers (codex, #4) is deletion, not fabrication: move the
    unfavourable run files out of `runs/bench/` and every check still passes
    on what remains. The ledger records each attempt before its outcome is
    known, so a result that was produced and then removed leaves a `completed`
    line with no file behind it.
    """
    if not Path(ledger_path).exists():
        return {"ledger_present": False}
    events = []
    for line in Path(ledger_path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event": "unparseable", "raw": line[:120]})
    def key(e):
        return (e.get("task_id"), e.get("seed"))
    started = {key(e) for e in events if e.get("event") == "started"}
    completed = {key(e) for e in events if e.get("event") == "completed"}
    failed_ev = {key(e) for e in events if e.get("event") == "failed"}
    on_disk = {(int(tid), r.get("random_state"))
               for tid, runs in bench.items() for r in runs}
    failed_files = {(f.get("task_id"), f.get("seed")) for f in (failed or [])}
    # started but never resolved: a killed or still-running attempt
    unresolved = sorted(started - completed - failed_ev)
    # completed in the ledger but absent from runs/bench: a deleted result
    missing_from_disk = sorted(completed - on_disk)
    # present on disk with no ledger entry: a run that bypassed the runner
    unledgered = sorted(on_disk - completed)
    return {
        "ledger_present": True,
        "n_events": len(events),
        "n_started": len(started),
        "n_completed": len(completed),
        "n_failed": len(failed_ev),
        "attempts_started_but_unresolved": [list(k) for k in unresolved],
        "completed_but_missing_from_disk": [list(k) for k in missing_from_disk],
        "on_disk_but_not_in_ledger": [list(k) for k in unledgered],
        "failed_attempts_with_records": len(failed_files & failed_ev),
        "reconciled": not (unresolved or missing_from_disk or unledgered),
    }


def verdict(rows: list[dict], base: dict, bench: dict,
            failed: list[dict] | None = None, ledger: dict | None = None) -> dict:
    """Derive the project status from the clause rows. Never a typed string."""
    n_tasks = len(base.get("selected_task_ids", [])) if base else 0
    clause1 = n_tasks == 5
    measured = [r for r in rows if r["ours"] is not None]
    # A task sitting closer to the 5% line than our own seed spread cannot be
    # called on the 3-seed screen; the registered escalation to 8 seeds and an
    # exact test has to actually happen first. Without this, three favourable
    # screen seeds were enough to claim the clause (codex, attack #2).
    escalation_pending = [r["task_id"] for r in measured
                          if (r.get("escalation") or {}).get("escalation_required")]
    not_called = [r["task_id"] for r in measured
                  if (r.get("escalation") or {}).get("near_line")
                  and not (r.get("escalation") or {}).get("called")]
    clause2 = (bool(measured) and len(measured) == n_tasks
               and all(r["primary_pass"] for r in measured)
               and not escalation_pending and not not_called)
    # The primary reading is `median_run` and it stays the primary -- it was
    # pre-registered and the brief named it. But codex is right that a PASS on
    # it while the selected-solution readings fail is a weak claim, and right
    # that `median_flow_best >= median_run` is NOT a mathematical guarantee,
    # only `median_flow_best >= median_flow` is. So the strictest-baseline
    # verdict is computed and reported beside the primary one rather than
    # being recorded and left unbinding.
    clause2_strict = bool(measured) and len(measured) == n_tasks and all(
        r.get("strict_pass") for r in measured)
    failed = failed or []
    interventions = sum(
        run.get("n_interventions", 0) for runs in bench.values() for run in runs)
    off_registry = any(run.get("off_registry") for runs in bench.values()
                       for run in runs)
    incomplete = [run["_file"] for runs in bench.values() for run in runs
                  if not run.get("complete", False)]
    # Which seeds did the protocol register, and which are actually present?
    #
    # `ours` is the mean over run *files*, but this completeness check used to
    # be over the *set* of random_state values. codex, asked how it would fake
    # this KPI, went looking exactly there, and the hole is real: copy the
    # best-scoring seed's record to a second filename matching `task_*.json`
    # and the mean moves (0.7567 -> 0.7660 on a three-seed screen, verified in
    # tests/test_seed_accounting.py) while the set still reads [0, 1, 2] and
    # every clause still passes. A set cannot see a duplicate.
    #
    # So the run files must be a *bijection* with the registered seeds: no
    # duplicate of a seed, and no seed outside the registered protocol. Both
    # sink clause 3 and both are printed, because a report that silently
    # averages whichever files happen to be on disk is choosing its own sample.
    protocol = base.get("run_protocol") or {}
    planned = list(protocol.get("seeds_screen") or [])
    registered = set(planned) | set(protocol.get("seeds_verdict") or [])
    seed_counts = {tid: Counter(r.get("random_state") for r in runs)
                   for tid, runs in bench.items()}
    seeds_present = {tid: sorted(c) for tid, c in seed_counts.items()}
    seeds_missing = {tid: [s_ for s_ in planned if s_ not in got]
                     for tid, got in seeds_present.items()}
    seeds_missing = {k: v for k, v in seeds_missing.items() if v}
    seeds_duplicated = {tid: {str(s_): n for s_, n in c.items() if n > 1}
                        for tid, c in seed_counts.items()}
    seeds_duplicated = {k: v for k, v in seeds_duplicated.items() if v}
    seeds_unregistered = {tid: sorted(s_ for s_ in c if s_ not in registered)
                          for tid, c in seed_counts.items()} if registered else {}
    seeds_unregistered = {k: v for k, v in seeds_unregistered.items() if v}
    registry_sha = {r.get("registry_sha256") for runs in bench.values()
                    for r in runs}
    # Two runs produced by different versions of `ads/`, or by a dirty tree,
    # are not one measurement of one agent (codex, attack #5).
    agent_sha = {(r.get("env") or {}).get("ads_sha256")
                 for runs in bench.values() for r in runs}
    agent_sha.discard(None)
    dirty_runs = [r["_file"] for runs in bench.values() for r in runs
                  if (r.get("env") or {}).get("ads_dirty_vs_head")]
    clause3 = (bool(measured) and interventions == 0 and not off_registry
               and not incomplete and not failed and not seeds_missing
               and not seeds_duplicated and not seeds_unregistered
               and len(registry_sha) <= 1
               and len(agent_sha) <= 1 and not dirty_runs
               and (ledger is None or not ledger.get("ledger_present")
                    or ledger.get("reconciled")))
    clauses = {
        "1_five_public_datasets": clause1,
        "2_within_5pct_of_human_baseline": clause2 if measured else None,
        "3_end_to_end_no_intervention": clause3 if measured else None,
    }
    if all(v is True for v in clauses.values()):
        status = "PASS"
    elif any(v is None for v in clauses.values()):
        status = "RUNNING (not all clauses measured)"
    else:
        status = "NOT MET as measured"
    return {"status": status, "clauses": clauses,
            "clause2_under_strictest_baseline": clause2_strict if measured else None,
            "escalation_pending_tasks": escalation_pending,
            "near_line_not_called_tasks": not_called,
            "n_interventions_total": interventions,
            "off_registry_runs": off_registry,
            "incomplete_runs": incomplete,
            "n_failed_attempts": len(failed),
            "failed_attempts": [f["_file"] for f in failed],
            "seeds_present": {str(k): v for k, v in seeds_present.items()},
            "seeds_registered_but_missing": {str(k): v for k, v
                                             in seeds_missing.items()},
            "seeds_duplicated": {str(k): v for k, v in seeds_duplicated.items()},
            "seeds_not_in_registered_protocol": {
                str(k): v for k, v in seeds_unregistered.items()},
            "distinct_registry_digests_across_runs": len(registry_sha),
            "distinct_agent_source_digests_across_runs": len(agent_sha),
            "runs_from_a_dirty_agent_tree": dirty_runs,
            "ledger": ledger or {"ledger_present": False},
            "n_tasks_measured": len(measured), "n_tasks_registered": n_tasks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(REPO / "runs/baselines.json"))
    ap.add_argument("--bench", default=str(REPO / "runs/bench"))
    ap.add_argument("--out", default=str(REPO / "RESULTS.md"))
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    args = ap.parse_args()

    base = read_json(args.baselines)
    bench, failed = (load_bench(Path(args.bench))
                     if Path(args.bench).exists() else ({}, []))

    doc: list[str] = ["# Results", ""]
    if base is None:
        doc += [f"`runs/baselines.json` is absent, so every number here is "
                f"`{NM}`. Run `scripts/fetch_baselines.py` first.", ""]
        Path(args.out).write_text("\n".join(doc) + "\n")
        print(f"wrote {args.out} (no baselines yet)")
        return 0

    sel = {t["task_id"]: t for t in base["tasks"] if t["selected"]}
    order = base["selected_task_ids"]

    # ---------------------------------------------------------- baseline block
    bl_rows = []
    for tid in order:
        t = sel[tid]
        bl_rows.append([
            tid, t["dataset_name"], t["n_instances"], t["n_features"],
            t["n_classes"], t["n_published_runs"], t["n_distinct_flows"],
            t["n_distinct_uploaders"], fmt(t["median_run"]),
            fmt(t["median_flow"]), fmt(t["median_flow_best"]),
            fmt(t["median_uploader_best"]), t["strictest_baseline"],
            fmt(t["q90"]), fmt(t["max_published"])])
    bl_table = table(bl_rows, [
        "task", "dataset", "n", "p", "classes", "published runs", "flows",
        "uploaders", "**median_run** (primary)", "median_flow",
        "median_flow_best", "median_uploader_best", "strictest",
        "q90 (context)", "max (context)"])

    rule = base["selection_rule"]
    bl_block = "\n".join([
        "### The five tasks and their pre-registered human baselines", "",
        f"Fetched by `scripts/fetch_baselines.py` on **{base['generated']}**, "
        f"before any agent run existed. Selected from {rule['source']} by: "
        f"`{rule['filter']}`, then the top {rule['top_k']} of "
        f"{rule['n_with_evals']} by `{rule['rank_by']}`.", "",
        bl_table, "",
        f"Metric is `{base['metric']}` under each task's own estimation "
        "procedure, and it is the **size-weighted (pooled)** accuracy — "
        "measured, not assumed, in `runs/metric_check.json`. Raw evaluations "
        "are committed under `runs/evals/` with a sha256 per file in "
        "`runs/baselines.json`, and "
        "`tests/test_registry_frozen.py` recomputes every reading above from "
        "them, so a baseline cannot be edited without the tests failing.", "",
        "`median_run` counts a 5000-point sweep 5000 times and scores our "
        "*selected* model against the distribution of *all* human trials, "
        "failures included — an asymmetry that flatters us. "
        "`median_flow_best` and `median_uploader_best` are the symmetric "
        "readings: their selected solution against ours. All four are "
        "reported for every task.", ""])

    # ------------------------------------------------------------ our results
    rows, res_rows, spread_rows = [], [], []
    for tid in order:
        t = sel[tid]
        runs = bench.get(tid, [])
        accs = [r["accuracy_pooled"] for r in runs]
        ours = statistics.mean(accs) if accs else None
        row = {"task_id": tid, "name": t["dataset_name"], "ours": ours,
               "primary_pass": None}
        if ours is not None:
            tol_run = tolerance_readings(ours, t["median_run"])
            tol_flow = tolerance_readings(ours, t["median_flow"])
            tol_strict = tolerance_readings(ours, t["strictest_baseline_value"])
            row["primary_pass"] = tol_run["rel_one_sided"]
            row["strict_pass"] = tol_strict["rel_one_sided"]
            row["escalation"] = escalation_state(
                accs, t["median_run"], base.get("run_protocol") or {})
            res_rows.append([
                tid, t["dataset_name"], len(accs), fmt(ours),
                fmt(t["median_run"]), f"{tol_run['rel_gap']*100:+.2f}%",
                "PASS" if tol_run["rel_one_sided"] else "FAIL",
                "PASS" if tol_run["rel_two_sided"] else "FAIL",
                "PASS" if tol_run["abs_two_sided"] else "FAIL",
                fmt(t["median_flow"]),
                "PASS" if tol_flow["rel_one_sided"] else "FAIL",
                t["strictest_baseline"], fmt(t["strictest_baseline_value"]),
                "PASS" if tol_strict["rel_one_sided"] else "FAIL",
                ",".join(sorted({f for r in runs for f in r["families_chosen"]}))])
            if len(accs) >= 2:
                spread_rows.append([
                    tid, t["dataset_name"], len(accs), fmt(min(accs)),
                    fmt(max(accs)), fmt(max(accs) - min(accs)),
                    fmt(statistics.pstdev(accs)),
                    f"{abs(tol_run['rel_gap'])*100:.2f}%"])
        else:
            res_rows.append([tid, t["dataset_name"], 0, NM,
                             fmt(t["median_run"]), NM, NM, NM, NM,
                             fmt(t["median_flow"]), NM,
                             t["strictest_baseline"],
                             fmt(t["strictest_baseline_value"]), NM, NM])
        rows.append(row)

    ledger = reconcile_ledger(Path(args.bench).parent / "attempts.jsonl",
                              bench, failed)
    v = verdict(rows, base, bench, failed, ledger)

    doc += [f"**Status: {v['status']}**", "",
            "Every number below was produced by a run in this repository and is "
            "regenerated from `runs/*.json` by `scripts/report.py`. Nothing is "
            "hand-typed.", "",
            "## KPI, clause by clause", "",
            table([["1", "공개 데이터셋 5개",
                    f"{v['n_tasks_registered']} registered, "
                    f"{v['n_tasks_measured']} measured",
                    str(v["clauses"]["1_five_public_datasets"])],
                   ["2", "사람 baseline ±5% 이내 자동도달",
                    "primary reading: ours >= 0.95 x median_run, on all five",
                    str(v["clauses"]["2_within_5pct_of_human_baseline"])],
                   ["3", "end-to-end 무개입",
                    f"{v['n_interventions_total']} interventions logged; "
                    f"off-registry runs: {v['off_registry_runs']}; "
                    f"incomplete: {len(v['incomplete_runs'])}; "
                    f"failed attempts: {v['n_failed_attempts']}; "
                    f"registered seeds missing: "
                    f"{len(v['seeds_registered_but_missing'])}; "
                    f"duplicated seeds: {len(v['seeds_duplicated'])}; "
                    f"seeds outside the protocol: "
                    f"{len(v['seeds_not_in_registered_protocol'])}; "
                    f"distinct agent-source digests: "
                    f"{v['distinct_agent_source_digests_across_runs']}; "
                    f"runs from a dirty agent tree: "
                    f"{len(v['runs_from_a_dirty_agent_tree'])}; "
                    f"ledger reconciled: "
                    f"{v['ledger'].get('reconciled') if v['ledger'].get('ledger_present') else '[no ledger]'}",
                    str(v["clauses"]["3_end_to_end_no_intervention"])]],
                  ["#", "clause", "measured as", "met"]), "",
            "## Our accuracy against the pre-registered baselines", "",
            table(res_rows, [
                "task", "dataset", "seeds", "ours (pooled)",
                "median_run", "rel gap", "primary (>=0.95x)", "rel 2-sided",
                "abs 2-sided", "median_flow", "vs flow", "strictest reading",
                "strictest value", "vs strictest", "families chosen"]),
            "",
            "`rel gap` is `(ours - median_run) / median_run`; positive means we "
            "are above the median published run. The three tolerance columns are "
            "the three readings fixed in `runs/baselines.json` before any run — "
            "all are shown so that none can be picked after the fact.", ""]

    if spread_rows:
        doc += ["## Our own run-to-run spread", "",
                "An effect smaller than this noise floor is not an effect. Each "
                "seed re-draws the agent's inner CV shuffle, its selection "
                "subsample and its random search; the outer folds are the "
                "task's own and are identical across seeds.", "",
                table(spread_rows, ["task", "dataset", "seeds", "min", "max",
                                    "range", "sd", "|gap to baseline|"]), ""]
    else:
        doc += ["## Our own run-to-run spread", "",
                f"{NM} — fewer than two seeds per task so far.", ""]

    doc += [bl_block, "## Provenance", "",
            table([[k, str(vv)] for k, vv in v.items() if k != "clauses"],
                  ["field", "value"]), ""]

    Path(args.out).write_text("\n".join(doc) + "\n")

    # ------------------------------------------------- README baseline block
    rp = Path(args.readme)
    if rp.exists():
        txt = rp.read_text()
        new = re.sub(r"<!-- BASELINES:BEGIN -->.*?<!-- BASELINES:END -->",
                     "<!-- BASELINES:BEGIN -->\n" + bl_block +
                     "<!-- BASELINES:END -->", txt, flags=re.S)
        rp.write_text(new)

    print(f"wrote {args.out} and refreshed the README block")
    print(f"status: {v['status']}")
    for k, val in v["clauses"].items():
        print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
