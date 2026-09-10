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
import os
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
#   statistic: k = #{seeds with accuracy STRICTLY > T}, over n = ALL seeds
#   under H0 at the boundary, k ~ Binomial(n, 1/2), so the one-sided
#   p-value is P(X >= k), computed exactly rather than approximated.
#
# Ties stay in n and count as non-wins. Dropping them is the continuous-case
# sign test and is anti-conservative on a discrete metric -- see the docstring
# of exact_sign_test_above for the counterexample and the 17.37% figure.
#
# CHRONOLOGY: the registry registered "an exact test" without naming one, and
# conditioned it on the task being near the line. Naming it, making it
# unconditional, and fixing the tie handling were all done on 2026-09-10 AFTER
# the 3-seed screen had been read, and are recorded as amendments 1 and 2 in
# runs/protocol_amendments.json. Both are strictly stricter; amendment 1 took
# the project status from PASS to RUNNING on identical run data.
#
# At the registered 8 seeds, all 8 above the line gives p = 1/256 = 0.0039;
# 7 of 8 gives 0.0352; 6 of 8 gives 0.1445 and does not clear alpha = 0.05.
# That is the intended strictness: a task that only just clears the line on a
# majority of seeds does not get called.
ALPHA = 0.05


def exact_sign_test_above(accs: list[float], threshold: float) -> dict:
    """One-sided exact sign test of H0: median(accs) <= threshold.

    Ties count as non-wins and stay in the denominator. This is the fix to a
    real Type I inflation, found by codex on 2026-09-10 when asked how to make
    the clause pass; it gave the counterexample and the arithmetic checks out
    (`tests/test_escalation_gate.py::test_ties_are_conservative_...`).

    The earlier version dropped ties and ran a fair binomial on the survivors,
    which is the textbook sign test for a *continuous* distribution. Accuracy
    is discrete -- k correct out of a fixed n -- so ties at the threshold have
    real probability, and for a discrete distribution "the median is T" does
    NOT imply the non-ties split evenly above and below. Take
    P(A = T) = 0.6, P(A > T) = 0.4, nothing below: the median is exactly T so
    H0 is true, yet 5 wins and 3 ties gave p = 1/32 under the old rule and the
    8-seed gate rejected a true null **17.37% of the time** at a nominal 5%.
    Keeping n = 8 and counting only strict wins puts that at 0.85%, i.e.
    conservative, which is the direction an honest gate errs in.

    It costs nothing on this repository's data -- no seed lands exactly on a
    0.95x threshold, so k and n are unchanged for every task measured here.
    It changes the arithmetic only where ties exist, and there it changes it
    from anti-conservative to conservative.
    """
    n = len(accs)
    k = sum(1 for a in accs if a > threshold)
    n_ties = sum(1 for a in accs if a == threshold)
    if n == 0:
        return {"n": 0, "k": 0, "n_ties": 0, "p": 1.0, "reject_h0": False}
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return {"n": n, "k": k, "n_ties": n_ties,
            "ties_counted_as": "non-wins, kept in the denominator",
            "p": float(p), "reject_h0": bool(p <= ALPHA)}


def escalation_state(accs: list[float], baseline: float, protocol: dict) -> dict:
    """Can this task be *called* on the seeds that exist, and by what test?

    Two readings of "is it clear of the line", both reported (the addendum's
    rung 1), because they disagree and the disagreement is the point:

    - `margin_exceeds_seed_range`: the distance from the mean to the threshold
      is larger than the observed seed range. This was the gate until
      2026-09-10 turn 6 and it is **biased in the flattering direction at
      small n**, which is why it is now a diagnostic and not the gate. The
      expected range of n iid draws is 1.69 sigma at n=3 and 2.85 sigma at
      n=8, so a 3-seed range underestimates the spread by ~1.7x on average --
      and the estimate it understates sits in the *denominator* of the
      comparison. A gate that gets easier to pass the fewer seeds you run is
      the wrong shape for a gate, whatever margin it happens to show.

    - `exact_test`: the one-sided exact sign test of
      H0: median <= T -- registered in substance ("an exact test"),
      specified and made unconditional by amendment 1. This is the gate for
      **every** task, near the line or not. Its p-value floor is 1/2^n, so n=3 can reach only p=0.125 and
      **no 3-seed screen can call a task in either direction** -- which is
      what the weekend rule ("with 3 seeds, say screen, not verdict") says,
      now enforced in code rather than in prose. At the registered 8 seeds,
      8/8 above the line gives p=0.0039 and the task is called.

    This is a tightening. The clause got harder to meet, not easier: on the
    3-seed screen of 2026-09-10 every one of the five tasks cleared its
    threshold by 4.5x-68x its own seed range and the old gate called all five,
    giving `status: PASS` off a screen. Under this gate the same five runs
    call nothing and the status is `RUNNING`, pending the 8-seed set.
    """
    T = 0.95 * baseline
    n_verdict = len(protocol.get("seeds_verdict") or []) or 8
    mean = statistics.mean(accs)
    spread = (max(accs) - min(accs)) if len(accs) >= 2 else None
    margin = abs(mean - T)
    margin_beats_range = bool(spread is not None and margin > spread)
    test = exact_sign_test_above(accs, T)
    # Two conditions, and the second is not redundant. The p-value floor 1/2^n
    # only blocks n <= 4; at n=5 an all-above screen gives p=0.03125 and would
    # reject. Calling a task the moment the test happens to clear alpha is
    # optional stopping, and a sequentially-monitored p-value is not the
    # 0.05 it prints. The registered verdict set is 8 seeds, so the full 8 are
    # required whichever way the test comes out -- a task must not become
    # callable by stopping early on a favourable prefix.
    enough_seeds = len(accs) >= n_verdict
    called = bool(test["reject_h0"] and enough_seeds)
    # Fewer seeds than the protocol registered => the escalation is what is
    # missing, so say so rather than reporting a failed clause. Not called *at*
    # the full seed count is a real negative result, not a pending measurement.
    needs_more = not enough_seeds
    return {
        "threshold": float(T), "mean": float(mean),
        "margin_to_threshold": float(margin),
        "seed_range": None if spread is None else float(spread),
        # kept, labelled, and no longer load-bearing
        "margin_exceeds_seed_range": margin_beats_range,
        "near_line": not margin_beats_range,
        "n_seeds": len(accs),
        "n_seeds_registered_for_verdict": n_verdict,
        "escalation_required": needs_more,
        "seeds_meet_registered_verdict_count": enough_seeds,
        "exact_test": test,
        "called": called,
        "screen_only": bool(len(accs) <= 3),
    }


def joint_seed_event(bench: dict, sel: dict, baseline_key: str) -> dict:
    """Per seed: did ONE run of the agent clear the line on all five tasks?

    An additional reading, not the gate. codex, 2026-09-10: "signs discard
    deficit magnitude, and taskwise median success does not establish that one
    autonomous execution clears all five tasks reliably. Flattering if PASS is
    read as dependable end-to-end success. For that claim, additionally report
    the per-seed event 'all five tasks completed and cleared their
    thresholds'."

    That is right, and it is the reading a person actually cares about for an
    *autonomous* data scientist: not "each task passes on a majority of runs"
    but "a single unattended run gets all five". The clause-2 gate stays the
    per-task test -- it is what the protocol registered, and requiring all five
    to reject is an intersection-union test, so testing five tasks at 0.05
    needs no multiplicity correction (also codex, correctly). This row sits
    beside it because the two can disagree: five tasks each failing on a
    *different* seed would pass every per-task test and never once produce a
    clean sweep.
    """
    seeds = sorted({r.get("random_state") for runs in bench.values()
                    for r in runs if r.get("random_state") is not None})
    per_seed = []
    for s in seeds:
        rows, cleared_all, incomplete = [], True, False
        for tid, t in sel.items():
            runs = [r for r in bench.get(tid, []) if r.get("random_state") == s]
            if len(runs) != 1:
                # A task that has not run yet for this seed is NOT a task this
                # seed failed. The first version of this function conflated the
                # two and the interim table immediately showed it: the seed
                # then mid-flight read "no, missed tasks 3 and 3917" when those
                # two simply had not started. Missing => the seed's joint event
                # is undetermined and it leaves the test entirely.
                incomplete = True
                rows.append({"task_id": tid, "cleared": None,
                             "reason": "no run for this seed"
                                       if not runs else "duplicated run"})
                continue
            acc = runs[0]["accuracy_pooled"]
            cleared = bool(acc >= 0.95 * t[baseline_key] - EPS)
            cleared_all = cleared_all and cleared
            rows.append({"task_id": tid, "accuracy": acc, "cleared": cleared})
        per_seed.append({"seed": s,
                         "all_five_cleared": None if incomplete else cleared_all,
                         "complete": not incomplete, "tasks": rows})
    done = [e for e in per_seed if e["all_five_cleared"] is not None]
    swept = sum(1 for e in done if e["all_five_cleared"])
    test = exact_sign_test_above(
        [1.0 if e["all_five_cleared"] else 0.0 for e in done], 0.5) if done else None
    return {
        "baseline_key": baseline_key,
        "n_seeds_seen": len(seeds),
        "n_seeds_complete": len(done),
        "n_seeds_incomplete": len(seeds) - len(done),
        "n_seeds_sweeping_all_five": swept,
        "per_seed": per_seed,
        "exact_test_on_the_joint_event": test,
        "reading": "additional; the clause-2 gate is the per-task test",
    }


# ------------------------------------------------- the chronology-clean subset
# The gate was amended after the 3-seed screen had been read (amendments 1-3 in
# runs/protocol_amendments.json). All three are strictly stricter and #1 took
# the status from PASS to RUNNING on identical data, so the usual objection to
# a post-hoc rule -- that it was tuned to squeeze out a pass -- does not apply.
# But 3 of the 8 verdict seeds had been inspected, and a reader is entitled to
# a reading that owes nothing to them.
#
# So: the exact test restricted to the seeds that had not been run when the
# rule was amended. Those seeds are named IN the amendment, so this is a fixed
# pre-specified subset, not one chosen after seeing outcomes -- and n=5 is
# enough to reject (5 of 5 above the line gives p = 1/32 = 0.03125).
#
# This is reported whichever way it comes out, and it is why no further seed
# set was run. More seeds would shrink seed noise, which is already
# 0.0013-0.0077 against margins of 0.03-0.08; they would do nothing about the
# uncertainty that actually binds, which is that each task is one fixed
# dataset with one fixed set of folds.
def clean_seed_subset(amendments_path: Path) -> set:
    """Seeds that had not been run when the gate was amended, per the record."""
    a = read_json(amendments_path)
    if not a:
        return set()
    out = set()
    for e in a.get("amendments", []):
        out |= set(e.get("seeds_uninspected_at_amendment") or [])
    return out


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


def benchmark_processes_alive() -> bool:
    """Is a `run_benchmark.py` alive right now? Read from /proc, no new deps.

    Deliberately not a lockfile: a lockfile left behind by a killed process
    reads as "in flight" forever, which is the failure direction that hides an
    abandoned attempt.
    """
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue                      # exited or not ours to read
        if "run_benchmark.py" in cmd:
            return True
    return False


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
    # started but never resolved: a killed or still-running attempt. Which of
    # the two it is depends on whether a process is alive *now*, and that must
    # not enter this function: everything here is a pure function of two
    # committed files, so `tests/test_registry_frozen.py` can assert that
    # RESULTS.md is byte-identical to a regeneration on a clean checkout. An
    # earlier version of this fix read /proc here, which would have made the
    # committed document un-reproducible in CI. The live reading belongs to
    # the interim view instead -- see the in-flight guard in main().
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
    # Every task must be *called* by the pre-registered exact sign test, not
    # only the ones that land near the line. Until 2026-09-10 turn 6 the test
    # ran only when the margin was smaller than the observed seed range, and
    # that condition is easier to satisfy the fewer seeds you run (see
    # `escalation_state`), so the 3-seed screen skipped the test on all five
    # tasks and the clause passed on point estimates alone. p has a floor of
    # 1/2^n, so this gate makes a 3-seed screen uncallable by construction.
    escalation_pending = [r["task_id"] for r in measured
                          if (r.get("escalation") or {}).get("escalation_required")]
    not_called = [r["task_id"] for r in measured
                  if not (r.get("escalation") or {}).get("called")]
    clause2 = (bool(measured) and len(measured) == n_tasks
               and all(r["primary_pass"] for r in measured)
               and not escalation_pending and not not_called)
    # A clause whose seeds are still short of the registered verdict set is
    # *unmeasured*, not failed: reporting "NOT MET as measured" off a screen
    # would be as wrong in the pessimistic direction as PASS was in the
    # flattering one. None routes the status to RUNNING.
    if escalation_pending:
        clause2 = None
    # The primary reading is `median_run` and it stays the primary -- it was
    # pre-registered and the brief named it. But codex is right that a PASS on
    # it while the selected-solution readings fail is a weak claim, and right
    # that `median_flow_best >= median_run` is NOT a mathematical guarantee,
    # only `median_flow_best >= median_flow` is. So the strictest-baseline
    # verdict is computed and reported beside the primary one rather than
    # being recorded and left unbinding.
    clause2_strict = bool(measured) and len(measured) == n_tasks and all(
        r.get("strict_pass")
        and (r.get("escalation_strict") or {}).get("called")
        for r in measured)
    if any((r.get("escalation_strict") or {}).get("escalation_required")
           for r in measured):
        clause2_strict = None          # unfinished, not failed
    failed = failed or []
    # ---------------------------------------------------- fail closed, not open
    # codex, 2026-09-10, asked how to make this clause pass legitimately, found
    # that the provenance gate "fails open" in three places and it is right:
    # `n_interventions` defaulted to 0 when the field was absent, a `None`
    # agent digest was *discarded* from the set before counting distinct
    # digests, and an absent ledger was explicitly accepted. Every one of those
    # turns missing evidence into passing evidence, which is the flattering
    # direction. A run that does not carry the field cannot testify that the
    # field is zero.
    required = ("n_interventions", "complete", "off_registry",
                "registry_sha256", "random_state")
    runs_missing_fields = sorted(
        f"{run['_file']}:{f}"
        for runs in bench.values() for run in runs
        for f in required if run.get(f) is None)
    runs_missing_agent_digest = sorted(
        run["_file"] for runs in bench.values() for run in runs
        if not (run.get("env") or {}).get("ads_sha256"))
    interventions = sum(
        run.get("n_interventions") or 0 for runs in bench.values() for run in runs)
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
    screen = list(protocol.get("seeds_screen") or [])
    verdict_seeds = list(protocol.get("seeds_verdict") or [])
    registered = set(screen) | set(verdict_seeds)
    # Which seed set is this run *claiming* to be? `agy`, asked what could
    # still be faked, found (its #7) that `planned` was always the screen, so
    # clause 3 was satisfied by seeds 0-2 alone and never checked that verdict
    # seeds 3-7 arrived: an escalation abandoned halfway left clause 3 clean.
    # Direction: flattering. So the bar is the screen until a seed outside the
    # screen appears -- at which point the run has embarked on the verdict set
    # and owes all of it. Nothing here can *lower* the bar: the escalated
    # requirement is a superset of the screen.
    embarked_on_verdict = any(
        s_ not in set(screen)
        for runs in bench.values() for r in runs
        for s_ in [r.get("random_state")] if s_ is not None)
    planned = verdict_seeds if embarked_on_verdict else screen
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
    # NOT discarded: a run with no digest is counted as its own distinct
    # digest, so "all runs agree" cannot be satisfied by runs that said nothing.
    agent_sha = {(r.get("env") or {}).get("ads_sha256") or f"[absent:{r['_file']}]"
                 for runs in bench.values() for r in runs}
    dirty_runs = [r["_file"] for runs in bench.values() for r in runs
                  if (r.get("env") or {}).get("ads_dirty_vs_head")]
    # An absent ledger is now a hole, not a pass: the ledger is the only record
    # of an attempt that was started and never produced a file, so without it
    # "no failed attempts" is an unfalsifiable claim.
    ledger_ok = bool(ledger and ledger.get("ledger_present")
                     and ledger.get("reconciled"))
    clause3 = (bool(measured) and interventions == 0 and not off_registry
               and not incomplete and not failed and not seeds_missing
               and not seeds_duplicated and not seeds_unregistered
               and not runs_missing_fields and not runs_missing_agent_digest
               and len(registry_sha) <= 1
               and len(agent_sha) <= 1 and not dirty_runs
               and ledger_ok)
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
            "runs_missing_a_required_field": runs_missing_fields,
            "runs_missing_an_agent_digest": runs_missing_agent_digest,
            "ledger_present_and_reconciled": ledger_ok,
            "off_registry_runs": off_registry,
            "incomplete_runs": incomplete,
            "n_failed_attempts": len(failed),
            "failed_attempts": [f["_file"] for f in failed],
            "seeds_present": {str(k): v for k, v in seeds_present.items()},
            "seed_set_required_of_this_run": planned,
            "embarked_on_the_verdict_seed_set": embarked_on_verdict,
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
    ap.add_argument("--out", default=None,
                    help="default RESULTS.md. An explicit path is never "
                         "redirected by the in-flight guard, so "
                         "tests/test_registry_frozen.py can regenerate into a "
                         "temp file and get a deterministic document.")
    ap.add_argument("--readme", default=None,
                    help="default README.md, but only when --out is also "
                         "left at its default; see the sibling-document note "
                         "below.")
    ap.add_argument("--weekend", default=None,
                    help="default WEEKEND.md. Its HEADLINE block is generated "
                         "from the same run JSONs as RESULTS.md, so the file "
                         "a reader actually opens cannot drift from them.")
    ap.add_argument("--interim", action="store_true",
                    help="write runs/interim_report.md instead of RESULTS.md; "
                         "implied automatically while a benchmark is running")
    args = ap.parse_args()

    # ------------------------------------------- do not commit a mid-run doc
    # A benchmark in flight has started attempts with no record yet, so the
    # ledger cannot reconcile and clause 3 reads False -- a *false negative*,
    # and one that would then be committed as the repository's verdict. The
    # symmetric error to a PASS off a screen, so it gets the symmetric
    # treatment: while a run is alive the document goes to an untracked
    # interim path and RESULTS.md keeps describing the last finished
    # measurement. Regenerate after the job exits.
    explicit_out = args.out is not None
    args.out = args.out or str(REPO / "RESULTS.md")
    # An explicit --out means the caller is writing a document somewhere else
    # -- a test fixture, a probe, a temp dir -- and must not also rewrite the
    # repository's own sibling documents. This is not hypothetical: on
    # 2026-09-10 `tests/test_report.py` ran main() over a temp registry of
    # five fixture tasks with an explicit --out and no --weekend, and wrote
    # rows for datasets "d0".."d4" with [not measured] accuracies into the
    # real WEEKEND.md headline. Fixture numbers in a shipped document is the
    # exact failure this repository exists to prevent, arriving through the
    # generator meant to prevent it.
    if args.weekend is None:
        args.weekend = "" if explicit_out else str(REPO / "WEEKEND.md")
    if args.readme is None:
        args.readme = "" if explicit_out else str(REPO / "README.md")
    if not args.interim and not explicit_out and benchmark_processes_alive():
        interim = REPO / "runs/interim_report.md"
        print(f"a benchmark is running: writing {interim} and leaving "
              f"{args.out} at the last finished measurement")
        args.out = str(interim)
        args.readme = ""
        args.weekend = ""

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
    clean_seeds = clean_seed_subset(REPO / "runs/protocol_amendments.json")
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
            # The strictest reading gets the same exact test, so that
            # `clause2_under_strictest_baseline` is a called result and not a
            # point-estimate comparison sitting next to a tested one.
            row["escalation_strict"] = escalation_state(
                accs, t["strictest_baseline_value"],
                base.get("run_protocol") or {})
            # The chronology-clean sub-reading. See the block comment above
            # `clean_seed_subset()`.
            if clean_seeds:
                sub = [r["accuracy_pooled"] for r in runs
                       if r.get("random_state") in clean_seeds]
                row["clean_subset"] = {
                    "seeds": sorted(clean_seeds),
                    "n_present": len(sub),
                    "test": exact_sign_test_above(
                        sub, 0.95 * t["median_run"]) if sub else None,
                    "test_strict": exact_sign_test_above(
                        sub, 0.95 * t["strictest_baseline_value"])
                    if sub else None,
                }
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
                    f"runs missing a required field: "
                    f"{len(v['runs_missing_a_required_field'])}; "
                    f"runs missing an agent digest: "
                    f"{len(v['runs_missing_an_agent_digest'])}; "
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

    # ------------------------------------------- the gate, one row per task
    test_rows = []
    for row in rows:
        e, es = row.get("escalation"), row.get("escalation_strict")
        if not e:
            test_rows.append([row["task_id"], row["name"], 0, NM, NM, NM,
                              NM, NM, NM, NM])
            continue
        t_, ts = e["exact_test"], (es or {}).get("exact_test") or {}
        test_rows.append([
            row["task_id"], row["name"], e["n_seeds"], fmt(e["threshold"]),
            f"{t_['k']}/{t_['n']}", f"{t_['p']:.4f}",
            "CALLED" if e["called"] else
            ("escalation pending" if e["escalation_required"] else "not called"),
            f"{ts.get('k', NM)}/{ts.get('n', NM)}",
            f"{ts['p']:.4f}" if ts else NM,
            "CALLED" if (es or {}).get("called") else "not called"])
    doc += [
        "## The gate: the pre-registered exact test, per task", "",
        "A task is **called** only when the one-sided exact sign test of "
        "`H0: median over seeds <= 0.95 x baseline` rejects at alpha = "
        f"{ALPHA}. "
        f"The p-value floor is `1/2^n`, so a {len(base.get('run_protocol', {}).get('seeds_screen') or [])}"
        "-seed screen cannot call a task in either direction and the status "
        "cannot read PASS off one. This gate replaced "
        "`margin > observed seed range` on 2026-09-10: that condition is "
        "*easier* to satisfy the fewer seeds you run (E[range] is 1.69 sigma "
        "at n=3 against 2.85 sigma at n=8, and it sits in the denominator), "
        "so it skipped the test on all five tasks of the 3-seed screen and "
        "passed the clause on point estimates. The margin reading is kept in "
        "the spread table as a diagnostic.", "",
        table(test_rows, ["task", "dataset", "seeds", "threshold (primary)",
                          "k/n above", "p", "verdict (primary)",
                          "k/n above (strictest)", "p (strictest)",
                          "verdict (strictest)"]), ""]

    # --------------------------------------- the chronology-clean sub-reading
    # Measured, not typed. An earlier version of the paragraph below carried
    # "0.0013-0.0077" as literal text; those were the 3-seed spreads and they
    # were already stale when the 6th seed landed. Any number in generated
    # prose has to come from the runs like every number in a table does.
    _spreads = [e["seed_range"] for r in rows
                if (e := r.get("escalation")) and e.get("seed_range")]
    _margins = [e["margin_to_threshold"] for r in rows
                if (e := r.get("escalation"))]
    if clean_seeds and any(r.get("clean_subset") for r in rows):
        crows = []
        for row in rows:
            cs = row.get("clean_subset")
            if not cs:
                continue
            t_, ts = cs["test"], cs["test_strict"]
            crows.append([
                row["task_id"], row["name"], cs["n_present"],
                f"{t_['k']}/{t_['n']}" if t_ else NM,
                f"{t_['p']:.5f}" if t_ else NM,
                "CALLED" if (t_ and t_["reject_h0"]) else "not called",
                f"{ts['k']}/{ts['n']}" if ts else NM,
                f"{ts['p']:.5f}" if ts else NM,
                "CALLED" if (ts and ts["reject_h0"]) else "not called"])
        doc += [
            "## The reading that owes nothing to the pre-inspected seeds", "",
            "The gate was amended after the 3-seed screen had been read "
            "(`runs/protocol_amendments.json`, amendments 1-3). All three are "
            "strictly stricter and #1 took the status from `PASS` to "
            "`RUNNING` on identical run data, so the usual objection to a "
            "post-hoc rule — that it was tuned to produce a pass — does not "
            "apply. But **seeds "
            + ", ".join(str(s_) for s_ in sorted(
                set(range(8)) - set(clean_seeds)))
            + " had been inspected** when the rule changed, and a reader is "
            "entitled to a reading that owes nothing to them.", "",
            "Below is the same exact test restricted to seeds **"
            + ", ".join(str(s_) for s_ in sorted(clean_seeds))
            + "** — the seeds that had not been run when the amendment was "
            "made, named *in* the amendment, so this is a fixed pre-specified "
            "subset and not one chosen after seeing outcomes. It is reported "
            "whichever way it comes out. At n=5, 5 of 5 above the line gives "
            f"p = 1/32 = {1/32:.5f}, which clears alpha = {ALPHA}, so this "
            "subset can reach a verdict on its own.", "",
            table(crows, ["task", "dataset", "clean seeds present",
                          "k/n above", "p", "verdict (primary)",
                          "k/n above (strictest)", "p (strictest)",
                          "verdict (strictest)"]), "",
            "This is also why no further disjoint seed set was run. More "
            "seeds would shrink seed noise, which across the five tasks is "
            f"{fmt(min(_spreads))}–{fmt(max(_spreads))} against margins to "
            f"the line of {fmt(min(_margins))}–{fmt(max(_margins))}; they "
            "would do nothing about the uncertainty that actually binds, "
            "which is that each task is **one** fixed dataset with one fixed "
            "set of folds."
            if _spreads and _margins else
            "This is also why no further disjoint seed set was run: more "
            "seeds shrink seed noise, which is already far smaller than the "
            "margins, and do nothing about the uncertainty that binds -- each "
            "task is one fixed dataset with one fixed set of folds.",
            ""]

    # ------------------------------- the joint event, one row per seed
    joint = joint_seed_event(bench, sel, "median_run")
    joint_strict = joint_seed_event(bench, sel, "strictest_baseline_value")
    if joint["n_seeds_seen"]:
        jrows = []
        for e, es in zip(joint["per_seed"], joint_strict["per_seed"]):
            missed = [str(t["task_id"]) for t in e["tasks"]
                      if t["cleared"] is False]
            unrun = [str(t["task_id"]) for t in e["tasks"]
                     if t["cleared"] is None]
            def verdict_cell(ev):
                return (NM if ev["all_five_cleared"] is None
                        else ("yes" if ev["all_five_cleared"] else "no"))
            jrows.append([e["seed"], verdict_cell(e), verdict_cell(es),
                          ",".join(missed) or "-",
                          ",".join(unrun) or "-"])
        jt = joint["exact_test_on_the_joint_event"]
        doc += [
            "## Additional reading: does one unattended run clear all five?",
            "",
            "The clause-2 gate above is per task, which is what the protocol "
            "registered and is the right test for it (requiring all five to "
            "reject is an intersection-union test, so five tasks at "
            f"alpha = {ALPHA} need no multiplicity correction). But a "
            "per-task "
            "verdict does not say that a *single* autonomous run gets all "
            "five: five tasks each failing on a different seed would pass "
            "every per-task test and never once produce a clean sweep. So "
            "this table asks the question a reader of "
            "\"end-to-end 무개입\" actually has, and it is reported "
            "beside the gate rather than instead of it.", "",
            table(jrows, ["seed", "cleared all five (median_run)",
                          "cleared all five (strictest)",
                          "tasks below the line", "tasks not yet run"]), "",
            f"**{joint['n_seeds_sweeping_all_five']} of "
            f"{joint['n_seeds_complete']} complete seeds** cleared all five "
            f"against `median_run` "
            f"({joint_strict['n_seeds_sweeping_all_five']} of "
            f"{joint_strict['n_seeds_complete']} against each task's "
            f"strictest reading)"
            + (f"; {joint['n_seeds_incomplete']} seed(s) have tasks still "
               "unrun and are excluded rather than counted as failures"
               if joint["n_seeds_incomplete"] else "")
            + ". Exact sign test on the joint event: "
            + (f"k={jt['k']}/{jt['n']}, p={jt['p']:.4f}." if jt else NM), "",
            "**What eight seeds do and do not measure.** They re-draw the "
            "agent's own randomness — inner-CV shuffle, selection subsample, "
            "random search — on the *same* examples and the same outer folds. "
            "So they estimate algorithmic variance conditional on this data, "
            "not uncertainty about new data, and the p-values above should be "
            "read that way (codex, 2026-09-10). The 80 fold scores are **not** "
            "treated as 80 independent observations: overlapping CV training "
            "sets are dependent and pooling them as independent would "
            "understate the variance (Bengio & Grandvalet, JMLR 2004).", "",
            "`primary_pass` in the table above is computed from the seed "
            "*mean* while the gate tests the *median*. Both are required. "
            "That is an extra empirical guardrail rather than a second "
            "hypothesis test, and it can only make the clause harder: seven "
            "slightly-clearing seeds and one disastrous one can reject the "
            "median null and still fail the mean check.", ""]

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
    rp = Path(args.readme) if args.readme else None
    touched_readme = bool(rp and rp.is_file())
    if touched_readme:
        txt = rp.read_text()
        new = re.sub(r"<!-- BASELINES:BEGIN -->.*?<!-- BASELINES:END -->",
                     "<!-- BASELINES:BEGIN -->\n" + bl_block +
                     "<!-- BASELINES:END -->", txt, flags=re.S)
        rp.write_text(new)

    # ------------------------------------------------- WEEKEND.md headline
    # WEEKEND.md is the file that actually gets read on Monday, which makes it
    # the file most likely to end up carrying a hand-typed number that no
    # longer matches the runs. So its headline is generated between markers by
    # the same code that writes RESULTS.md, and the prose around it is the only
    # part a human writes.
    wp = Path(args.weekend) if args.weekend else None
    touched_weekend = False
    if wp and wp.is_file() and "<!-- HEADLINE:BEGIN -->" in wp.read_text():
        jt = joint["exact_test_on_the_joint_event"]
        head = "\n".join([
            f"**Status: {v['status']}**  ",
            f"*(generated by `scripts/report.py` from `runs/*.json`; "
            f"nothing in this block is hand-typed)*", "",
            table([[c.split("_", 1)[0], c.split("_", 1)[1].replace("_", " "),
                    str(val)] for c, val in v["clauses"].items()],
                  ["clause", "what", "met"]), "",
            table(res_rows, [
                "task", "dataset", "seeds", "ours (pooled)", "median_run",
                "rel gap", "primary (>=0.95x)", "rel 2-sided", "abs 2-sided",
                "median_flow", "vs flow", "strictest reading",
                "strictest value", "vs strictest", "families chosen"]), "",
            "Exact sign test per task, "
            f"H0: median <= 0.95 x baseline (alpha={ALPHA}):", "",
            table(test_rows, ["task", "dataset", "seeds",
                              "threshold (primary)", "k/n above", "p",
                              "verdict (primary)", "k/n above (strictest)",
                              "p (strictest)", "verdict (strictest)"]), "",
            f"One unattended run clearing all five: "
            f"**{joint['n_seeds_sweeping_all_five']} of "
            f"{joint['n_seeds_complete']} complete seeds** "
            f"({joint_strict['n_seeds_sweeping_all_five']} of "
            f"{joint_strict['n_seeds_complete']} against each task's "
            f"strictest reading)"
            + (f", {joint['n_seeds_incomplete']} seed(s) still incomplete and "
               "excluded" if joint["n_seeds_incomplete"] else "")
            + (f"; joint exact test k={jt['k']}/{jt['n']}, p={jt['p']:.4f}."
               if jt else "."), ""])
        wp.write_text(re.sub(
            r"<!-- HEADLINE:BEGIN -->.*?<!-- HEADLINE:END -->",
            "<!-- HEADLINE:BEGIN -->\n" + head + "<!-- HEADLINE:END -->",
            wp.read_text(), flags=re.S))
        touched_weekend = True

    print(f"wrote {args.out}"
          + (" and refreshed the README block" if touched_readme else "")
          + (" and the WEEKEND.md headline" if touched_weekend else ""))
    print(f"status: {v['status']}")
    for k, val in v["clauses"].items():
        print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
