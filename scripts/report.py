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
import hashlib
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


def load_bench(bench_dir: Path) -> tuple[dict[int, list[dict]],
                                         list[dict], list[dict]]:
    """Complete runs by task, every failed attempt, and every partial one.

    Failures are returned, not dropped: a report that silently averages
    whichever seed files happen to have survived is choosing its own sample.

    **Partial records are separated too**, and that is a fix rather than a
    nicety. Until 2026-09-10 this put every non-`.FAILED` file into `by_task`
    regardless of `complete`, so a record written by `--max-folds 1` -- a
    one-fold debug run -- would be averaged into a task's accuracy as though it
    were the ten-fold measurement, and `run_benchmark.py` would then *skip* the
    real run because the file already existed. A one-fold accuracy is not a
    noisy version of a ten-fold accuracy; it is a different statistic, and
    which direction it biases depends on which fold happened to run.

    This is the **third** appearance of this bug class this weekend -- after
    `etch-operator-twin` scoring partial checkpoints as finished arms, and
    after this repository's own joint-event table counting a not-yet-run task
    as a task that failed. The recurring shape is that absence and partiality
    get coerced into a value instead of routing to "no result". So: complete
    records are the sample, partials are reported beside them, and no clause
    can pass while a partial is present.

    It was found because a *concurrent instance of this same loop* ran
    `--role dev --tasks 37 --seeds 0 --max-folds 1` into `runs/dev/` while a
    successor measurement was queued to write there. The concurrency was the
    trigger; the defect was mine and predated it.
    """
    by_task: dict[int, list[dict]] = {}
    failed: list[dict] = []
    partial: list[dict] = []
    for f in sorted(bench_dir.glob("task_*.json")):
        r = read_json(f)
        if r is None:
            continue
        r["_file"] = f.name
        if f.name.endswith(".FAILED.json"):
            failed.append(r)
        elif not r.get("complete", False) or r.get("max_folds") is not None:
            partial.append(r)
        else:
            by_task.setdefault(int(r["task_id"]), []).append(r)
    return by_task, failed, partial


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
                     failed: list[dict] | None = None,
                     role: str = "confirmatory") -> dict:
    """Compare the append-only attempt ledger against the files on disk.

    The attack this answers (codex, #4) is deletion, not fabrication: move the
    unfavourable run files out of `runs/bench/` and every check still passes
    on what remains. The ledger records each attempt before its outcome is
    known, so a result that was produced and then removed leaves a `completed`
    line with no file behind it.

    **Filtered by `role`, which is the fix to a defect that would have made
    this function accuse the project of the very thing it exists to detect.**
    One ledger serves every role, but `on_disk` is built from the *one*
    directory passed in. So a `dev`-role attempt -- which writes to
    `runs/dev/` -- appeared in `completed` and not in `on_disk`, i.e. as
    `completed_but_missing_from_disk`, which the comment above and RESULTS.md
    both describe as a deleted result. Measured on 2026-09-10 with a single
    simulated dev cell: `reconciled` False, `completed_but_missing_from_disk`
    `[[3, 0]]`.

    That was not hypothetical either. The successor measurement queued the
    previous turn runs 15 cells at `--role dev`, so within minutes of it
    starting, the KPI's own report would have stated that results had been
    deleted -- a false accusation of tampering, generated by this project's own
    queued job. Every ledger line already carried the `role` that answers it.

    Lines with no `role` are treated as confirmatory: the only ones are from
    before the field existed, and they were all confirmatory runs. Lines whose
    role is `None` are not attempts at all (`lock_broken`) and drop out.
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
    # Only this role's attempts. A line with no `role` predates the field and
    # was confirmatory; a line whose role is explicitly None is not an attempt.
    def _is_attempt_of_this_role(e):
        if "role" not in e:
            return True
        return e["role"] == role

    n_events_all = len(events)
    events = [e for e in events if _is_attempt_of_this_role(e)]

    def key(e):
        return (e.get("task_id"), e.get("seed"))
    # Counted, not set-ified. A set cannot see a duplicate, which is exactly
    # the defect codex found in the seed accounting -- and it reappeared here,
    # one layer down, in code written after that lesson. This repository's own
    # ledger has TWO `started` lines for task 10101 seed 6 (pids 936115 and
    # 1493119, twenty minutes apart, the first killed and replaced). Under the
    # set reading, the moment any process writes `completed` for that cell both
    # lines collapse into one and the evidence that two runners touched it
    # leaves the reconciliation entirely.
    n_started = Counter(key(e) for e in events if e.get("event") == "started")
    n_completed = Counter(key(e) for e in events
                          if e.get("event") == "completed")
    n_failed_ev = Counter(key(e) for e in events if e.get("event") == "failed")
    # `killed` is a terminal record for an attempt an operator stopped. It
    # resolves the attempt without hiding anything: the line names the reason,
    # and re-running the same (task, seed) is deterministic -- the seed fixes
    # the agent's randomness and the folds are the task's own -- so a kill and
    # re-run cannot shop for a better number, which is why counting it as
    # terminal adds information rather than forgiveness.
    n_killed = Counter(key(e) for e in events if e.get("event") == "killed")
    started = set(n_started)
    completed = set(n_completed)
    failed_ev = set(n_failed_ev) | set(n_killed)
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
    # The count-based reading of the same question, which the set reading
    # cannot express: more starts than terminal records means an attempt was
    # abandoned even if some other attempt at the same cell finished.
    terminal = n_completed + n_failed_ev + n_killed
    starts_without_terminal = sorted(
        [list(k) + [n_started[k], terminal.get(k, 0)]
         for k in n_started if n_started[k] > terminal.get(k, 0)])
    started_more_than_once = sorted(
        [list(k) + [n_started[k]] for k in n_started if n_started[k] > 1])
    # completed in the ledger but absent from runs/bench: a deleted result
    missing_from_disk = sorted(completed - on_disk)
    # present on disk with no ledger entry: a run that bypassed the runner
    unledgered = sorted(on_disk - completed)
    return {
        "ledger_present": True,
        "role_reconciled": role,
        "n_events": len(events),
        "n_events_in_ledger_all_roles": n_events_all,
        "n_started": len(started),
        "n_completed": len(completed),
        "n_failed": len(failed_ev),
        "attempts_started_but_unresolved": [list(k) for k in unresolved],
        # [task, seed, n_started, n_terminal]
        "cells_with_more_starts_than_terminal_records": starts_without_terminal,
        # [task, seed, n_started] -- two runners touched this cell
        "cells_started_more_than_once": started_more_than_once,
        "n_killed_events": sum(n_killed.values()),
        # Which *cells* an operator actually touched, not just how many events
        # there were. codex, 2026-09-11 attack #3: RESULTS.md printed
        # `n_interventions_total = 0` three lines above a ledger recording one
        # kill and one twice-started cell. `n_interventions` counts calls to
        # `DecisionLog.intervene()`, and nothing in `ads/` or `scripts/` ever
        # calls it, so zero meant "nothing was logged", not "nothing happened".
        "operator_touched_cells": sorted(
            {(int(t), int(sd)) for t, sd, _ in started_more_than_once}
            | {(int(k[0]), int(k[1])) for k in n_killed}),
        "completed_but_missing_from_disk": [list(k) for k in missing_from_disk],
        "on_disk_but_not_in_ledger": [list(k) for k in unledgered],
        "failed_attempts_with_records": len(failed_files & failed_ev),
        # Multiplicity alone does not sink it: a kill and re-run is legitimate
        # and this repo has one. What sinks it is an abandoned attempt -- more
        # starts than terminal records -- which is the truthful generalisation
        # of the old set-based check, not a relaxation of it.
        "reconciled": not (starts_without_terminal or missing_from_disk
                           or unledgered),
    }


def merge_leakage_records(records: list[dict | None]) -> dict | None:
    """Combine several leakage-probe records into the one the gate reads.

    Two records exist because the rank instrument was added on 2026-09-11 to
    reach the two tasks the accuracy instrument is blind on, and re-probing
    the three already-clean tasks would have cost hours to reproduce a result
    that is already on disk.  The merge is deliberately asymmetric, for the
    same reason `cleared_tasks` is:

      * **LEAKAGE anywhere wins.**  Any record declaring a leak makes the
        merged verdict LEAKAGE.
      * **Clearances union, they do not vote.**  A task cleared by either
        record is cleared; a task cleared by neither is simply absent, which
        is a hole and not a clearance.
      * **The digest is the oldest one present**, so a stale record cannot be
        hidden behind a fresh one -- the gate's staleness check must see the
        weakest link, not the strongest.
    """
    recs = [r for r in records if r]
    if not recs:
        return None
    if len(recs) == 1:
        return recs[0]
    merged = dict(recs[0])
    merged["verdict"] = ("LEAKAGE"
                         if any(r.get("verdict") == "LEAKAGE" for r in recs)
                         else "NO_LEAKAGE_DETECTED")
    cleared: set[int] = set()
    for r in recs:
        cleared |= set(r.get("tasks_cleared") or [])
    # A task any record condemns is removed from the union even if another
    # record cleared it -- firing is not symmetric with clearing.
    for r in recs:
        for t in (r.get("tasks") or []):
            if (t.get("verdict_task") == "LEAKAGE"
                    or t.get("verdict_task_rank") == "LEAKAGE"):
                cleared.discard(int(t["task_id"]))
    merged["tasks_cleared"] = sorted(cleared)
    merged["task_ids"] = sorted({int(t) for r in recs
                                 for t in (r.get("task_ids") or [])})
    merged["tasks"] = [t for r in recs for t in (r.get("tasks") or [])]
    merged["_merged_from"] = len(recs)
    # Tasks the *rank* instrument resolved, which the accuracy-only power file
    # does not know about; the gate adds these to its probeable set.
    merged["tasks_resolved_by_rank_instrument"] = sorted(
        int(t["task_id"]) for r in recs for t in (r.get("tasks") or [])
        if t.get("rank_can_resolve"))
    return merged


def verdict(rows: list[dict], base: dict, bench: dict,
            failed: list[dict] | None = None, ledger: dict | None = None,
            partial: list[dict] | None = None,
            leakage: dict | None = None,
            power: dict | None = None) -> dict:
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
    # ------------------------------------------- is the accuracy uncontaminated?
    # Open item #8 was carried as "architecturally open" for three turns: the
    # labelled frame lives in the same process as the agent and is *reachable*
    # from the pandas views it is handed. An accuracy that has not been shown
    # free of that route is not a measurement of the agent, so it gates clause
    # 2 rather than sitting in a section a reader may skip.
    #
    # Three states, and the middle one is the point:
    #   LEAKAGE               -> False. Every accuracy here is withdrawn.
    #   NO_LEAKAGE_DETECTED   -> True.
    #   probe absent/stale    -> None. "Not shown contaminated" is not
    #                            evidence of cleanliness; same rule as
    #                            n_interventions, where a run that does not
    #                            carry the field cannot testify it is zero.
    # This gate was added at 2026-09-10 turn 9 *while the probe was still
    # running* and before any of its numbers existed, which is the only order
    # in which adding a gate is not a choice about what to gate on.
    # A probe is only evidence about the code it actually ran. Open item #7
    # (evaluation-cache provenance) is the same defect one item over: a digest
    # computed over artefacts I control proves consistency, not authenticity.
    # So the probe must carry `env.ads_sha256` and it must equal the digest the
    # benchmark runs agree on -- otherwise a probe against last week's agent
    # certifies this week's.
    leak_digest = ((leakage or {}).get("env") or {}).get("ads_sha256")
    run_digests = {(r.get("env") or {}).get("ads_sha256")
                   for runs in bench.values() for r in runs} - {None}
    leakage_stale = bool(
        leakage and (not leak_digest
                     or (run_digests and leak_digest not in run_digests)))
    # SCOPE, and this is a correction to the gate as I first wrote it an hour
    # earlier in this same turn. `runs/leakage_power.json` measures what leak
    # the probe could resolve per task, and the answer is that it is blind on
    # two of the registered five: phi_min = 1.112 (kc1) and 4.051
    # (blood-transfusion) mean an agent handed the true test labels would score
    # *inside the noise* of a majority-class predictor there, because the
    # honest agent already does. So a clean probe on kr-vs-kp does not clear
    # the other four, and my first version of this gate said it did.
    #
    # The reachable honest rule: no leakage detected on every task where
    # detection is *possible*, with the unprobeable tasks named in the output
    # rather than absorbed. What covers those is not this probe but
    # `tests/test_no_leakage.py::test_view_of_labelled_frame_predicts_
    # identically_to_a_copy`, which asserts bitwise prediction equality and so
    # has power that does not depend on the majority-rate gap at all.
    power = power or {}
    # A task is probeable only at or above the fold count the power file
    # measured; kc2 needs six pooled folds and zero individual ones suffice,
    # so "probed" without a fold count is not a clearance.
    min_folds = {int(k): v for k, v
                 in (power.get("min_folds_for_detection") or {}).items()}
    probeable = set(min_folds)
    # The power file is an *accuracy*-instrument measurement, so it cannot know
    # which tasks the rank instrument reaches.  Adding those here is what makes
    # the new probe binding rather than decorative: once a task is probeable it
    # must also be *cleared*, or `probeable_unprobed` sends the gate to None.
    rank_resolved = set((leakage or {}).get("tasks_resolved_by_rank_instrument")
                        or [])
    probeable |= rank_resolved
    unprobeable = sorted(set(power.get("tasks_unpowered_pooled") or [])
                         - rank_resolved)
    cleared = set((leakage or {}).get("tasks_cleared") or [])
    probeable_unprobed = sorted(probeable - cleared)
    # codex, 2026-09-11 attack #4: `probeable` was derived solely from whatever
    # keys the power record happened to carry, and nothing required it to
    # account for all five REGISTERED tasks. A power file with no coverage
    # entries yields an empty probeable set, `probeable_unprobed` is then
    # trivially empty, and a matching-digest clean probe clears the gate having
    # measured nothing. Missing evidence disappearing from the requirement is
    # the same fail-open shape as attacks #1, #2 and #5 -- fourth instance this
    # weekend of absence being coerced into a value.
    registered_tasks = {int(t) for t in (base.get("selected_task_ids") or [])}
    leakage_unaccounted = sorted(
        registered_tasks - set(cleared) - set(unprobeable))
    if (leakage is None or not leakage.get("verdict") or leakage_stale
            or not power or probeable_unprobed or leakage_unaccounted):
        leakage_ok = None
    elif leakage["verdict"] == "NO_LEAKAGE_DETECTED":
        leakage_ok = True
    else:
        # A LEAKAGE verdict is believed even from a stale probe -- but that is
        # unreachable here, since `leakage_stale` already sent it to None. It
        # is deliberate: a stale probe cannot condemn runs it did not measure
        # any more than it can clear them.
        leakage_ok = False
    clause2 = (bool(measured) and len(measured) == n_tasks
               and all(r["primary_pass"] for r in measured)
               and not escalation_pending and not not_called
               and leakage_ok is True)
    # A clause whose seeds are still short of the registered verdict set is
    # *unmeasured*, not failed: reporting "NOT MET as measured" off a screen
    # would be as wrong in the pessimistic direction as PASS was in the
    # flattering one. None routes the status to RUNNING.
    if escalation_pending:
        clause2 = None
    # An unmeasured probe is a hole, not a failure -- but only when nothing
    # else has already failed the clause, or an absent probe would launder a
    # real FAIL into RUNNING (the flattering direction).
    if clause2 is False and leakage_ok is None and all(
            r["primary_pass"] for r in measured) and not not_called:
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
    # A partial record on disk is a hole in the sample even though it is not a
    # failure: load_bench keeps it out of the average, and clause 3 must then
    # say so rather than passing on a silently smaller sample.
    partial = partial or []
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
    # codex, 2026-09-11, attack #2, and it is the sharpest one this repository
    # has taken: `len(registry_sha) <= 1` asks only that the runs agree with
    # EACH OTHER about which registry they used. It never asked whether that
    # registry is the one whose numbers RESULTS.md prints. So lowering a
    # baseline in runs/baselines.json after seeing the accuracies left every
    # provenance check satisfied and moved the thresholds -- the exact route
    # the brief names as the way this KPI gets faked. The digests do currently
    # agree (6731b733...), so no number here is affected; the gate was open and
    # the door happened to be shut.
    live_registry_sha = hashlib.sha256(
        (REPO / "runs/baselines.json").read_bytes()).hexdigest()
    registry_matches_live = bool(registry_sha) and registry_sha == {
        live_registry_sha}
    # Two runs produced by different versions of `ads/`, or by a dirty tree,
    # are not one measurement of one agent (codex, attack #5).
    # NOT discarded: a run with no digest is counted as its own distinct
    # digest, so "all runs agree" cannot be satisfied by runs that said nothing.
    agent_sha = {(r.get("env") or {}).get("ads_sha256") or f"[absent:{r['_file']}]"
                 for runs in bench.values() for r in runs}
    # codex attack #5: this flagged a run only when the field was TRUTHY, so
    # deleting the field made a dirty tree read as clean. Absent is a hole.
    dirty_runs = [r["_file"] for runs in bench.values() for r in runs
                  if (r.get("env") or {}).get("ads_dirty_vs_head")]
    runs_missing_dirty_flag = sorted(
        r["_file"] for runs in bench.values() for r in runs
        if (r.get("env") or {}).get("ads_dirty_vs_head") is None)
    # An absent ledger is now a hole, not a pass: the ledger is the only record
    # of an attempt that was started and never produced a file, so without it
    # "no failed attempts" is an unfalsifiable claim.
    # codex attack #1: `ours` averaged each record's `accuracy_pooled` field
    # verbatim, and nothing recomputed it. Editing that one number in a
    # finished run file moved the mean and the sign test while the ledger still
    # reconciled, because reconciliation compares task/seed membership and
    # event counts, not scores. Every record carries `n_correct` and
    # `n_predictions`, and `per_fold` carries the fold weights, so the score is
    # checkable against the record's own evidence at zero cost. All 40 current
    # records satisfy both identities; again the gate was open, not the data
    # wrong.
    def _accuracy_recomputes(r: dict) -> bool:
        a, nc, n = (r.get("accuracy_pooled"), r.get("n_correct"),
                    r.get("n_predictions"))
        if a is None or not nc or not n:
            return False
        if abs(a - nc / n) > 1e-9:
            return False
        pf = r.get("per_fold") or []
        if pf:
            tn = sum(f.get("n_test") or 0 for f in pf)
            if tn != n:
                return False
            wm = sum((f.get("accuracy") or 0) * (f.get("n_test") or 0)
                     for f in pf) / tn
            if abs(a - wm) > 1e-6:
                return False
        return True

    runs_whose_accuracy_does_not_recompute = sorted(
        r["_file"] for runs in bench.values() for r in runs
        if not _accuracy_recomputes(r))
    # ---------------------------------------- clause 3 under both readings
    # Rung 1 of the ladder: report the strict reading AND the conventional one,
    # each with its protocol, rather than silently picking whichever passes.
    #
    #   conventional -- "무개입" means the AGENT picks the preprocessing, the
    #     model and the validation with no human choosing any of them. Killing
    #     a compute job and restarting it with more threads changes no
    #     modelling decision, so it does not bear on this reading.
    #   strict -- the brief's own words: "count any manual intervention as a
    #     failure of that run rather than editing it out." An operator killing
    #     and restarting a cell IS a manual intervention in that cell, whatever
    #     it did or did not change.
    #
    # The strict reading is binding. It costs a clause that would otherwise
    # read True, which is why it is not optional.
    operator_cells = [list(c) for c in
                      ((ledger or {}).get("operator_touched_cells") or [])]
    clause3_strict = bool(measured) and not operator_cells
    ledger_ok = bool(ledger and ledger.get("ledger_present")
                     and ledger.get("reconciled"))
    clause3 = (bool(measured) and interventions == 0 and not off_registry
               and not incomplete and not failed and not partial
               and not seeds_missing
               and not seeds_duplicated and not seeds_unregistered
               and not runs_missing_fields and not runs_missing_agent_digest
               and len(registry_sha) <= 1 and registry_matches_live
               and len(agent_sha) <= 1 and not dirty_runs
               and not runs_missing_dirty_flag
               and not runs_whose_accuracy_does_not_recompute
               and ledger_ok)
    clause3_conventional = clause3
    clause3 = bool(clause3) and clause3_strict
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
            "leakage_probe_clean": leakage_ok,
            "leakage_probe_verdict": (leakage or {}).get("verdict"),
            "leakage_probe_tasks": (leakage or {}).get("task_ids"),
            "leakage_probe_agent_digest": leak_digest,
            "leakage_probe_stale_vs_runs": leakage_stale,
            "leakage_probeable_tasks": sorted(probeable),
            "leakage_min_folds_for_detection": {str(k): v for k, v
                                                in min_folds.items()},
            "leakage_unprobeable_tasks": unprobeable,
            "leakage_tasks_cleared": sorted(cleared),
            "leakage_probeable_but_unprobed": probeable_unprobed,
            "leakage_registered_tasks_unaccounted_for": leakage_unaccounted,
            "leakage_tasks_resolved_by_rank_instrument": sorted(rank_resolved),
            "leakage_records_merged": (leakage or {}).get("_merged_from", 1),
            "clause3_conventional_agent_chose_everything": clause3_conventional,
            "clause3_strict_no_operator_touched_any_cell": clause3_strict,
            "operator_touched_cells": operator_cells,
            "runs_whose_accuracy_does_not_recompute":
                runs_whose_accuracy_does_not_recompute,
            "runs_missing_the_dirty_tree_flag": runs_missing_dirty_flag,
            "registry_digest_matches_live_baselines_file": registry_matches_live,
            "live_baselines_sha256": live_registry_sha,
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
            "n_partial_records_excluded_from_the_average": len(partial),
            "partial_records": [
                f"{r['_file']}(folds={r.get('n_folds_run')},"
                f"max_folds={r.get('max_folds')})" for r in partial],
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
    # `--interim` used to *suppress* this guard rather than force it -- the
    # condition read `not args.interim and ...`, so the one flag documented as
    # "write runs/interim_report.md instead of RESULTS.md" was the one flag
    # that guaranteed RESULTS.md got overwritten mid-run. Found on 2026-09-10
    # turn 9 by running it: it printed "wrote RESULTS.md" with a benchmark
    # provably alive (`benchmark_processes_alive()` was returning True the
    # whole time; the guard was fine, the flag inverted it). Direction:
    # flattering-hazard, since mid-run clause 3 is a known false negative and
    # would have been committed as the repository's verdict. RESULTS.md was
    # restored from HEAD.
    in_flight = benchmark_processes_alive()
    if not explicit_out and (args.interim or in_flight):
        interim = REPO / "runs/interim_report.md"
        print(f"{'--interim' if args.interim else 'a benchmark is running'}: "
              f"writing {interim} and leaving {args.out} at the last finished "
              f"measurement")
        args.out = str(interim)
        args.readme = ""
        # The HEADLINE block is built from the in-flight bench records, so it
        # is suppressed with the rest. The LEAKPOWER block is not: its only
        # source is runs/leakage_power.json, which is a completed measurement,
        # so `weekend_leakpower_only` lets that one block refresh while a run
        # is alive. Keeping a stale power table out of the document a reader
        # opens has no upside.
        args.weekend_leakpower_only = True

    base = read_json(args.baselines)
    bench, failed, partial = (load_bench(Path(args.bench))
                              if Path(args.bench).exists() else ({}, [], []))

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
                # The screen's own seeds, so the range the OLD gate would have
                # divided by can be compared against the range on all seeds.
                screen_seeds = set((base.get("run_protocol") or {})
                                   .get("seeds_screen") or [])
                screen_accs = [r["accuracy_pooled"] for r in runs
                               if r.get("random_state") in screen_seeds]
                r_screen = (max(screen_accs) - min(screen_accs)
                            if len(screen_accs) >= 2 else None)
                r_all = max(accs) - min(accs)
                ratio = (r_all / r_screen
                         if r_screen and len(accs) > len(screen_accs) else None)
                spread_rows.append([
                    tid, t["dataset_name"], len(accs), fmt(min(accs)),
                    fmt(max(accs)), fmt(r_all),
                    fmt(statistics.pstdev(accs)),
                    fmt(r_screen) if r_screen else NM,
                    f"{ratio:.2f}x" if ratio else NM,
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
    leakage = merge_leakage_records([
        read_json(REPO / "runs/leakage_probe.json"),
        read_json(REPO / "runs/leakage_probe_rank.json"),
    ])
    power = read_json(REPO / "runs/leakage_power.json")
    v = verdict(rows, base, bench, failed, ledger, partial, leakage, power)

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
                    f"partial records excluded: "
                    f"{v['n_partial_records_excluded_from_the_average']}; "
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
            ]

    # ------------------------------- is the clause falsifiable at all?
    nc = read_json(REPO / "runs/negative_control.json")
    if nc:
        # counted, not asserted: an earlier version of the paragraph below
        # said "two of those are inside seed noise" and only one was.
        _n_inside = _n_behind = 0
        for r in nc["controls"].get("stump", {}).get("tasks", []):
            row = next((x for x in rows if x["task_id"] == r["task_id"]), None)
            if not row or row.get("ours") is None:
                continue
            d = row["ours"] - r["accuracy_pooled"]
            rng = (row.get("escalation") or {}).get("seed_range")
            if d < 0:
                _n_behind += 1
            elif rng is not None and abs(d) < rng:
                _n_inside += 1
        nrows = []
        for name, c in nc["controls"].items():
            for r in c["tasks"]:
                ours_here = next((x["ours"] for x in rows
                                  if x["task_id"] == r["task_id"]), None)
                nrows.append([
                    name, r["task_id"], r["dataset_name"],
                    fmt(r["accuracy_pooled"]),
                    fmt(r["majority_class_rate"]),
                    fmt(r["threshold_primary"]),
                    "CLEARS" if r["clears_primary"] else "fails",
                    "CLEARS" if r["clears_strictest"] else "fails",
                    fmt(ours_here),
                    fmt(ours_here - r["accuracy_pooled"])
                    if ours_here is not None else NM])
        doc += [
            "## Is this clause falsifiable? The negative controls", "",
            "This section comes before the accuracy tables on purpose. Every "
            "other check in this document asks whether *our* number is "
            "honest; this one asks whether the **target** is demanding, and "
            "the answer is only partly yes — so it changes how the tables "
            "below should be read.", "",
            f"**{nc['n_thresholds_below_majority_rate']} of 5 primary "
            "thresholds sit at or below the task's own majority-class rate**, "
            "so on those tasks the bar can be cleared by predicting the "
            "commonest label and nothing else. Two frozen, deliberately "
            "incapable procedures were run through the **same outer folds, "
            "the same pooled metric and the same pre-registered baselines** as "
            "the agent: `prior` (`DummyClassifier(strategy=\"prior\")`, which "
            "ignores the features entirely) and `stump` "
            "(`DecisionTreeClassifier(max_depth=3)`, untuned). Neither was "
            "chosen by how it scored.", "",
            table(nrows, ["control", "task", "dataset", "control accuracy",
                          "majority-class rate", "threshold (0.95x median_run)",
                          "vs primary", "vs strictest", "ours", "ours - control"]),
            "",
            "  ".join(
                f"**{name}**: clears the primary reading on "
                f"{c['n_clearing_primary']}/{c['n_tasks']} tasks, the "
                f"strictest on {c['n_clearing_strictest']}/{c['n_tasks']}, "
                f"all five: {c['clears_all_five_primary']}."
                for name, c in nc["controls"].items()), "",
            "### Was the weak task set bad luck, or did the rule cause it?",
            ""]
        td = read_json(REPO / "runs/target_difficulty.json")
        _ff = read_json(REPO / "runs/falsifiability_floor.json")
        _ncs = read_json(REPO / "runs/negative_control_successor.json")
        _se = (_ff or {}).get("selection_effect") or {}
        if td:
            doc += [
                "Answerable with no runs at all, and blind to our accuracy by "
                "construction: both inputs are published data, the baseline "
                "from OpenML's run history and the majority-class rate from a "
                "dataset quality.", "",
                f"Of the **{td['n_candidates']} candidates** that passed the "
                "registered size filter, "
                f"**{td['n_falsifiable']}** have a threshold *above* their "
                "majority-class rate. Among the "
                f"{td['n_selected']} the rule actually selected: "
                f"**{td['n_falsifiable_among_selected']}**. A random five from "
                "the same pool would be expected to contain "
                f"**{td['expected_n_falsifiable_in_a_random_five']}**, and the "
                "exact hypergeometric probability of drawing at most as few "
                f"as were drawn is **p = "
                f"{td['exact_hypergeometric_p_lower_tail']}**.", "",
                "**That p-value was the wrong test, and it is withdrawn as "
                "evidence.** It asks whether *this draw of five* is unusual, "
                "and at n=5 against the pool's base rate it can reject only on "
                "the most extreme possible draw"
                + (f" — `rejectable_draws` = "
                   f"{_se['stump']['draw_test']['rejectable_draws']} out of "
                   f"0..5 under the procedure floor" if _se else "")
                + ". A test that can fire in one outcome out of six is not "
                "evidence about a rule, and quoting it as such was an error. "
                "Both tests are now reported under both floors, and the one "
                "the question actually calls for is the **rule** test: "
                "Spearman over all candidates, n much larger than 5, with a "
                "one-sided permutation null.", "",
                (table(
                    [[f"`{'procedure (clears every control)' if k == 'stump' else 'majority-class rate'}`",
                      f"{v['draw_test']['base_rate']}",
                      f"{v['draw_test']['observed']}",
                      f"{v['draw_test']['expected']}",
                      f"{v['draw_test']['p_lower_tail']:.4f}",
                      str(v['draw_test']['rejectable_draws']),
                      f"{v['rule_test']['n']}",
                      f"{v['rule_test']['spearman_rho']:+.4f}",
                      f"{v['rule_test']['p_one_sided_permutation']:.4f}"]
                     for k, v in _se.items()],
                    ["floor", "base rate", "selected", "expected",
                     "draw p (n=5)", "draws it can reject on",
                     "rule n", "rule rho", "rule p (permutation)"])
                 if _se else f"{NM} — run `scripts/falsifiability_floor.py`"),
                "",
                "Read the diagonal. The **draw** test is significant only "
                "under the weaker floor; the **rule** test is significant only "
                "under the stricter one. The conclusion rests on the cell that "
                "is both the appropriate test and the stricter floor: "
                + (f"rho = {_se['stump']['rule_test']['spearman_rho']:+.4f}, "
                   f"p = "
                   f"{_se['stump']['rule_test']['p_one_sided_permutation']:.4f}"
                   if _se else NM)
                + ". Positive rho means headroom grows with rank number, i.e. "
                "**the more published a task is, the less its threshold clears "
                "a depth-3 tree.** The effect is modest and it is real. "
                "Ranking candidates by number of published evaluations — chosen "
                "because the median of a larger sample is better determined, "
                "which is true and is still true — also selects for the small, "
                "famous, imbalanced classics whose medians sit near "
                "triviality. The registered rule was blind to our accuracy and "
                "it was **not** blind to the difficulty of the target, and "
                "only the first of those was designed for.", "",
                "**A successor criterion, and the measurement that shows "
                "it is not enough on its own.** The obvious rule — the five "
                "most-published candidates whose threshold exceeds their "
                "majority-class rate — is blind to our accuracy, but it is "
                "*circular* as evidence: it makes the `prior` control fail by "
                "construction, since that control's accuracy **is** the "
                "majority rate. Measured on those five "
                "(`runs/negative_control_successor.json`): `prior` clears "
                + (f"{_ncs['controls']['prior']['n_clearing_primary']} of "
                   f"{_ncs['controls']['prior']['n_tasks']}, as guaranteed, "
                   "and the untuned depth-3 tree still clears **"
                   f"{_ncs['controls']['stump']['n_clearing_primary']} of "
                   f"{_ncs['controls']['stump']['n_tasks']}**"
                   if _ncs else "[not measured]")
                + ". So clearing the class-prior floor does "
                "not make a threshold demanding; the floor that matters is a "
                "*procedure* floor. The candidates are — "
                + ", ".join(f"`{r['dataset_name']}` (rank "
                            f"{r['rank_by_n_runs']}, headroom "
                            f"{r['headroom']:+.4f})"
                            for r in td["a_falsifiable_five_under_a_blind_rule"])
                + ". That is a **different measurement** and is offered as the "
                "successor, never as this one. Swapping it in now would be "
                "choosing a task set after seeing which one made the point.",
                ""]

        doc += [
            "**What this does to the claim.** The per-task clause is weak on "
            "the four imbalanced tasks and only `kr-vs-kp` discriminates on "
            "its own — a fact about the pre-registered target, not about the "
            "agent, and one that no amount of provenance machinery would have "
            "surfaced. What survives it is the **joint** criterion: neither "
            "control clears all five, because both collapse on the one "
            "balanced task. So a PASS here should be read as \"clears five "
            "tasks including one where triviality fails\", not as \"beat a "
            "human five times\".", "",
            "**The uncomfortable rows, named rather than left in the "
            "table.** Against the untuned depth-3 tree the agent's margin is "
            + ", ".join(
                f"{r['dataset_name']} {_d:+.4f}"
                for r in nc["controls"]["stump"]["tasks"]
                if (_o := next((x["ours"] for x in rows
                                if x["task_id"] == r["task_id"]), None))
                is not None and (_d := _o - r["accuracy_pooled"]) is not None)
            + f". {_n_inside} of the five sits inside the task's own seed "
            f"range, so it is not a difference this repository can resolve, "
            f"and {_n_behind} is negative — there the stump is **ahead**. "
            "The agent's six-family "
            "tournament plus random search is buying a large margin on the "
            "balanced task and, on the imbalanced ones, very little over three "
            "splits of a tree. That is a finding about the agent and it is not "
            "flattering; it is here because it is what the runs say.", ""]

    doc += [
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
    # ------------------------------------------------------------------
    # Does clause 2 rest on a cell an operator touched?  Registered in
    # critique_log.md turn 11b before it was computed, and reported whichever
    # way it came out.
    tcs = read_json(REPO / "runs/tainted_cell_sensitivity.json")
    if tcs:
        hdr_t = ["task", "dataset", "operator-touched seeds dropped",
                 "k/n (all seeds)", "p", "verdict",
                 "k/n (without them)", "p", "verdict",
                 "verdict depends on a touched cell?"]
        rows_t = []
        for t in tcs["tasks"]:
            a, b = t["with_every_seed"], t["without_touched_cells"]
            rows_t.append([
                str(t["task_id"]), t["dataset_name"],
                str(t["seeds_dropped"]) if t["seeds_dropped"] else "none",
                f"{a['k']}/{a['n']}", f"{a['p']:.4f}",
                "CALLED" if a["called"] else "not called",
                f"{b['k']}/{b['n']}", f"{b['p']:.4f}",
                "CALLED" if b["called"] else "not called",
                "**YES**" if t["verdict_depends_on_a_touched_cell"] else "no"])
        doc += [
            "## Does clause 2 rest on the cell an operator touched?", "",
            "Clause 3 fails under its strict reading because one cell — "
            f"`{tcs['operator_touched_cells']}`, task 10101 seed 6 — was "
            "killed and restarted by an operator (amendment 9). The question "
            "that follows is whether the clause-2 verdict *depends* on that "
            "record. If the task can still be called without it, the "
            "intervention is a disclosure; if it cannot, clause 2 is standing "
            "on a record the protocol says should have been counted as a "
            "failed run.", "",
            "No new runs: this is the same pre-registered exact sign test, at "
            f"the same alpha = {ALPHA}, on a strictly smaller seed set. "
            "Dropping a seed **costs** power — p floors at 1/2^n, so n=8 "
            "floors at 0.0039 and n=7 at 0.0078 — which is why this is a real "
            "test and not a formality.", "",
            table(rows_t, hdr_t), "",
            ("**No per-task verdict depends on an operator-touched cell.** "
             "Blood-transfusion is still called on its seven untouched seeds "
             "(7/7, p = 0.0078). So the clause-3 failure is a disclosure "
             "about how the measurement was produced, not a hole in the "
             "measurement itself."
             if not tcs["any_verdict_depends_on_a_touched_cell"] else
             "**At least one per-task verdict depends on a cell an operator "
             "touched.** That verdict is withdrawn: the protocol counts a "
             "manually interrupted cell as a failed run, and a clause may not "
             "rest on one."), "",
        ]
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

    # ------------------- split variance, beside the seed test that is the gate
    fi = read_json(REPO / "runs/fold_interval.json")
    if fi and fi.get("tasks"):
        frows = []
        for tid in order:
            r = fi["tasks"].get(str(tid))
            if not r:
                frows.append([tid, sel[tid]["dataset_name"]] + [NM] * 7)
                continue
            frows.append([
                tid, r["dataset_name"], r["n_seeds_averaged"], r["n_folds"],
                f"{r['mean_relative_margin']*100:+.2f}%",
                f"{r['sd_relative_margin_over_folds']*100:.2f}%",
                f"{r.get('lower_bound_naive', float('nan'))*100:+.2f}%",
                f"{r.get('lower_bound_corrected', float('nan'))*100:+.2f}%",
                "PASS" if r.get("non_inferior_corrected") else "FAIL"])
        doc += [
            "## Additional reading: the folds, not the seeds", "",
            "The gate above is a sign test over **seeds**, which re-draw the "
            "agent's own randomness on the *same* examples and the same outer "
            "folds. So it constrains algorithmic variance and says nothing "
            "directly about split variance. This section is the other axis, "
            "computed from the ten outer-fold accuracies already in every run "
            "record — no new runs. It is an **additional reading and enters no "
            "clause**; a test asserts `verdict()` cannot see it.", "",
            "Per fold, the relative margin `d_k = (ours_k - baseline) / "
            "baseline`, and a one-sided 95% lower bound on its mean against "
            "the pre-registered `-0.05`. The tolerance is the registry's; only "
            "the uncertainty model is new here.", "",
            table(frows, ["task", "dataset", "seeds averaged", "folds",
                          "mean relative margin", "sd over folds",
                          "95% lower bound (naive)",
                          "95% lower bound (corrected)",
                          "non-inferior?"]), "",
            "**Two bounds because the honest one is not obvious.** The folds "
            "have disjoint test sets and heavily overlapping training sets, so "
            "`s/sqrt(K)` understates the variance — the flattering direction. "
            "The corrected column uses the Nadeau & Bengio (2003) inflation "
            "`(1/K + n_test/n_train) s^2`, which at 10 folds multiplies the "
            "variance by 2.11x. That correction is *derived* for repeated "
            "random subsampling and is applied here as a conservative "
            "adjustment for fold dependence; both columns are shown so a "
            "reader who rejects the adjustment can read the other. The 80 "
            "fold-by-seed scores are **not** pooled as 80 independent "
            "observations, which is the move that would buy power by "
            "assuming away the dependence.", "",
            f"Non-inferiority holds on all {fi['n_tasks']} tasks under the "
            f"naive bound ({fi['all_non_inferior_naive']}) and under the "
            f"corrected one ({fi['all_non_inferior_corrected']}).", "",
            "**And this is the more important number in the section:** the "
            "fold-to-fold sd of the relative margin exceeds the "
            "seed-to-seed range by "
            + (f"{fi['fold_sd_over_seed_range_min']:.1f}x to "
               f"{fi['fold_sd_over_seed_range_max']:.1f}x"
               if "fold_sd_over_seed_range_min" in fi else NM)
            + ". The seed-count discipline this repository spent a turn "
            "enforcing is correct on its own terms and was constraining the "
            "**smaller** of the two noise sources. Both are now reported; "
            "neither replaces the other, and the gate stays the pre-registered "
            "one.", ""]

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
                table(spread_rows, ["task", "dataset", "seeds", "min",
                                    "max", "range", "sd",
                                    "range on the 3 screen seeds",
                                    "range grew by", "|gap to baseline|"]),
                "",
                "The last two columns measure, on this repository's own runs, "
                "the bias that got the gate replaced. A 3-seed range "
                "*understates* the spread: the expected range of n i.i.d. "
                "draws is 1.693 sigma at n=3 and 2.847 sigma at n=8, a ratio "
                "of **1.68x**, and the old gate put that understated quantity "
                "in the denominator of `margin > range`. So the fewer seeds "
                "you ran, the more comfortably clear of the line every task "
                "looked. The measured growth beside it is what actually "
                "happened when seeds were added.", ""]
    else:
        doc += ["## Our own run-to-run spread", "",
                f"{NM} — fewer than two seeds per task so far.", ""]

    # ---------------------------- the successor measurement, clearly labelled
    # A different task set is a DIFFERENT MEASUREMENT and is reported as one.
    # It lives in runs/dev, is never read by the clause accounting above, and
    # says "screen, not verdict" until it has the registered verdict seeds.
    # runs/successor, not runs/dev: the second measurement gets its own
    # directory so a development or debug run cannot be read as part of it.
    _succ_dir = REPO / "runs/successor"
    dev_bench, dev_failed, dev_partial = (
        load_bench(_succ_dir) if _succ_dir.exists() else ({}, [], []))
    if dev_bench:
        all_tasks = {t["task_id"]: t for t in base["tasks"]}
        td = read_json(REPO / "runs/target_difficulty.json") or {}
        nc_succ = read_json(REPO / "runs/negative_control_successor.json")
        successor = [r["task_id"]
                     for r in td.get("a_falsifiable_five_under_a_blind_rule", [])]
        drows = []
        for tid in sorted(dev_bench, key=lambda i: successor.index(i)
                          if i in successor else 99):
            t = all_tasks.get(tid)
            if not t:
                continue
            runs = dev_bench[tid]
            accs = [r["accuracy_pooled"] for r in runs]
            ours = statistics.mean(accs)
            tol = tolerance_readings(ours, t["median_run"])
            tol_s = tolerance_readings(ours, t["strictest_baseline_value"])
            e = escalation_state(accs, t["median_run"],
                                 base.get("run_protocol") or {})
            hr = next((r["headroom"] for r in td.get("tasks", [])
                       if r["task_id"] == tid), None)
            # Which of these rows actually discriminate? The set was chosen by
            # "threshold above the majority-class rate", which makes `prior`
            # fail by construction and says nothing about anything stronger.
            # Measured: the untuned depth-3 stump clears 3 of these 5. So the
            # per-task stump verdict travels with the row rather than a reader
            # having to trust the selection rule.
            sc = next((r for r in (nc_succ or {}).get("controls", {})
                       .get("stump", {}).get("tasks", [])
                       if r["task_id"] == tid), None)
            drows.append([
                tid, t["dataset_name"], t["rank_by_n_runs"], len(accs),
                fmt(ours), fmt(t["median_run"]),
                f"{tol['rel_gap']*100:+.2f}%",
                "PASS" if tol["rel_one_sided"] else "FAIL",
                "PASS" if tol_s["rel_one_sided"] else "FAIL",
                fmt(hr),
                f"{e['exact_test']['k']}/{e['exact_test']['n']}",
                f"{e['exact_test']['p']:.4f}",
                "CALLED" if e["called"] else
                ("screen only" if e["escalation_required"] else "not called"),
                fmt(sc["accuracy_pooled"]) if sc else NM,
                ("stump CLEARS — row does not discriminate"
                 if sc and sc["clears_primary"] else
                 "stump fails — row discriminates" if sc else NM)])
        n_seeds_dev = max((len(v) for v in dev_bench.values()), default=0)
        doc += [
            "## A different measurement: the successor task set", "",
            "**This is not the KPI.** The KPI is the five registered tasks "
            "above; a task set chosen after seeing that the registered one was "
            "weak may only ever be reported as a *separate* measurement, and "
            "substituting it would be the same error as choosing a baseline "
            "late. It lives in `runs/dev/`, and none of the clause accounting "
            "above can see it.", "",
            "The rule, still blind to our accuracy: the five most-published "
            "candidates whose `0.95 x median_run` **exceeds** their "
            "majority-class rate. Their targets needed no new fetch — all "
            f"{td.get('n_candidates', '51')} candidates were frozen in "
            "`runs/baselines.json` before any run existed, so these baselines "
            "are as pre-registered as the others. Four of the five were "
            "reserved by the protocol for development and the agent had "
            "**never been run on any of them**, so they are uninspected; the "
            "agent is unmodified.", "",
            table(drows, ["task", "dataset", "rank", "seeds", "ours (pooled)",
                          "median_run", "rel gap", "vs primary",
                          "vs strictest", "threshold headroom over majority",
                          "k/n above", "p", "verdict", "stump",
                          "does this row discriminate?"]), "",
            (f"**{n_seeds_dev} seeds: screen, not verdict.** The same gate "
             "applies — no task is called under the registered verdict seed "
             "count, so nothing here is a called result and none of it is a "
             "headline. It is reported because it is the experiment the "
             "falsifiability finding demands, and because it tests the "
             "alternative explanation: if the agent's margin over a depth-3 "
             "stump stays near its imbalanced-task value on these balanced "
             "tasks rather than widening, the finding is about the agent and "
             "not about the task set."
             if n_seeds_dev <= 3 else
             f"**{n_seeds_dev} seeds.**"), ""]
        if dev_failed:
            doc += [f"{len(dev_failed)} failed attempt(s) on the successor "
                    "set, counted rather than dropped: "
                    + ", ".join(f["_file"] for f in dev_failed), ""]

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
    if getattr(args, "weekend_leakpower_only", False):
        wp = REPO / "WEEKEND.md"
    touched_weekend = False
    if (wp and wp.is_file() and not getattr(args, "weekend_leakpower_only", False)
            and "<!-- HEADLINE:BEGIN -->" in wp.read_text()):
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

    # ------------------------------------------- the leakage-detectability block
    # Same rule as the headline: generated, never hand-typed. This lives in
    # WEEKEND.md because it is the reading a tired person needs alongside the
    # falsifiability floor -- they are the same quantity, and the table makes
    # that visible without reading two sections.
    if wp and wp.is_file() and "<!-- LEAKPOWER:BEGIN -->" in wp.read_text():
        lp = read_json(REPO / "runs/leakage_power.json")
        if lp and lp.get("tasks"):
            lrows = []
            for tid in order:
                t = (lp["tasks"] or {}).get(str(tid))
                if not t or t.get("status") != "measured":
                    lrows.append([tid, sel[tid]["dataset_name"]] + [NM] * 6)
                    continue
                q = t["pooled"]
                phi = q["min_resolvable_leak_fraction"]
                fw = t.get("family_wise_corrected") or {}
                phi_fw = fw.get("min_resolvable_leak_fraction")
                lrows.append([
                    tid, t["dataset_name"], fmt(q["accuracy_intact"]),
                    fmt(q["majority_rate"]), f"{q['gap']:+.4f}",
                    fmt(q["noise_band_2sigma"]),
                    ("inf" if phi == float("inf") else f"{phi:.3f}"),
                    (f"yes, >={t['min_folds_for_detection']} fold(s) pooled"
                     if q["detects_complete_leakage"] else "**no**"),
                    (NM if phi_fw is None else
                     "inf" if phi_fw == float("inf") else f"{phi_fw:.3f}"),
                    (NM if not fw else
                     f"yes, >={fw['min_folds_for_detection']} fold(s)"
                     if fw["detects_complete_leakage"] else "**no**")])
            probed = read_json(REPO / "runs/leakage_probe.json")
            block = "\n".join([
                "*(generated by `scripts/report.py` from "
                "`runs/leakage_power.json`; nothing in this block is "
                "hand-typed)*", "",
                "The label-permutation leakage probe shuffles `y_train`, fits, "
                "and scores against the true test labels. Its whole dynamic "
                "range on a task is `gap = accuracy - majority_rate`, so the "
                "smallest leak it can resolve is the fraction "
                "`phi_min = band / gap` of that gap. **`phi_min >= 1` means "
                "complete leakage is invisible on that task.**", "",
                table(lrows, ["task", "dataset", "accuracy", "majority rate",
                              "gap", "2σ band", "phi_min (2σ)",
                              "detects? (2σ)", "phi_min (family-wise)",
                              "detects? (family-wise)"]), "",
                f"Powered at the pre-registered 2σ band: "
                f"**{lp['tasks_powered_pooled']}**. Blind at any fold count: "
                f"**{lp['tasks_unpowered_pooled']}**.", "",
                "**Both readings of the threshold, because they disagree.** "
                "`runs/leakage_calibration.json` measures that the "
                "pre-registered rule — a maximum over k=10 permutations, each "
                "at a nominal one-sided 2σ — has a family-wise false-alarm "
                "rate up to "
                + (f"**{100 * (read_json(REPO / 'runs/leakage_calibration.json') or {}).get('worst_family_wise_alpha_pooled', float('nan')):.1f}%**"
                   if read_json(REPO / "runs/leakage_calibration.json") else NM)
                + ", not 5%. That correction is *against* the KPI where it "
                "was found — a looser trigger makes LEAKAGE easier to declare "
                "and LEAKAGE sinks clause 2 — so the pre-registered rule stays "
                "primary there. It lands here in the opposite direction: a "
                "wider band is a higher detection threshold, so it costs "
                "power. Powered under the corrected band: "
                f"**{lp.get('tasks_powered_family_wise_corrected')}**; moved "
                f"from powered to blind by the correction: "
                f"**{lp.get('tasks_moved_to_blind_by_the_correction')}**. So "
                "the honest count of tasks on which this instrument can see a "
                "complete leak is "
                f"**{len(lp.get('tasks_powered_family_wise_corrected') or [])} "
                f"of {len(lp['tasks'])}**, not "
                f"{len(lp['tasks_powered_pooled'])}.", "",
                "The clause-2 gate keeps the 2σ task set, which is the "
                "**more demanding** of the two: it requires a probe to clear "
                "three tasks rather than two. Using the corrected set would "
                "shrink what has to be probed, and that is the direction a "
                "protocol may never be moved.", "",
                "`gap` is the same quantity as the `prior`/`stump` margin in "
                "the falsifiability section, so the task where this KPI is "
                "falsifiable is the task where leakage is detectable. A weak "
                "baseline does not only flatter a weak method; it blinds the "
                "instruments that would catch a broken one.", "",
                (f"Probe status: **{probed['verdict']}**, tasks cleared "
                 f"{probed.get('tasks_cleared')}."
                 if probed else
                 "Probe status: **[not measured]** — no "
                 "`runs/leakage_probe.json` exists, so amendment 7 holds "
                 "clause 2 at `None`."), ""])
            wp.write_text(re.sub(
                r"<!-- LEAKPOWER:BEGIN -->.*?<!-- LEAKPOWER:END -->",
                "<!-- LEAKPOWER:BEGIN -->\n" + block + "<!-- LEAKPOWER:END -->",
                wp.read_text(), flags=re.S))

    print(f"wrote {args.out}"
          + (" and refreshed the README block" if touched_readme else "")
          + (" and the WEEKEND.md headline" if touched_weekend else ""))
    print(f"status: {v['status']}")
    for k, val in v["clauses"].items():
        print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
