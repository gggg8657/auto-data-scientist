# critique log — auto-data-scientist (B5)

KPI: **공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 · end-to-end 무개입**

---

## 2026-09-10, turn 1 — before any number exists

### State found

`ads/` held a working package (profiling, decision log, agent, evaluation) from
the previous turn, with **no commits, no README, no baselines, no tests and no
report script**. So there was nothing to critique numerically and the whole turn
goes to the thing that has to happen before a number exists: fixing the target.

### What was done

1. `scripts/fetch_baselines.py` — the pre-registration. Selects five tasks from
   OpenML-CC18 by a rule that cannot see our accuracy (`n ≤ 20000`, `p ≤ 100`,
   then top 5 of the 51 survivors by number of published evaluations), pages
   **every** published `predictive_accuracy` evaluation on each, and writes
   `runs/baselines.json` with a sha256 per raw evaluation file.
2. README carries the KPI verbatim, all three tolerance readings and both
   baseline readings, **written before any result**.
3. `scripts/report.py` — the only code allowed to write a number into a
   document. `verdict()` derives the status from the clause rows.
4. 15 tests across three files.

### Critique, in advance of the numbers

**1. The primary baseline is probably weak, and the report must say so where the
number appears.** `median_run` is the median over every accuracy anyone ever
uploaded for the task, which on a classic CC18 task means a large mass of
default-parameter Weka and early-sklearn runs from 2014 onward. "Within 5% of
the median published run" is a much weaker claim than "competitive with the
published frontier", and a modern `HistGradientBoosting` with a small random
search should clear it comfortably. This is the *brief's* chosen definition and
it is defensible as a measured, citable statistic — but if our accuracy lands
above `q90` then the KPI is met by a criterion that was never demanding, and the
honest report says that in the same table. `q75`, `q90` and `max_published` are
therefore in `baselines.json` as context columns from the start, not added later.
**The prediction, recorded now so it can be wrong:** we clear the primary
reading on all five tasks and land between `q75` and `q90` on most of them.

**2. Task selection by run count is a bias, fixed before measurement rather
than removed.** The most-run CC18 tasks are the oldest and most famous ones,
which are also the small, clean, easy ones. Selecting on run count therefore
selects for datasets where the median is well determined *and* where the
problem is easy. The alternative — selecting for difficulty — would be choosing
tasks to make a point. I take the bias, name it, and note that a harder
five-task set is a different measurement, not a better one.

**3. The comparability of our number and theirs is asserted, not yet verified —
and this is the one that could invalidate the whole comparison.** The claim in
`ads/openml_io.py` is that evaluating on the task's own estimation procedure
makes our accuracy comparable to a published run's. That is necessary but not
sufficient: OpenML's server computes one scalar `value` per run, and I do not
yet know whether it is the **pooling** of predictions over all folds or the
**unweighted mean** of per-fold accuracies. For 10-fold CV on an n divisible by
10 the two agree to rounding; where folds are unequal they do not, and I would
be comparing two different statistics while calling it one. `run_task` already
records both (`accuracy_pooled`, `accuracy_mean`), so the fix is a measurement,
not a guess: the fetched evaluation rows carry a `values` column with the
per-fold array, so I can check, on real published runs, which aggregation
reproduces the server's `value`. **Until that check runs, no comparison in this
repo is legitimate.** That is the next thing to do after the fetch finishes,
ahead of running the agent.

**4. The noise floor is unmeasured, so no task near the line may be called
either way.** The outer folds are the task's own and identical across seeds, so
the only variance is the agent's: the inner-CV shuffle, the selection
subsample and the random search. Found and fixed this turn: every estimator was
pinned at `random_state=0` regardless of the agent's seed, so a seed sweep would
have reported a spread far below the truth and I would have believed it.
`test_seeds_actually_change_the_agent` now fails if that regresses (seed 0 vs 7
on the same synthetic data: rf inner-CV 0.7825 vs 0.7500 — a 0.033 spread on
n=400, which is already larger than most gaps I expect to be arguing about).
**Pre-registered protocol:** 3 seeds per task as a screen; any task whose
`|gap to baseline|` is within the measured seed range of the 5% line gets 8
seeds and an exact test before it is called.

### Infrastructure findings, recorded because both cost real time

- `/tmp/struct.py` exists on this box (another loop's artifact) and **shadows
  the stdlib `struct` module** for any script run from `/tmp`, which breaks
  `import openml` with a `FileNotFoundError` about a file in a different
  repository. Every scratch script in this project therefore lives in
  `.tmp/` inside the repo. Not touched, not mine.
- `pgrep -f fetch_baselines` matches the shell whose own command line contains
  the pattern, so `kill $(pgrep -f ...)` killed the shell issuing it (exit 144)
  and the edit that command was supposed to make silently did not happen. Kill
  by recorded pid.
- The OpenML evaluation endpoint refuses `limit ≥ 2000` (546 *Requested result
  limit too high*), so paging is at 1000; and the bare REST endpoint returned
  412 from this host, as the board already recorded. One page would sample the
  oldest runs by `run_id` and bias the median toward the 2014 Weka era, so the
  fetch pages to exhaustion. Serially that is hours for 51 tasks, so it runs
  6-way threaded, resumable, with `.part` → rename so an interrupted fetch
  cannot leave a truncated cache the next run would trust.

### Not done, and named rather than left implicit

- No agent run has happened yet, so clauses 2 and 3 are `[not measured]` and
  `RESULTS.md` says exactly that rather than omitting the rows.
- `median_flow` is computed but its sensitivity is unexplored.
- No adversary has been run against this design yet.

---

## 2026-09-10, turn 2 — the registry was wrong in three ways, and the disk was full

No agent accuracy exists yet, so there is still nothing to critique numerically.
What this turn found instead is that **the pre-registration committed last turn
was not the pre-registration the code on disk describes**, in three independent
ways, and all three were caught by tests that were written to fail until the
registry is actually correct. That is the tests doing their job, and it is worth
being explicit that none of these were found by reading the numbers — they were
found by running `pytest`, which the previous turn did not do after the fetch
finished.

### Finding 1 (infrastructure, blocked every track): the volume was 100% full

`df -h /` reported 176G free and was the wrong filesystem. `/home/dongjukim` is
a **separate ext4 mount** (`/dev/nvme1n1p1`, 7.0T) and it was at **0 bytes
available**. Every write in this repo failed with `ENOSPC`: the test suite could
not even collect (4 collection errors), and the smoke run died writing its
output.

Of the 6.6T used, only 1.5T is under `/home/dongjukim` — `.cache/huggingface`
267G, `.ollama` 278G, `.cache/pip` 40G — and the other ~5.1T is outside this
container's view on the same device, so it is not mine to reclaim. I deleted
**`~/.cache/pip` (40G)** and nothing else: a pure download cache, regenerable,
with nothing depending on it at runtime. I deliberately did **not** touch
`.cache/huggingface` (F4 needs the Qwen3 weights and other tracks may be relying
on it) or `.ollama` (not used by any track in my brief, and not mine).

40G free on a 7.0T volume three tracks are writing to is not a fix, it is
headroom. **This needs a human decision (see WEEKEND.md).** Checked and ruled
out as the cause: inodes (3% used) and large deleted-but-open files (the only
ones are trivial npm logs).

### Finding 2: `run_benchmark.py`'s "complete or absent, never half-written"
guarantee was prose, and the outage falsified it

Its module docstring claims a run record is never half-written. `ENOSPC` left
**exactly 4096 bytes** — one page — of `runs/bench/task_3913_seed99.json` on
disk at the record's own path. `write_text` raising did not remove it, and
`run_benchmark`'s own `out.exists()` skip-guard would then have treated that
stump as a finished task on the next invocation.

I got lucky in the detail that matters: 4096 bytes cut mid-token, so
`json.loads` raised and `test_baselines_recompute_from_the_raw_evaluations`
died on a `JSONDecodeError`. **The dangerous version of this bug is the one
where the prefix is well-formed** — a shorter but valid JSON document — because
then a task gets scored on however many folds landed inside the first page and
nothing complains. Fixed with `write_json_atomic` (serialise fully in memory →
`.part` → `flush` → `fsync` → `rename`, `.part` unlinked on any failure), and
pinned by four tests in `tests/test_atomic_run_write.py`, including that a
failed rewrite leaves the *previous* complete record intact. The corrupt file
was deleted; it was an off-registry `--max-folds 1` debug run created this turn,
so nothing was lost.

`report.py` already gates clause 3 on `complete`, `off_registry`, `.FAILED.json`
presence, missing seeds and registry-digest agreement, so a partial run cannot
reach a clause even if one survives on disk.

### Finding 3: the registry was produced by code that no longer exists on disk

`runs/baselines.json` (written 09:10) is missing four keys and a whole block
that `scripts/fetch_baselines.py` (mtime 08:45) emits: `median_flow_best`,
`median_uploader_best`, `strictest_baseline`, `strictest_baseline_value`, and
the entire `run_protocol`. The reason is a race, not a bug in either file: the
fetch ran 08:33–09:10, the script was **edited at 08:45 while that fetch was in
flight**, and the running interpreter kept the code it had loaded. So the
committed artifact and its committed producer disagree, and five of the eight
test failures reduce to this single cause.

This is the same family as the race `etch-operator-twin` recorded on 2026-09-10
(an eval JSON that predated its own checkpoint by 859 s). The lesson generalises:
**a long-running producer plus a mid-flight edit yields an artifact no code in
the repository can reproduce.** The fix is free, because the current script
already computes everything the tests demand — re-run it.

### Finding 4, the one that actually matters: two of the five targets were
medians of a truncated prefix, and the commit message claimed otherwise

`fetch_baselines.py` paged with `HARD_CAP = 300_000`. **Tasks 31 (credit-g) and
10101 (blood-transfusion), ranks 1 and 2, both hit it exactly** — the only 2 of
51 that did. So their `median_run` was the median over the first 300,000
evaluations *ordered by `run_id`*, which is oldest-first. I probed the true
extent: **task 31 has 500k–600k published evaluations and task 10101 has
400k–500k**, so the medians that were about to become my primary targets were
computed over roughly 55% and 70% of the population, drawn from the oldest end —
precisely the "biased toward the 2014 Weka era" failure that turn 1's own
critique named as the reason to page to exhaustion in the first place.

Worse, commit `0c9dd8a`'s message states that the registry was built "with every
published `predictive_accuracy` evaluation **paged to exhaustion** and
sha256-pinned". That was false for 2 of the 5 selected tasks at the moment it
was written. It is the same species of error as the `claim-auditor` headline
that survived in `RESULTS.md` while the table under it disagreed: prose asserting
a property the artifact did not have.

And a fourth defect, smaller but of a piece: **`runs/baselines.json` is not
tracked by git.** It was written at 09:10, 33 minutes *after* the 08:37 commit
whose message describes having committed it. "Fetched, committed and frozen
before any agent run exists" was two-thirds true — fetched, frozen by nothing,
committed not at all.

**Why raising the cap is a fix and not a loosened protocol.** This is the one
distinction the weekend's rules care most about, so: the change is to the
*sampling of the baseline population*, made **before any accuracy of mine
exists**, in the direction the pre-registered protocol already specified
("page to exhaustion"). I cannot be steering the target toward my own result
because I do not have one. The superseded readings are recorded in
`runs/baselines_truncated_superseded.json` and quoted here so the move is
auditable in both directions:

| task | superseded `median_run` (300k prefix) | superseded `median_flow` |
|---|---|---|
| 31 credit-g | 0.7200 | 0.7290 |
| 10101 blood-transfusion | 0.763369 | 0.762032 |

If the exhaustive medians come out **higher**, the targets got harder and this
was necessary. If they come out **lower**, the targets got easier and I will say
so in the same table in `RESULTS.md`, because a fix that happens to help me
still has to be reported as a fix that happened to help me.

**The selection rule is provably stable under this change**, which is the one
thing that could have made the fix unrecoverable: only tasks 31 and 10101 were
truncated, both at ≥300,000, while rank 3 (task 3913) has 173,611. De-truncating
can only raise their counts, so the top-5 by published-run count cannot change,
and I am not swapping tasks — only correcting two of their baselines. Had a
*non-selected* task been truncated, the ranking itself would have been in
question and the honest move would have been a full re-fetch of all 51.

### Prediction, recorded before the fetch finishes

Turn 1 predicted we clear the primary reading on all five tasks and land between
`q75` and `q90`. I am adding a sharper one, because the de-truncation is a real
experiment on the baseline: the newer runs are the ones that will be missing from
the old prefix, and uploads since ~2018 skew toward tuned scikit-learn rather
than default-parameter Weka. **So I expect both exhaustive medians to come out
above the superseded ones** — targets get harder — and I expect the effect to be
larger on task 31 (credit-g, 766 flows, where 45% of the population was missing)
than on 10101 (321 flows, 30% missing).

### Still not measured, and named rather than left implicit

- Clauses 2 and 3 remain `[not measured]`. No agent run exists.
- `runs/metric_check.json` does not exist under the name the test requires. The
  task-11 probe that settled the aggregation question (server value is the
  **size-weighted/pooled** accuracy, reproduced to 1.1e-16 on 200 published
  runs, vs up to 1.5e-4 error for the unweighted mean) is real and is the
  gating result turn 1 said had to come first — but it lives in
  `runs/metric_check_task11_probe.json` and covers one task. `verify_metric.py`
  already defaults to the canonical path; it needs to run across the five
  selected tasks once the registry regenerates.
- No adversary has been run against this design yet. Still true, still a gap.

### The adversary, asked the right question

The addendum's rung 4 is to ask `codex` *"how would you make this pass?"*
rather than *"what is wrong with this?"*. The pre-measurement form of that
question is sharper still, and it is the one I asked: **"how would you fake
this KPI under this exact design, without editing any number by hand and
without failing any existing test?"** It returned eight ranked attacks. Its
own framing of the result is worth quoting, because it is the correct
diagnosis of what this repo's tests do and do not cover:

> "I would attack the evidence-selection process, not the arithmetic. The
> design verifies that retained artifacts agree with one another; it does not
> establish that they represent an untouched, prospectively executed
> experiment."

> "The existing recomputation, partial-run, missing-task, and registry-binding
> checks close real shortcuts. They do not yet establish evidence
> completeness, genuine pre-registration chronology, or independent
> enforcement of 'no intervention.'"

Four were real and are closed this turn. One I had already found and fixed
independently the same turn, which is the only one I would claim as a draw.

**#3, duplicated seeds — real, and I had it first.** `ours` is a mean over run
*files*; completeness was checked against the *set* of `random_state` values.
A set cannot see a duplicate. `cp task_31_seed2.json task_31_seed2_rerun.json`
reweights the mean toward the luckiest seed with no number edited and no test
failing. Measured on a three-seed screen of (0.74, 0.75, 0.78): **0.7567 →
0.7660**, which is larger than several gaps this benchmark will have to
decide. Closed by making run files a bijection with the registered seeds.

**#2, the escalation that was registered and never enforced — real, and the
worst of the eight**, because it required no action at all to exploit: run the
three screen seeds and report. `run_protocol.escalation_rule` had said "8 seeds
and an exact test" since before any result existed, `report.py` checked only
that the screen seeds were *present*, and "an exact test" named no test. Both
are now specified and enforced; the definition is in `report.py` and the
p-values are pinned in `tests/test_escalation_gate.py`. Note what this does to
turn 1's prediction: if we clear the primary reading comfortably, escalation
never triggers and the screen is enough. It bites exactly where it should —
tasks whose verdict is inside our own noise.

**#5, developing on the confirmatory tasks — real.** `git rev-parse HEAD` was
recorded; the tree that actually ran was not. Adjust a threshold in `ads/`,
re-run, report: every run genuine, zero logged interventions, HEAD unchanged.
Closed with a digest over `ads/*.py` plus a dirty flag, and **it fired
immediately** — `ads/agent.py` was already modified vs HEAD. The content turned
out benign (the `N_JOBS` knob, covered by
`test_n_jobs_does_not_change_predictions`), which is the point: the check
cannot tell benign from not, so the tree has to be clean before the
confirmatory run, and now it must be.

**#4, deleting unfavourable attempts — real.** `mv` the bad files out of
`runs/bench/` and every check passes on what remains; the regeneration test
faithfully reproduces the curated collection. Closed with
`runs/attempts.jsonl`, written *before* each attempt's outcome is known, and
reconciled in `report.py`. Honest limit, in the docstring rather than here:
it is append-only by convention, not by permission. It converts a one-command
deletion into two coordinated edits and puts the reconciliation in
`RESULTS.md` where a reader sees it.

**#1, passing the weak reading while failing the strong ones — real as a
criticism, and I am not changing the primary.** `median_run` was
pre-registered, the brief named it, and swapping the primary now — even for
something harder — is still rewriting the target after the fact. What was
wrong is that `strict_pass` was computed and left unbinding. The
strictest-baseline verdict is now derived and reported beside the primary one,
so a PASS on the weak reading with a FAIL on the selected-solution readings is
visible in the status block, not buried in a column.

**And the one substantive correction to my own documentation.** codex:

> "the producer's claim that both 'best' readings are necessarily stricter
> than `median_run` overstates the mathematics."

It is right, and this is a false claim I wrote. `median_flow_best ≥
median_flow` is guaranteed on the same groups; `median_flow_best ≥ median_run`
is **not** — a task with many weak flows can have a per-flow-best median below
the median over all runs, because `median_run` is weighted by how often each
flow was submitted. The `asymmetry_note` in `fetch_baselines.py` asserts the
"best" readings "are strictly harder". That has to become a statement about
which reading *is* strictest per task, which the registry already records
empirically in `strictest_baseline`. Queued for the moment the fetch releases
the file — editing it now would repeat exactly the mid-flight-edit race of
finding 3.

**#6 (identity encoded as a measured property), #7 (poisoning the evaluation
cache before freezing it) and #8 (label leakage via a feature fingerprint)
are real and are not closed.** #6 is partly answered now: the literal-id scan
was replaced by an invariance test — renaming every column and reversing their
order leaves the chosen family, the logged decisions and 100% of predictions
unchanged — which closes the schema-fingerprint route but not a rule keyed on
values, and I have said so where the test lives rather than implying more. #7
is correct that a sha256 computed over rows I control proves consistency, not
authentic acquisition; the cheap partial answer is a uniqueness check on
`run_id` and a candidate-membership check, neither of which is built. #8 is
architecturally open: `ads/evaluate.py` holds the full labelled frame in the
same process that calls the agent, so isolation would mean a subprocess with
only the permitted arrays mounted. All three are in `WEEKEND.md` as work, not
as answered.

---

## 2026-09-10, turn 6 — the first accuracy numbers exist, and the gate that passed them was biased at n=3

### What the number is

The 3-seed screen finished. Five tasks, `runs/bench/task_*_seed{0,1,2}.json`,
pooled accuracy against the pre-registered `median_run`:

| task | dataset | ours (3-seed mean) | median_run | rel gap | seed range | margin to 0.95x line / range |
|---|---|---|---|---|---|---|
| 31 | credit-g | 0.7563 | 0.7250 | +4.32% | 0.0040 | 16.9x |
| 10101 | blood-transfusion | 0.7678 | 0.7634 | +0.58% | 0.0053 | 8.0x |
| 3913 | kc2 | 0.8372 | 0.8276 | +1.16% | 0.0077 | 6.7x |
| 3 | kr-vs-kp | 0.9969 | 0.9599 | +3.85% | 0.0013 | 67.8x |
| 3917 | kc1 | 0.8592 | 0.8516 | +0.89% | 0.0038 | 13.2x |

All five are *above* the median published run, not merely within 5% of it, and
all five clear the strictest of the four registered readings too. These are the
first accuracy numbers this repository has ever produced; the previous three
turns reported `[not measured]` and that was correct at the time.

### The claim I am withdrawing before anyone reads it

`scripts/report.py` printed **`Status: PASS`** off that screen, and wrote it
into `RESULTS.md`. That is withdrawn. Not because any number above is wrong —
they are all real, and they are all comfortably clear of the line — but because
the gate that granted the clause was the wrong test:

    near = margin_to_threshold < observed_seed_range
    called = (not near and mean >= T) or (near and test.reject_h0)

The pre-registered exact sign test therefore **ran on none of the five tasks**.
The clause was granted on point estimates, with a statistical test present in
the file, named in the protocol, and unreachable in this branch.

Why it is biased and in which direction: the expected range of `n` iid draws is
**1.69σ at n=3 and 2.85σ at n=8**. The seed range is the quantity in the
*denominator* of `margin < range`, so running fewer seeds shrinks it and makes
`near` *false* more often — i.e. makes the "comfortably clear, no test needed"
branch easier to enter. **A gate that is easier to clear on less evidence is
not a gate.** It is the same shape of error as `etch-operator-twin`'s partial
checkpoints: a defect whose sign is fixed by the mechanism, so it cannot
average out.

This is exactly what the weekend rules mean by "with 3 seeds, say screen, not
verdict", and it was prose in this repo rather than code.

### The fix, and why it is a tightening and not a protocol change

The pre-registered exact test is now the gate for **every** task:
`H0: median over seeds ≤ 0.95 × baseline`, one-sided exact sign test,
α = 0.05. Its p-value floor is `1/2^n`, so:

- n=3, all three above the line → p = 0.1250. **Not callable, in either
  direction, at any margin.** The 70×-margin fixture in
  `tests/test_escalation_gate.py` asserts precisely this.
- n=8, all eight above → p = 0.0039 → called.

Under the new gate the same five screen runs call **nothing**, and the status
went `PASS` → `RUNNING (not all clauses measured)`. The clause got harder; no
number moved. The old margin reading is kept as
`margin_exceeds_seed_range` and printed beside the test (rung 1: report both
readings, labelled), because a reader should see that the two disagree here.

### A second hole, found by writing the test rather than by reading the code

The p-value floor blocks n ≤ 4 only. At **n=5** an all-above run gives
p = 0.03125 and would reject. So a runner that watched seeds land and stopped
at the first rejection would be reporting a **sequentially monitored p-value as
if it were fixed-sample** — optional stopping, and the favourable-prefix
version of the duplicated-seed attack codex found last turn. `called` now
additionally requires the full registered verdict set of 8 seeds, whichever way
the test comes out. `test_a_favourable_prefix_of_the_verdict_set_cannot_be_called_early`
asserts n=5,6,7 reject H0 and are still not called.

### Unfinished is not failed — the mirror-image error, which I then made

Draining the same gate in the other direction: with the exact test enforced, a
3-seed screen made `clause2 = False` and the status read **"NOT MET as
measured"**. That is as wrong as the PASS was, with the sign flipped: a
measurement that has not reached its registered seed count has not measured the
clause. `clause2` and `clause2_under_strictest_baseline` are now `None` while
`escalation_required` holds, which routes to `RUNNING`. A task that fails at
the *full* 8 seeds is a genuine negative and still reports `NOT MET`
(`test_a_task_that_fails_at_the_full_seed_count_is_a_result_not_a_pending_run`).

The same error hit clause 3 through the ledger: an attempt in flight has a
`started` line and no record yet, so the ledger cannot reconcile and clause 3
read `False` mid-run. My first fix read `/proc` inside `reconcile_ledger` to
tell "in flight" from "abandoned" — and that was **worse**, because it made the
committed document a function of live process state, so
`test_results_md_matches_what_report_regenerates` could never hold on a clean
checkout and CI would have been permanently red. Reverted. The live reading now
sits in `main()` as a guard instead: while a benchmark is alive, the default
output is redirected to the untracked `runs/interim_report.md` and `RESULTS.md`
keeps describing the last finished measurement. An explicit `--out` is never
redirected, so the freshness test still regenerates deterministically. That
test defers, loudly, while a run is in flight, and CI sets
`ADS_REQUIRE_FRESH_RESULTS=1` to make the deferral itself a failure — verified
both ways this turn.

### What is the binding constraint on this KPI

Not the agent, on this evidence. The agent is above the median published run on
5 of 5 tasks with a seed spread of 0.0013–0.0077, i.e. one to eight thousandths
of accuracy, against margins of 0.03–0.08. If clause 2 fails at 8 seeds it will
not be because the models are weak; it will be because a task's *strictest*
reading (`median_uploader_best`: 0.7741 on 10101, 0.8448 on 3913, 0.9947 on 3)
sits above our mean, and only the 0.95× tolerance carries the row. The primary
reading is `median_run` and stays primary — it was pre-registered — but the
honest summary is that **clause 2 is comfortable on `median_run` and tight on
`median_uploader_best`**, and the report prints both.

The distinguishing prediction, written before the 8-seed run lands: the added
seeds move each mean by less than 0.005 and change no task's `primary_pass`;
what changes is that `p` goes 0.1250 → 0.0039 and the tasks become callable. If
instead a mean moves by more than 0.005, the 3-seed spread was an
underestimate of exactly the kind argued above, and that is the more
interesting outcome.

### Still open, unchanged from last turn

Identity-as-property (#6), evaluation-cache provenance (#7), in-process label
leakage (#8). None of them is closed and none is claimed to be.

### The adversary, asked the rung-4 question ("how would you make this pass?")

`codex exec`, given `escalation_state` / `exact_sign_test_above` / `verdict`
and the state above, and asked (1) how to make the clause pass legitimately and
(2) where the new gate is still wrong. Four findings, in the order they matter.

**1. The tie handling inflated Type I error to 17.37% at a nominal 5%. Real,
fixed.** Quote:

> "Dropping ties invalidates the stated general median null. [...] With
> discrete accuracy, the median null does not imply that non-ties split equally
> above and below. A concrete counterexample: suppose P(A=T)=0.6, P(A>T)=0.4,
> and nothing lies below. The unique median is T, so the null is true. Five
> wins and three ties produce your p=1/32, and the full eight-seed gate rejects
> with probability **17.37%**, not at most 5%. Direction: flattering,
> anti-conservative."

I recomputed it rather than taking the number: dropping ties rejects a true
null **17.367%** of the time; keeping n=8 and counting only strict wins rejects
**0.852%**. The exact figure it quoted is exactly right. `k` is now strict wins
out of all `n` seeds, ties included in the denominator as non-wins. This costs
nothing on this repository's data — no seed lands on a `0.95×` threshold, so
every `k` and `n` here is unchanged — and it changes the test from
anti-conservative to conservative wherever ties do occur. The counterexample
is a test now (`test_dropping_ties_would_inflate_type_i_error_to_17_percent`).

This is the single best thing any adversary has found in this repo. I wrote the
old version *citing* the textbook sign test and did not notice that the
textbook version assumes a continuous distribution, which accuracy is not.

**2. The provenance gate fails open. Real, fixed.** Quote:

> "Missing intervention counts become zero [...]; missing agent digests are
> discarded [...]; an absent ledger is explicitly accepted. Direction:
> flattering. [...] Agreement among available self-reports is weaker than
> evidence of no intervention."

All three were `.get(field, default)` written in the accommodating direction, so
a run that *omitted* a field read as a run that *reported it clean*. Now: a run
missing any of `n_interventions` / `complete` / `off_registry` /
`registry_sha256` / `random_state` sinks clause 3; an absent agent digest counts
as its own distinct digest instead of being dropped from the set (so 39 runs
agreeing can no longer outvote the one that said nothing); an absent or
unreconciled ledger sinks clause 3, because without it "no attempt was started
and abandoned" has no record behind it. One test per route.

**3. The chronology: this is an amendment, not a pre-registration. Real,
recorded.** Quote:

> "The code says the particular test was chosen after seeing the screen [...]
> and the registry still contains the old conditional escalation rule. If no
> earlier timestamped record specifies this test, call this a protocol
> amendment."

Correct, and I had been sloppy about it: the registry's `escalation_rule` says
"a task whose |gap| is within the measured seed range gets the 8-seed set and an
exact test", which is *conditional*; the amended rule tests every task
unconditionally. The exact sign test was specified on 2026-09-10 in response to
an adversary, not on 2026-09-09 when the protocol was frozen, and the 3-seed
screen had been read. `runs/protocol_amendments.json` now records all three of
this turn's rule changes with what was registered, what replaced it, whether
results had been seen, and what the amendment did to the claim as it stood.
`tests/test_protocol_amendments.py` asserts that **no amendment is recorded as
making a clause easier**, and fingerprints the three functions that decide the
verdict so an undeclared gate change goes red.

The defence of amendment #1 is not that it was planned. It is that it took the
claim from `PASS` to `RUNNING` on identical data. An amendment that withdraws a
claim needs no chronological alibi; one that grants a claim cannot have one.
What is still owed, and is now written into the file and the report: **3 of the
8 verdict seeds (0, 1, 2) were inspected before the amendment.** Seeds 3-7 are
disjoint from anything I had seen.

**4. The unit of analysis. Half real, and it produced a new reported row.**
Quote:

> "Eight seeds reuse the same examples. They can estimate algorithmic
> randomness conditional on those data; they do not create eight independent
> datasets. [...] taskwise median success does not establish that one autonomous
> execution clears all five tasks reliably. [...] For that claim, additionally
> report the per-seed event 'all five tasks completed and cleared their
> thresholds'." And: "Do not 'fix' this by treating the 80 fold scores as
> independent observations."

The caveat is right and is now prose in `RESULTS.md` rather than something a
reader has to know. The joint-event suggestion is better than a caveat, because
it is the reading a person actually wants from an *autonomous* data scientist:
`joint_seed_event()` reports, per seed, whether one unattended run cleared all
five, and runs the same exact test on that indicator. It is reported *beside*
the per-task gate, not instead of it — five tasks each failing on a different
seed would pass every per-task test and never produce a clean sweep.

Interim, seeds 0-3 (seeds 4-7 still running): **4 of 4 complete seeds cleared
all five**, against the strictest per-task reading as well as `median_run`;
k=4/4, p=0.0625 — not yet significant, which is the p-floor doing its job at
n=4.

Writing that function immediately caught a bug in itself, in the pessimistic
direction: it counted a task that had *not yet run* for a seed as a task that
seed had missed, so the mid-flight seed read "no — missed tasks 3 and 3917"
when those two had not started. Unrun tasks now take the seed out of the test
entirely rather than scoring it a failure, and the table has separate columns
for "below the line" and "not yet run". Regression test added. That is the
second time this weekend a partial artefact scored as a bad result rather than
as no result — the same mechanism as `etch-operator-twin`'s partial checkpoints,
and worth naming as a recurring class rather than two coincidences.

**Where codex was wrong, or at least not actionable:** it flagged that
`RESULTS.md` "shows zero measured seeds and no new gate table", which was an
artefact of reading the file at the moment I had reverted it to HEAD pending
the 8-seed run. It also proposed a matched human comparator (freeze a
human-selected pipeline per task on training data only, evaluate both on
identical untouched data, test non-inferiority of the pairing). That is the
right experiment for the question "is this agent as good as a person", but it
is **not this KPI** — the KPI names a public-baseline comparison and the
baselines are pre-registered from OpenML's run history. Building a human arm
would be replacing the registered target with a better one after the fact.
Noted in `paper_draft.md` as the experiment this design cannot do, not adopted.

Its argument that no multiplicity correction is needed is correct and I am
recording it as an argument I do *not* have to fix: the claim requires all five
tasks to reject, which is an intersection-union test, so a false global PASS
needs at least one true task null rejected and Bonferroni would only make the
global claim gratuitously harder.

### A second adversary, and it found the sharper set

`agy` (after two failed invocations — it reads prompts only from `-p`/stdin,
and headless mode auto-denies file reads without
`--dangerously-skip-permissions`; recorded so the next turn does not re-derive
it) was asked the same two questions against the *amended* state. It returned
seven holes. Five were real. They are, in order of how much they mattered:

**#3, and the best finding of the weekend so far: `runs/bench/` and
`runs/attempts.jsonl` were never tracked by git.** Quote:

> "If a seed produces low accuracy or crashes, deleting both the JSON file in
> `runs/bench/` and the corresponding `started`/`completed`/`failed` lines from
> `attempts.jsonl` leaves zero git diff. [...] If matching pairs are purged
> simultaneously, `reconciled` evaluates to `True`."

This is exactly right and it invalidates the *stated purpose* of the ledger.
The ledger was introduced last turn to convert "delete the bad run" from one
edit into two coordinated edits, and `paper_draft.md` says so in as many words.
That argument silently assumed the ledger was in the history — a removed line
is only a visible `-` in a diff if the file is tracked. On an untracked file,
"append-only by convention" is not a claim about anything. Nine turns of
provenance machinery sat on top of an artifact directory that git had never
seen. `test_registry_is_committed.py` now asserts every `runs/bench/task_*.json`
and the ledger are tracked; it is red as I write this and goes green when the
8-seed run lands and the records are committed.

**#7: clause 3 never checked that the verdict seeds arrived.**
`seeds_registered_but_missing` was always computed against `seeds_screen`
`[0,1,2]`, so an escalation started and abandoned halfway left clause 3 clean
— flattering. The required set is now the screen until a seed outside the
screen appears, at which point the run has embarked on the verdict set and owes
all of it. This cannot lower the bar: the escalated requirement is a superset.
Amendment 4.

**#6: the external probe re-checked the same 25 rows every time.** The sample
seed defaulted to 0, so of task 31's 415,112 rows the probe looked at the same
25 on every run and the other 415,087 could be edited downward — lowering the
human baseline — with the probe never looking. Flattering, and it is the flaw
that made a small *k* *exploitable* rather than merely weak. The sample seed is
now derived from the evaluation file's own sha256 XOR `--seed`: edit any row and
the digest changes and the whole sample re-draws, so an editor cannot know
which rows are safe. Coverage per run is unchanged (25 of ~10⁵, and the JSON
still records the undetected-tampering probabilities); what changes is that the
choice of checked rows is no longer predictable. Re-ran: 125 rows, 0 mismatches
under the new draw as well.

**#4: the gate fingerprint covered the test but not its inputs.** It hashed
`exact_sign_test_above`, `escalation_state` and `verdict` — so `load_bench`
could start skipping `.FAILED.json` files, or `reconcile_ledger` could return a
hard-coded `reconciled: True`, with the fingerprint still matching. Now covers
`load_bench`, `reconcile_ledger`, `tolerance_readings`, `joint_seed_event` and
`clean_seed_subset` as well. Amendment 5.

**#5: `"direction": "stricter"` is a JSON string.** Correct: loosen α to 0.10,
update the fingerprint, write "stricter", and every test passes. No test can
verify mathematical strictness in general. What a test *can* do is pin the
specific knobs a loosening would have to turn, so that turning one is a visible
diff in a test file rather than a one-character edit in a script. Pinned: α =
0.05, the full-verdict-seed requirement (n = 5, 6, 7 reject and are still not
called), ties as non-wins in the denominator, the one-sided 0.95× tolerance,
and EPS ≤ 1e-9 — plus the reverse direction, that a strict 8/8 still gives
p = 1/256 and is called, so the pins have not quietly made the gate
unreachable. Amendment 5.

The pattern across both adversaries is worth stating plainly, because it is the
paper's thesis arriving as a lived result: **every hole either of them found was
in the flattering direction, and not one was in the arithmetic.** They were in
what gets loaded, what gets counted when a field is absent, which rows get
checked, which seed set the bar refers to, and what a document is allowed to
say while a job is running. The numbers were never the attack surface.

### The most damaging finding of the turn came from asking the constructive question

`agy`, asked *only* "how would you make this clause pass more defensibly", and
told not to list defects, put one thing first and it is not about our number at
all:

> "On imbalanced datasets like blood-transfusion (majority share 76.2%) or kc1
> (84.5%), the 0.95 × median_run threshold is 72.5% and 80.9%. A dumb
> `DummyClassifier(strategy='prior')` or single-split decision tree clears the
> primary threshold on 4 of the 5 tasks without learning anything. Does passing
> this clause actually prove data science competence, or is the bar
> unfalsifiable?"

**It is right, and it is measured now.** `scripts/negative_control.py` runs two
frozen, deliberately incapable procedures through the *same* outer folds, the
same pooled statistic and the same pre-registered baselines as the agent:

| control | clears primary | clears strictest | all five |
|---|---|---|---|
| `prior` (`DummyClassifier(strategy="prior")`) | **4/5** | 3/5 | No |
| `stump` (`DecisionTreeClassifier(max_depth=3)`) | **4/5** | 4/5 | No |

and **4 of the 5 primary thresholds sit at or below the task's own
majority-class rate** — a fact checkable from the registry with no run at all,
which is what makes it an objection rather than a hypothesis. Only `kr-vs-kp`
(majority 0.5222, threshold 0.9120) discriminates on its own.

This is a statement about the **pre-registered target**, not about the agent,
and it is the one thing nine turns of provenance machinery could never have
surfaced: every check built so far asks whether our number is honest, and none
asked whether the bar was demanding. A KPI can be perfectly audited and
vacuous.

What survives it: the **joint** five-task criterion. Neither control clears all
five, because both collapse on the balanced task. So the defensible reading of
a PASS here is "clears five tasks including one where triviality fails" — not
"beat a human five times". That sentence is now in `RESULTS.md` next to the
result, and `tests/test_negative_control.py` asserts that if any control ever
does clear all five, the reassuring paragraph is **false and must be rewritten
rather than the test relaxed**.

**And the agent comes out of it worse than I expected.** Margins against an
untuned depth-3 tree: credit-g +0.0093, blood-transfusion **−0.0209**, kc2
+0.0010, kr-vs-kp +0.0921, kc1 +0.0094. One of the five is inside that task's
own seed range, so it is not a difference this repository can resolve, and on
blood-transfusion **the stump wins**. A six-family tournament with a random
search over the winner is buying a large margin on the one balanced task and
almost nothing over three splits of a tree on the four imbalanced ones. That is
a negative result about the agent, it is in the report, and it was not visible
from any comparison against the human baseline — because the human baseline is
*also* below the majority rate on those tasks.

The binding constraint on this KPI, stated properly for the first time: it is
**the task set**. Four of the five most-published CC18 tasks are small,
imbalanced and near-saturated, so their published medians sit close to
triviality and a ±5% band around such a median cannot separate competence from
its absence. My own §2.1 named the selection bias ("the most-run CC18 tasks are
the oldest and most famous, which are also the small and clean ones") and
treated it as a limitation of *coverage*. It is worse than that: it partly
dissolves the clause. The distinguishing prediction, for anyone who wants to
test this reading: on a task set selected for balance rather than popularity,
the gap between the agent and `stump` should widen by roughly the kr-vs-kp
margin (~0.09) rather than the imbalanced-task margin (~0.01), and the dummy
should clear nothing. That experiment is a *different* measurement and is not
this KPI, so it is named in `paper_draft.md` §7 rather than run and swapped in.

### And it was not bad luck: the selection rule caused it, at p = 0.0222

The negative-control finding raises an immediate question that needs **no runs
at all**, so it was answered before the 8-seed job finished
(`scripts/target_difficulty.py`, `runs/target_difficulty.json`). Both inputs
are published data — the baseline is OpenML's run history, the majority-class
rate is a dataset quality — so the analysis is blind to any accuracy of mine by
construction, and a test asserts the producing script has no executable
reference to `runs/bench`.

Of the **51 candidates** that passed the registered size filter, **36 have a
threshold above their majority-class rate**, i.e. a clause that triviality
cannot clear. Among the **5 the rule selected: 1**. A random five from the same
pool would be expected to contain **3.53**, and the exact lower-tail
hypergeometric probability of drawing at most as few as were drawn is
**p = 0.0222**.

So the weak task set is not sampling noise. **Ranking candidates by number of
published evaluations selects, at better than the 5% level, for tasks whose
±5% band is beneath triviality.** The rule was chosen for a good and still-true
reason — the median of a larger sample is better determined — and popularity on
OpenML-CC18 tracks the small famous imbalanced classics, whose published
medians sit near the majority rate. Spearman(popularity rank, headroom) = 0.196
is weak on its own, which is why the hypergeometric test rather than the
correlation is the statement.

The general form, which I think is the transferable finding of this whole
project: **I pre-registered a rule that was blind to my accuracy, and treated
that as the whole of the requirement.** A selection rule has to be blind to
your result *and* blind to the difficulty of the target, and only the first was
designed for. Every audit built over nine turns constrains the relation between
my claim and my evidence; none constrained the relation between the target and
the difficulty of the problem. A target can be pre-registered, hash-pinned,
externally re-fetched, amendment-logged, tie-corrected — and vacuous.

A successor rule that stays blind to accuracy is in the JSON: the five
most-published candidates whose threshold exceeds their majority rate —
`kr-vs-kp` (+0.3897), `qsar-biodeg` (+0.1371), `wdbc` (+0.2541), `diabetes`
(+0.0639), `phoneme` (+0.0053). It is **offered, not substituted**, and a test
asserts the registry's `selected_task_ids` still holds the original five:
swapping in a task set found after seeing which one made the point would be the
same error as choosing a baseline late. Ranks 6–15 were reserved by the
protocol for development and no run against them ever existed (`runs/dev` did
not exist), so they are also genuinely uninspected — which makes the successor
measurement cheap for whoever wants it and is recorded in `WEEKEND.md` as a
decision rather than taken unilaterally.

What would distinguish this reading from the obvious alternative — that the
agent is simply weak on imbalanced data: on the successor five, the agent's
margin over the depth-3 stump should look like the `kr-vs-kp` margin (≈0.09)
rather than the imbalanced-task margin (≈0.01), and `prior` should clear
nothing. If instead the margin stays ≈0.01 on balanced tasks too, the finding
is about the agent and not about the task set, and the tournament-plus-search
architecture is what needs attacking.

---

## 2026-09-10, turn 7 — the seed discipline was constraining the smaller noise source

### The measurement, from data already on disk

`scripts/fold_interval.py`, `runs/fold_interval.json`. No new runs: every run
record already carries `accuracy_folds`, the ten outer-fold accuracies. Seeds
are averaged per fold so that seed randomness drops out and what remains is
fold-to-fold variation; the interval is then over folds, `df = K-1`.

| task | mean rel. margin | sd over folds | 95% lower bound (naive) | (corrected) | non-inferior |
|---|---|---|---|---|---|
| 31 credit-g | +4.43% | 3.80% | +2.23% | +1.23% | yes |
| 10101 blood-transfusion | +0.76% | 3.04% | −1.00% | −1.80% | yes |
| 3913 kc2 | +1.28% | 4.09% | −1.09% | −2.18% | yes |
| 3 kr-vs-kp | +3.80% | 0.37% | +3.58% | +3.49% | yes |
| 3917 kc1 | +1.05% | 2.07% | −0.15% | −0.69% | yes |

Non-inferiority against the pre-registered −0.05 holds on all five under both
bounds. This is a **stronger** reading than the sign test in the sense that
matters — it uses the magnitudes the sign test throws away, and it is on the
axis of variation that generalisation actually depends on — and it enters **no
clause**. A test asserts `verdict()` cannot reference it, because a reading
invented after the fact that granted a clause would be exactly the move the
amendment ledger exists to forbid.

Two bounds are reported because the honest one is not obvious. Folds have
disjoint test sets and heavily overlapping training sets, so `s/√K` understates
the variance — the flattering direction. The corrected column uses the
Nadeau & Bengio (2003) inflation `(1/K + n_test/n_train)·s²`, 2.11× the
variance at 10 folds. **The assumption is stated rather than buried:** that
correction is derived for repeated random subsampling, not k-fold, and is used
here as a conservative adjustment for fold dependence. Both columns are shown
so a reader who rejects the adjustment can read the other. What is not done,
because it is the move that buys power by assuming away the dependence, is
pooling the 80 fold-by-seed scores as 80 independent observations — the thing
codex explicitly warned against.

### The finding, and it is about my own emphasis

**The fold-to-fold sd of the relative margin exceeds the seed-to-seed range by
1.6× to 4.4×.**

Last turn I rewrote the gate, wrote three amendments, added seven tests and
spent two hours of CPU escalating from 3 seeds to 8, all to constrain the
**seed** axis. That work is correct on its own terms — the old gate really was
biased in the flattering direction at small *n*, and the tie handling really
did inflate Type I error to 17.37%. But the axis it constrains is measurably
the smaller of the two, by a factor of up to four and a half. If I had
measured both noise sources before choosing where to spend the turn, I would
have built this section first: it costs no compute, it uses data that was
already on disk from the 3-seed screen, and it speaks to the uncertainty a
reader cares about.

The pattern is the same one the negative controls exposed and I want it stated
as one thing rather than two coincidences: **I have twice now built rigour on
the axis I was already looking at, rather than the axis that dominated.** Seed
noise over split noise; provenance of my claim over difficulty of the target.
The cheap diagnostic — *measure the sizes of the things you are choosing
between, before choosing* — was available in both cases and skipped in both.

The distinguishing prediction, so this is not just self-criticism: if fold
variance really dominates, then the seeds-3-to-7 escalation should have moved
the per-task means by less than the fold sd, i.e. by under ~2–4 percentage
points relative. Measured across seeds 0–2 → 0–6: task 31 moved 0.7563 →
0.7573 (+0.13% relative), 10101 0.7678 → 0.7692 (+0.18%), 3913 0.8372 →
0.8381 (+0.11%), 3 0.9969 → 0.9964 (−0.05%), 3917 0.8592 → 0.8605 (+0.15%).
All under 0.2% relative, against fold sds of 0.37%–4.09%. The prediction holds:
**adding five seeds moved every task by roughly an order of magnitude less than
its own fold-to-fold spread.**

### Infrastructure, measured rather than complained about

The 8-seed run is at 31 of 40 and slowing sharply: on task 10101 the per-fold
time went 13.0s, 14.3s, then **495.6s** — a 35× slowdown inside one task, with
`uptime` load average moving from ~125 to ~360 on 192 cores as other tracks
started. My process holds ~46 cores across 328 threads. I did **not** kill and
relaunch it with a smaller `ADS_N_JOBS`, for a reason worth recording: the
slowdown arrived *between two consecutive folds of the same configuration*, so
it is external load and not my thread count, and cutting my own share would
not recover a 35× factor on a saturated box.

That decision exposed a real gap in my own instrumentation, which I am
recording rather than fixing under time pressure: **the attempt ledger cannot
express an operator interruption.** Its vocabulary is `started` / `completed` /
`failed`, and a `SIGKILL` leaves a bare `started` — indistinguishable from the
deletion attack the ledger was built to catch. So "kill the slow cell and
re-run it" is currently unavailable to me *as an honest act*, even though
re-running a fixed `(task, seed)` cannot shop for a better number: the seed
determines the agent's randomness and the folds are the task's own, so the
re-run reproduces the same accuracy. The right shape, for whoever builds it: a
recorded `killed` event with a reason, plus a reconciliation rule requiring
every killed cell to have a later `completed` record and file — strictly more
information on the record, and no ability to hide a result, since the same
cell re-runs to the same number. Named in `WEEKEND.md` as work.
