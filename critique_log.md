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
