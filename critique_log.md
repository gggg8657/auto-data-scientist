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
