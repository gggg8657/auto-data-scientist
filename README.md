# auto-data-scientist

An agent that, given an OpenML classification task and nothing else, profiles the
data, picks its own preprocessing, model family, inner validation scheme and
search budget, and reports its accuracy against a **human baseline that was
fixed before it ran**.

## KPI (verbatim from the portfolio board)

> **공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 · end-to-end 무개입**

Clause by clause, with how each is measured:

| # | clause | measured as | status |
|---|---|---|---|
| 1 | 공개 데이터셋 5개 | five OpenML-CC18 tasks, selected by a rule that cannot see our accuracy (below) | see `RESULTS.md` |
| 2 | 사람 baseline ±5% 이내 자동도달 | our pooled `predictive_accuracy` on the task's **own** estimation procedure vs the median of every published run on that same task | see `RESULTS.md` |
| 3 | end-to-end 무개입 | the agent emits a decision log per fold; a run with `n_interventions > 0` is a **failed run**, not a run to be edited | see `RESULTS.md` |

## The pre-registration

The way this KPI gets faked is by choosing the baseline after seeing the result.
So the baselines are fetched, committed and frozen by `scripts/fetch_baselines.py`
**before any agent run exists**, and everything below is fixed at that point.

**Task selection rule.** From OpenML-CC18 (study 99): keep tasks with
`n ≤ 20000` and `p ≤ 100` (a CPU budget — the agent runs a six-family tournament
plus a random search on each of ten folds); among those, take the five with the
most published evaluations, because the median of a larger sample is the
better-determined baseline. Neither criterion can see our accuracy.

**Human baseline, four readings, all fixed in advance.** OpenML stores every
`predictive_accuracy` a user ever uploaded for a task, computed under that task's
own estimation procedure — so a published run and our run are numbers about the
same splits. Every published evaluation is paged to **exhaustion**: two of these
five tasks have over 385,000 of them, and a capped fetch takes an oldest-first
prefix whose median is a different statistic (see `critique_log.md`, turn 2).

- `median_run` — median over **all published runs**. Primary. One prolific
  uploader's 5000-point sweep counts 5000 times.
- `median_flow` — median over **per-flow medians**, one number per published
  *method*. De-weights sweeps.
- `median_flow_best` — median over each flow's **best** run: every method at its
  best published configuration.
- `median_uploader_best` — median over each uploader's **best** run. The
  symmetric comparison, our selected model against their selected model.

`median_run` is the primary because the brief named it and it was registered
first; it is also the *weakest*, since it scores our selected model against the
distribution of all human trials, failures included. The strictest of the four
is recorded per task in `strictest_baseline` and its verdict is reported beside
the primary one. Which reading is hardest is an **empirical** fact per task, not
a mathematical one: only `median_flow_best ≥ median_flow` holds by construction,
and task 10101 is a live counterexample to the rest, with `median_flow_best`
0.7500 **below** `median_run` 0.7634.

**How demanding is that target? Less than it looks, and this is measured.**
On **four of the five registered tasks, `0.95 × median_run` sits at or below
the dataset's own majority-class rate**, so a `DummyClassifier(strategy=
"prior")` clears the primary clause there without using a single feature. Two
frozen negative controls — that dummy and an untuned depth-3 decision tree —
were run through the *same* outer folds and the same pooled statistic as the
agent, and each clears the primary reading on 4 of 5 tasks
(`runs/negative_control.json`).

Nor was that bad luck. Of the 51 candidates passing the size filter, 36 have a
threshold above their majority-class rate; the top-5-by-published-runs rule
selected **1** of them, against 3.53 expected under random selection — exact
hypergeometric **p = 0.0222** (`runs/target_difficulty.json`). Ranking by
popularity is defensible for determining a median, and it also selects for the
small famous imbalanced classics whose medians sit near triviality. **A
selection rule has to be blind to your result *and* blind to the difficulty of
the target; only the first was designed for.**

What survives is the **joint** five-task criterion, which neither control
clears because both collapse on the one balanced task. So a `PASS` on this KPI
should be read as *"clears five tasks including one where triviality fails"* —
not as *"beat a human five times"*. A successor task set chosen by a rule still
blind to our accuracy is recorded, and **offered rather than substituted**: the
registered five remain the measurement.

And the obvious successor rule is not good enough either, which was measured
rather than assumed. "Threshold above the majority-class rate" makes the
`prior` control fail **by construction** — that control's accuracy *is* the
majority rate — so it is circular as evidence. Run on the five it selects
(`runs/negative_control_successor.json`): `prior` clears 0 of 5, as guaranteed,
and the untuned depth-3 tree still clears **3 of 5**. Clearing the class-prior
floor does not make a threshold demanding; the floor that matters is a
*procedure* floor, and `scripts/falsifiability_floor.py` measures it over the
whole candidate pool.

`q75`, `q90` and `max_published` appear in the tables as **context, never as the
target**: they say how far the published frontier is above the median, which is
what a reader needs to judge how demanding the median is.

**Tolerance, three readings, all fixed in advance.** "±5% 이내" is ambiguous
between relative and absolute, and between one- and two-sided:

- **primary** `ours ≥ 0.95 × baseline` — one-sided relative. A reaching
  criterion; a two-sided rule would fail a run for *beating* the baseline by 6%.
- `|ours − baseline| / baseline ≤ 0.05` — two-sided relative.
- `|ours − baseline| ≤ 0.05` — two-sided absolute (percentage points).

All three are reported for every task. Picking whichever one passes, after the
fact, would be the same fraud as picking the baseline late.

<!-- BASELINES:BEGIN -->
### The five tasks and their pre-registered human baselines

Fetched by `scripts/fetch_baselines.py` on **2026-09-10T12:56:40+0000**, before any agent run existed. Selected from OpenML-CC18 (study 99) by: `NumberOfInstances <= 20000 and NumberOfFeatures <= 100`, then the top 5 of 51 by `n_published_runs, descending`.

| task | dataset | n | p | classes | published runs | flows | uploaders | **median_run** (primary) | median_flow | median_flow_best | median_uploader_best | strictest | q90 (context) | max (context) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 1000 | 21 | 2 | 415112 | 1022 | 119 | 0.7250 | 0.7290 | 0.7310 | 0.7030 | median_flow_best | 0.7670 | 0.7860 |
| 10101 | blood-transfusion-service-center | 748 | 5 | 2 | 385696 | 728 | 60 | 0.7634 | 0.7500 | 0.7500 | 0.7741 | median_uploader_best | 0.7781 | 0.8075 |
| 3913 | kc2 | 522 | 22 | 2 | 173611 | 580 | 37 | 0.8276 | 0.8257 | 0.8295 | 0.8448 | median_uploader_best | 0.8448 | 0.8697 |
| 3 | kr-vs-kp | 3196 | 37 | 2 | 173303 | 926 | 70 | 0.9599 | 0.9585 | 0.9634 | 0.9947 | median_uploader_best | 0.9944 | 0.9981 |
| 3917 | kc1 | 2109 | 22 | 2 | 158237 | 525 | 29 | 0.8516 | 0.8473 | 0.8487 | 0.8615 | median_uploader_best | 0.8625 | 0.8734 |

Metric is `predictive_accuracy` under each task's own estimation procedure, and it is the **size-weighted (pooled)** accuracy — measured, not assumed, in `runs/metric_check.json`. Raw evaluations are committed under `runs/evals/` with a sha256 per file in `runs/baselines.json`, and `tests/test_registry_frozen.py` recomputes every reading above from them, so a baseline cannot be edited without the tests failing.

`median_run` counts a 5000-point sweep 5000 times and scores our *selected* model against the distribution of *all* human trials, failures included — an asymmetry that flatters us. `median_flow_best` and `median_uploader_best` are the symmetric readings: their selected solution against ours. All four are reported for every task.
<!-- BASELINES:END -->

## What "무개입" is allowed to mean

The agent may not branch on the **identity** of a dataset — no task id, no
dataset name, anywhere in `ads/`. Rules keyed on *measured properties* — `n`,
`p/n`, cardinality, missingness, class imbalance — are the entire point; rules
keyed on which dataset it is would be the way this clause gets faked.

`tests/test_no_dataset_specific_logic.py` enforces this two ways, because the
obvious way does not work on its own. Dataset **names** are scanned in full, but
literal **task ids** only for ids ≥ 1000: two of the five registered ids are
`31` and `3`, and `ads/agent.py` legitimately contains
`max_leaf_nodes=[15, 31, 63]` and `INNER_FOLDS_LARGE = 3`, so a bare-integer
scan was failing on hyperparameters rather than on identity — a guard that is
always red tells you nothing. The property the scan stood in for is therefore
tested directly: renaming every column to an opaque label and reversing the
column order leaves the chosen family, the logged decisions and **100% of
predictions** unchanged. That closes the schema-fingerprint route; it cannot
rule out a rule keyed on feature *values*, and the test says so.

Every choice is written to a `DecisionLog` **with the profile quantity that drove
it**; `DecisionLog.record` raises on a decision carrying no evidence.

## Layout

```
ads/                package: profiling, decisions, the agent, evaluation
scripts/
  fetch_baselines.py   stage 1  — pre-register the five baselines (run first)
  verify_evals.py      stage 1b — re-fetch a digest-seeded sample of the
                                  baseline evidence from OpenML and compare
  target_difficulty.py stage 1c — how demanding is the target, across the pool?
                                  (blind to our runs; needs no run of ours)
  run_benchmark.py     stage 2  — run the agent on the five selected tasks
  negative_control.py  stage 2b — frozen incapable procedures, same folds:
                                  is the clause falsifiable at all?
  report.py            regenerate RESULTS.md and the tables above from runs/*.json
tests/
runs/                  the JSON record; every number in every document comes from here
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_baselines.py     # writes runs/baselines.json
.venv/bin/python scripts/verify_evals.py        # writes runs/evals_provenance.json
.venv/bin/python scripts/target_difficulty.py   # writes runs/target_difficulty.json
.venv/bin/python scripts/run_benchmark.py       # writes runs/bench/task_*.json
.venv/bin/python scripts/negative_control.py    # writes runs/negative_control.json
.venv/bin/python scripts/report.py              # regenerates RESULTS.md
```

`run_benchmark.py` with no flags runs the **pre-registered screen** (3 seeds,
from `run_protocol.seeds_screen`). Escalating to the 8-seed verdict set is an
explicit act — `--seeds 0 1 2 3 4 5 6 7` — and no task can be *called* on
fewer, because the exact sign test's p-value floor of `1/2^n` cannot reach
α = 0.05 at n < 5 and the full registered set is required to rule out optional
stopping.

No number in this repository is hand-typed. `scripts/report.py` is the only
thing that writes a number into a document, and it reads only `runs/*.json`.
