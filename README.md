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

**Human baseline, two readings, both fixed in advance.** OpenML stores every
`predictive_accuracy` a user ever uploaded for a task, computed under that task's
own estimation procedure — so a published run and our run are numbers about the
same splits.

- `median_run` — median over **all published runs**. Primary. One prolific
  uploader's 5000-point sweep counts 5000 times.
- `median_flow` — median over **per-flow medians**, one number per published
  *method*. De-weights sweeps.

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
*(`runs/baselines.json` not yet generated — run `python scripts/fetch_baselines.py`.)*
<!-- BASELINES:END -->

## What "무개입" is allowed to mean

The agent may not branch on the **identity** of a dataset — no task id, no
dataset name, anywhere in `ads/`. `tests/test_no_dataset_specific_logic.py`
enforces that by searching the package for the benchmark's task ids and dataset
names. Rules keyed on *measured properties* — `n`, `p/n`, cardinality,
missingness, class imbalance — are the entire point; rules keyed on which
dataset it is would be the way this clause gets faked.

Every choice is written to a `DecisionLog` **with the profile quantity that drove
it**; `DecisionLog.record` raises on a decision carrying no evidence.

## Layout

```
ads/                package: profiling, decisions, the agent, evaluation
scripts/
  fetch_baselines.py   stage 1 — pre-register the five baselines (run first)
  run_benchmark.py     stage 2 — run the agent on the five selected tasks
  report.py            regenerate RESULTS.md and the table above from runs/*.json
tests/
runs/                  the JSON record; every number in every document comes from here
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_baselines.py     # writes runs/baselines.json
.venv/bin/python scripts/run_benchmark.py       # writes runs/bench/task_*.json
.venv/bin/python scripts/report.py              # regenerates RESULTS.md
```

No number in this repository is hand-typed. `scripts/report.py` is the only
thing that writes a number into a document, and it reads only `runs/*.json`.
