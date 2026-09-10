# Results

**Status: RUNNING (not all clauses measured)**

Every number below was produced by a run in this repository and is regenerated from `runs/*.json` by `scripts/report.py`. Nothing is hand-typed.

## KPI, clause by clause

| # | clause | measured as | met |
|---|---|---|---|
| 1 | 공개 데이터셋 5개 | 5 registered, 0 measured | True |
| 2 | 사람 baseline ±5% 이내 자동도달 | primary reading: ours >= 0.95 x median_run, on all five | None |
| 3 | end-to-end 무개입 | 0 interventions logged; off-registry runs: False; incomplete: 0; failed attempts: 0; registered seeds missing: 0; duplicated seeds: 0; seeds outside the protocol: 0; distinct agent-source digests: 0; runs from a dirty agent tree: 0; ledger reconciled: [no ledger] | None |

## Our accuracy against the pre-registered baselines

| task | dataset | seeds | ours (pooled) | median_run | rel gap | primary (>=0.95x) | rel 2-sided | abs 2-sided | median_flow | vs flow | strictest reading | strictest value | vs strictest | families chosen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 0 | [not measured] | 0.7250 | [not measured] | [not measured] | [not measured] | [not measured] | 0.7290 | [not measured] | median_flow_best | 0.7310 | [not measured] | [not measured] |
| 10101 | blood-transfusion-service-center | 0 | [not measured] | 0.7634 | [not measured] | [not measured] | [not measured] | [not measured] | 0.7500 | [not measured] | median_uploader_best | 0.7741 | [not measured] | [not measured] |
| 3913 | kc2 | 0 | [not measured] | 0.8276 | [not measured] | [not measured] | [not measured] | [not measured] | 0.8257 | [not measured] | median_uploader_best | 0.8448 | [not measured] | [not measured] |
| 3 | kr-vs-kp | 0 | [not measured] | 0.9599 | [not measured] | [not measured] | [not measured] | [not measured] | 0.9585 | [not measured] | median_uploader_best | 0.9947 | [not measured] | [not measured] |
| 3917 | kc1 | 0 | [not measured] | 0.8516 | [not measured] | [not measured] | [not measured] | [not measured] | 0.8473 | [not measured] | median_uploader_best | 0.8615 | [not measured] | [not measured] |

`rel gap` is `(ours - median_run) / median_run`; positive means we are above the median published run. The three tolerance columns are the three readings fixed in `runs/baselines.json` before any run — all are shown so that none can be picked after the fact.

## Our own run-to-run spread

[not measured] — fewer than two seeds per task so far.

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

## Provenance

| field | value |
|---|---|
| status | RUNNING (not all clauses measured) |
| clause2_under_strictest_baseline | None |
| escalation_pending_tasks | [] |
| near_line_not_called_tasks | [] |
| n_interventions_total | 0 |
| off_registry_runs | False |
| incomplete_runs | [] |
| n_failed_attempts | 0 |
| failed_attempts | [] |
| seeds_present | {} |
| seeds_registered_but_missing | {} |
| seeds_duplicated | {} |
| seeds_not_in_registered_protocol | {} |
| distinct_registry_digests_across_runs | 0 |
| distinct_agent_source_digests_across_runs | 0 |
| runs_from_a_dirty_agent_tree | [] |
| ledger | {'ledger_present': False} |
| n_tasks_measured | 0 |
| n_tasks_registered | 5 |

