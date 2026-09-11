# Results

**Status: NOT MET as measured**

Every number below was produced by a run in this repository and is regenerated from `runs/*.json` by `scripts/report.py`. Nothing is hand-typed.

## KPI, clause by clause

| # | clause | measured as | met |
|---|---|---|---|
| 1 | 공개 데이터셋 5개 | 5 registered, 5 measured | True |
| 2 | 사람 baseline ±5% 이내 자동도달 | primary reading: ours >= 0.95 x median_run, on all five | True |
| 3 | end-to-end 무개입 | 0 interventions logged; off-registry runs: False; incomplete: 0; failed attempts: 0; partial records excluded: 0; runs missing a required field: 0; runs missing an agent digest: 0; registered seeds missing: 0; duplicated seeds: 0; seeds outside the protocol: 0; distinct agent-source digests: 1; runs from a dirty agent tree: 0; ledger reconciled: True | False |

## Is this clause falsifiable? The negative controls

This section comes before the accuracy tables on purpose. Every other check in this document asks whether *our* number is honest; this one asks whether the **target** is demanding, and the answer is only partly yes — so it changes how the tables below should be read.

**4 of 5 primary thresholds sit at or below the task's own majority-class rate**, so on those tasks the bar can be cleared by predicting the commonest label and nothing else. Two frozen, deliberately incapable procedures were run through the **same outer folds, the same pooled metric and the same pre-registered baselines** as the agent: `prior` (`DummyClassifier(strategy="prior")`, which ignores the features entirely) and `stump` (`DecisionTreeClassifier(max_depth=3)`, untuned). Neither was chosen by how it scored.

| control | task | dataset | control accuracy | majority-class rate | threshold (0.95x median_run) | vs primary | vs strictest | ours | ours - control |
|---|---|---|---|---|---|---|---|---|---|
| prior | 31 | credit-g | 0.7000 | 0.7000 | 0.6887 | CLEARS | CLEARS | 0.7582 | 0.0583 |
| prior | 10101 | blood-transfusion-service-center | 0.7620 | 0.7620 | 0.7252 | CLEARS | CLEARS | 0.7697 | 0.0077 |
| prior | 3913 | kc2 | 0.7950 | 0.7950 | 0.7862 | CLEARS | fails | 0.8374 | 0.0424 |
| prior | 3 | kr-vs-kp | 0.5222 | 0.5222 | 0.9120 | fails | fails | 0.9961 | 0.4739 |
| prior | 3917 | kc1 | 0.8454 | 0.8454 | 0.8090 | CLEARS | CLEARS | 0.8594 | 0.0140 |
| stump | 31 | credit-g | 0.7480 | 0.7000 | 0.6887 | CLEARS | CLEARS | 0.7582 | 0.0102 |
| stump | 10101 | blood-transfusion-service-center | 0.7901 | 0.7620 | 0.7252 | CLEARS | CLEARS | 0.7697 | -0.0204 |
| stump | 3913 | kc2 | 0.8372 | 0.7950 | 0.7862 | CLEARS | CLEARS | 0.8374 | 0.0002 |
| stump | 3 | kr-vs-kp | 0.9043 | 0.5222 | 0.9120 | fails | fails | 0.9961 | 0.0918 |
| stump | 3917 | kc1 | 0.8511 | 0.8454 | 0.8090 | CLEARS | CLEARS | 0.8594 | 0.0083 |

**prior**: clears the primary reading on 4/5 tasks, the strictest on 3/5, all five: False.  **stump**: clears the primary reading on 4/5 tasks, the strictest on 4/5, all five: False.

### Was the weak task set bad luck, or did the rule cause it?

Answerable with no runs at all, and blind to our accuracy by construction: both inputs are published data, the baseline from OpenML's run history and the majority-class rate from a dataset quality.

Of the **51 candidates** that passed the registered size filter, **36** have a threshold *above* their majority-class rate. Among the 5 the rule actually selected: **1**. A random five from the same pool would be expected to contain **3.529**, and the exact hypergeometric probability of drawing at most as few as were drawn is **p = 0.0222**.

**That p-value was the wrong test, and it is withdrawn as evidence.** It asks whether *this draw of five* is unusual, and at n=5 against the pool's base rate it can reject only on the most extreme possible draw — `rejectable_draws` = [0] out of 0..5 under the procedure floor. A test that can fire in one outcome out of six is not evidence about a rule, and quoting it as such was an error. Both tests are now reported under both floors, and the one the question actually calls for is the **rule** test: Spearman over all candidates, n much larger than 5, with a one-sided permutation null.

| floor | base rate | selected | expected | draw p (n=5) | draws it can reject on | rule n | rule rho | rule p (permutation) |
|---|---|---|---|---|---|---|---|---|
| `procedure (clears every control)` | 25/51 | 1/5 | 2.451 | 0.1871 | [0] | 51 | +0.2807 | 0.0236 |
| `majority-class rate` | 36/51 | 1/5 | 3.529 | 0.0222 | [0, 1] | 51 | +0.1959 | 0.0847 |

Read the diagonal. The **draw** test is significant only under the weaker floor; the **rule** test is significant only under the stricter one. The conclusion rests on the cell that is both the appropriate test and the stricter floor: rho = +0.2807, p = 0.0236. Positive rho means headroom grows with rank number, i.e. **the more published a task is, the less its threshold clears a depth-3 tree.** The effect is modest and it is real. Ranking candidates by number of published evaluations — chosen because the median of a larger sample is better determined, which is true and is still true — also selects for the small, famous, imbalanced classics whose medians sit near triviality. The registered rule was blind to our accuracy and it was **not** blind to the difficulty of the target, and only the first of those was designed for.

**A successor criterion, and the measurement that shows it is not enough on its own.** The obvious rule — the five most-published candidates whose threshold exceeds their majority-class rate — is blind to our accuracy, but it is *circular* as evidence: it makes the `prior` control fail by construction, since that control's accuracy **is** the majority rate. Measured on those five (`runs/negative_control_successor.json`): `prior` clears 0 of 5, as guaranteed, and the untuned depth-3 tree still clears **3 of 5**. So clearing the class-prior floor does not make a threshold demanding; the floor that matters is a *procedure* floor. The candidates are — `kr-vs-kp` (rank 4, headroom +0.3897), `qsar-biodeg` (rank 6, headroom +0.1371), `wdbc` (rank 7, headroom +0.2541), `diabetes` (rank 10, headroom +0.0639), `phoneme` (rank 12, headroom +0.0053). That is a **different measurement** and is offered as the successor, never as this one. Swapping it in now would be choosing a task set after seeing which one made the point.

**What this does to the claim.** The per-task clause is weak on the four imbalanced tasks and only `kr-vs-kp` discriminates on its own — a fact about the pre-registered target, not about the agent, and one that no amount of provenance machinery would have surfaced. What survives it is the **joint** criterion: neither control clears all five, because both collapse on the one balanced task. So a PASS here should be read as "clears five tasks including one where triviality fails", not as "beat a human five times".

**The uncomfortable rows, named rather than left in the table.** Against the untuned depth-3 tree the agent's margin is credit-g +0.0102, blood-transfusion-service-center -0.0204, kc2 +0.0002, kr-vs-kp +0.0918, kc1 +0.0083. 3 of the five sits inside the task's own seed range, so it is not a difference this repository can resolve, and 1 is negative — there the stump is **ahead**. The agent's six-family tournament plus random search is buying a large margin on the balanced task and, on the imbalanced ones, very little over three splits of a tree. That is a finding about the agent and it is not flattering; it is here because it is what the runs say.

## Our accuracy against the pre-registered baselines

| task | dataset | seeds | ours (pooled) | median_run | rel gap | primary (>=0.95x) | rel 2-sided | abs 2-sided | median_flow | vs flow | strictest reading | strictest value | vs strictest | families chosen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 8 | 0.7582 | 0.7250 | +4.59% | PASS | PASS | PASS | 0.7290 | PASS | median_flow_best | 0.7310 | PASS | extra,hgb,logreg,rf |
| 10101 | blood-transfusion-service-center | 8 | 0.7697 | 0.7634 | +0.83% | PASS | PASS | PASS | 0.7500 | PASS | median_uploader_best | 0.7741 | PASS | hgb,knn,logreg |
| 3913 | kc2 | 8 | 0.8374 | 0.8276 | +1.19% | PASS | PASS | PASS | 0.8257 | PASS | median_uploader_best | 0.8448 | PASS | extra,hgb,knn,logreg,rf |
| 3 | kr-vs-kp | 8 | 0.9961 | 0.9599 | +3.76% | PASS | PASS | PASS | 0.9585 | PASS | median_uploader_best | 0.9947 | PASS | extra,hgb,rf |
| 3917 | kc1 | 8 | 0.8594 | 0.8516 | +0.92% | PASS | PASS | PASS | 0.8473 | PASS | median_uploader_best | 0.8615 | PASS | extra,hgb,logreg,rf |

`rel gap` is `(ours - median_run) / median_run`; positive means we are above the median published run. The three tolerance columns are the three readings fixed in `runs/baselines.json` before any run — all are shown so that none can be picked after the fact.

## The gate: the pre-registered exact test, per task

A task is **called** only when the one-sided exact sign test of `H0: median over seeds <= 0.95 x baseline` rejects at alpha = 0.05. The p-value floor is `1/2^n`, so a 3-seed screen cannot call a task in either direction and the status cannot read PASS off one. This gate replaced `margin > observed seed range` on 2026-09-10: that condition is *easier* to satisfy the fewer seeds you run (E[range] is 1.69 sigma at n=3 against 2.85 sigma at n=8, and it sits in the denominator), so it skipped the test on all five tasks of the 3-seed screen and passed the clause on point estimates. The margin reading is kept in the spread table as a diagnostic.

| task | dataset | seeds | threshold (primary) | k/n above | p | verdict (primary) | k/n above (strictest) | p (strictest) | verdict (strictest) |
|---|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 8 | 0.6887 | 8/8 | 0.0039 | CALLED | 8/8 | 0.0039 | CALLED |
| 10101 | blood-transfusion-service-center | 8 | 0.7252 | 8/8 | 0.0039 | CALLED | 8/8 | 0.0039 | CALLED |
| 3913 | kc2 | 8 | 0.7862 | 8/8 | 0.0039 | CALLED | 8/8 | 0.0039 | CALLED |
| 3 | kr-vs-kp | 8 | 0.9120 | 8/8 | 0.0039 | CALLED | 8/8 | 0.0039 | CALLED |
| 3917 | kc1 | 8 | 0.8090 | 8/8 | 0.0039 | CALLED | 8/8 | 0.0039 | CALLED |

## The reading that owes nothing to the pre-inspected seeds

The gate was amended after the 3-seed screen had been read (`runs/protocol_amendments.json`, amendments 1-3). All three are strictly stricter and #1 took the status from `PASS` to `RUNNING` on identical run data, so the usual objection to a post-hoc rule — that it was tuned to produce a pass — does not apply. But **seeds 0, 1, 2 had been inspected** when the rule changed, and a reader is entitled to a reading that owes nothing to them.

Below is the same exact test restricted to seeds **3, 4, 5, 6, 7** — the seeds that had not been run when the amendment was made, named *in* the amendment, so this is a fixed pre-specified subset and not one chosen after seeing outcomes. It is reported whichever way it comes out. At n=5, 5 of 5 above the line gives p = 1/32 = 0.03125, which clears alpha = 0.05, so this subset can reach a verdict on its own.

| task | dataset | clean seeds present | k/n above | p | verdict (primary) | k/n above (strictest) | p (strictest) | verdict (strictest) |
|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 5 | 5/5 | 0.03125 | CALLED | 5/5 | 0.03125 | CALLED |
| 10101 | blood-transfusion-service-center | 5 | 5/5 | 0.03125 | CALLED | 5/5 | 0.03125 | CALLED |
| 3913 | kc2 | 5 | 5/5 | 0.03125 | CALLED | 5/5 | 0.03125 | CALLED |
| 3 | kr-vs-kp | 5 | 5/5 | 0.03125 | CALLED | 5/5 | 0.03125 | CALLED |
| 3917 | kc1 | 5 | 5/5 | 0.03125 | CALLED | 5/5 | 0.03125 | CALLED |

This is also why no further disjoint seed set was run. More seeds would shrink seed noise, which across the five tasks is 0.0025–0.0230 against margins to the line of 0.0445–0.0841; they would do nothing about the uncertainty that actually binds, which is that each task is **one** fixed dataset with one fixed set of folds.

## Additional reading: the folds, not the seeds

The gate above is a sign test over **seeds**, which re-draw the agent's own randomness on the *same* examples and the same outer folds. So it constrains algorithmic variance and says nothing directly about split variance. This section is the other axis, computed from the ten outer-fold accuracies already in every run record — no new runs. It is an **additional reading and enters no clause**; a test asserts `verdict()` cannot see it.

Per fold, the relative margin `d_k = (ours_k - baseline) / baseline`, and a one-sided 95% lower bound on its mean against the pre-registered `-0.05`. The tolerance is the registry's; only the uncertainty model is new here.

| task | dataset | seeds averaged | folds | mean relative margin | sd over folds | 95% lower bound (naive) | 95% lower bound (corrected) | non-inferior? |
|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 7 | 10 | +4.43% | 3.80% | +2.23% | +1.23% | PASS |
| 10101 | blood-transfusion-service-center | 6 | 10 | +0.76% | 3.04% | -1.00% | -1.80% | PASS |
| 3913 | kc2 | 6 | 10 | +1.28% | 4.09% | -1.09% | -2.18% | PASS |
| 3 | kr-vs-kp | 6 | 10 | +3.80% | 0.37% | +3.58% | +3.49% | PASS |
| 3917 | kc1 | 6 | 10 | +1.05% | 2.07% | -0.15% | -0.69% | PASS |

**Two bounds because the honest one is not obvious.** The folds have disjoint test sets and heavily overlapping training sets, so `s/sqrt(K)` understates the variance — the flattering direction. The corrected column uses the Nadeau & Bengio (2003) inflation `(1/K + n_test/n_train) s^2`, which at 10 folds multiplies the variance by 2.11x. That correction is *derived* for repeated random subsampling and is applied here as a conservative adjustment for fold dependence; both columns are shown so a reader who rejects the adjustment can read the other. The 80 fold-by-seed scores are **not** pooled as 80 independent observations, which is the move that would buy power by assuming away the dependence.

Non-inferiority holds on all 5 tasks under the naive bound (True) and under the corrected one (True).

**And this is the more important number in the section:** the fold-to-fold sd of the relative margin exceeds the seed-to-seed range by 1.6x to 4.4x. The seed-count discipline this repository spent a turn enforcing is correct on its own terms and was constraining the **smaller** of the two noise sources. Both are now reported; neither replaces the other, and the gate stays the pre-registered one.

## Additional reading: does one unattended run clear all five?

The clause-2 gate above is per task, which is what the protocol registered and is the right test for it (requiring all five to reject is an intersection-union test, so five tasks at alpha = 0.05 need no multiplicity correction). But a per-task verdict does not say that a *single* autonomous run gets all five: five tasks each failing on a different seed would pass every per-task test and never once produce a clean sweep. So this table asks the question a reader of "end-to-end 무개입" actually has, and it is reported beside the gate rather than instead of it.

| seed | cleared all five (median_run) | cleared all five (strictest) | tasks below the line | tasks not yet run |
|---|---|---|---|---|
| 0 | yes | yes | - | - |
| 1 | yes | yes | - | - |
| 2 | yes | yes | - | - |
| 3 | yes | yes | - | - |
| 4 | yes | yes | - | - |
| 5 | yes | yes | - | - |
| 6 | yes | yes | - | - |
| 7 | yes | yes | - | - |

**8 of 8 complete seeds** cleared all five against `median_run` (8 of 8 against each task's strictest reading). Exact sign test on the joint event: k=8/8, p=0.0039.

**What eight seeds do and do not measure.** They re-draw the agent's own randomness — inner-CV shuffle, selection subsample, random search — on the *same* examples and the same outer folds. So they estimate algorithmic variance conditional on this data, not uncertainty about new data, and the p-values above should be read that way (codex, 2026-09-10). The 80 fold scores are **not** treated as 80 independent observations: overlapping CV training sets are dependent and pooling them as independent would understate the variance (Bengio & Grandvalet, JMLR 2004).

`primary_pass` in the table above is computed from the seed *mean* while the gate tests the *median*. Both are required. That is an extra empirical guardrail rather than a second hypothesis test, and it can only make the clause harder: seven slightly-clearing seeds and one disastrous one can reject the median null and still fail the mean check.

## Our own run-to-run spread

An effect smaller than this noise floor is not an effect. Each seed re-draws the agent's inner CV shuffle, its selection subsample and its random search; the outer folds are the task's own and are identical across seeds.

| task | dataset | seeds | min | max | range | sd | range on the 3 screen seeds | range grew by | \|gap to baseline\| |
|---|---|---|---|---|---|---|---|---|---|
| 31 | credit-g | 8 | 0.7540 | 0.7660 | 0.0120 | 0.0040 | 0.0040 | 3.00x | 4.59% |
| 10101 | blood-transfusion-service-center | 8 | 0.7660 | 0.7741 | 0.0080 | 0.0027 | 0.0053 | 1.50x | 0.83% |
| 3913 | kc2 | 8 | 0.8238 | 0.8467 | 0.0230 | 0.0063 | 0.0077 | 3.00x | 1.19% |
| 3 | kr-vs-kp | 8 | 0.9950 | 0.9975 | 0.0025 | 0.0008 | 0.0013 | 2.00x | 3.76% |
| 3917 | kc1 | 8 | 0.8540 | 0.8634 | 0.0095 | 0.0029 | 0.0038 | 2.50x | 0.92% |

The last two columns measure, on this repository's own runs, the bias that got the gate replaced. A 3-seed range *understates* the spread: the expected range of n i.i.d. draws is 1.693 sigma at n=3 and 2.847 sigma at n=8, a ratio of **1.68x**, and the old gate put that understated quantity in the denominator of `margin > range`. So the fewer seeds you ran, the more comfortably clear of the line every task looked. The measured growth beside it is what actually happened when seeds were added.

## A different measurement: the successor task set

**This is not the KPI.** The KPI is the five registered tasks above; a task set chosen after seeing that the registered one was weak may only ever be reported as a *separate* measurement, and substituting it would be the same error as choosing a baseline late. It lives in `runs/dev/`, and none of the clause accounting above can see it.

The rule, still blind to our accuracy: the five most-published candidates whose `0.95 x median_run` **exceeds** their majority-class rate. Their targets needed no new fetch — all 51 candidates were frozen in `runs/baselines.json` before any run existed, so these baselines are as pre-registered as the others. Four of the five were reserved by the protocol for development and the agent had **never been run on any of them**, so they are uninspected; the agent is unmodified.

| task | dataset | rank | seeds | ours (pooled) | median_run | rel gap | vs primary | vs strictest | threshold headroom over majority | k/n above | p | verdict | stump | does this row discriminate? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | kr-vs-kp | 4 | 3 | 0.9969 | 0.9599 | +3.85% | PASS | PASS | 0.3897 | 3/3 | 0.1250 | screen only | 0.9043 | stump fails — row discriminates |
| 9957 | qsar-biodeg | 6 | 3 | 0.8664 | 0.8417 | +2.93% | PASS | PASS | 0.1371 | 3/3 | 0.1250 | screen only | 0.7905 | stump fails — row discriminates |
| 9946 | wdbc | 7 | 3 | 0.9754 | 0.9279 | +5.11% | PASS | PASS | 0.2541 | 3/3 | 0.1250 | screen only | 0.9262 | stump CLEARS — row does not discriminate |
| 37 | diabetes | 10 | 3 | 0.7708 | 0.7526 | +2.42% | PASS | PASS | 0.0639 | 3/3 | 0.1250 | screen only | 0.7409 | stump CLEARS — row does not discriminate |
| 9952 | phoneme | 12 | 3 | 0.9165 | 0.7493 | +22.33% | PASS | PASS | 0.0053 | 3/3 | 0.1250 | screen only | 0.7685 | stump CLEARS — row does not discriminate |

**3 seeds: screen, not verdict.** The same gate applies — no task is called under the registered verdict seed count, so nothing here is a called result and none of it is a headline. It is reported because it is the experiment the falsifiability finding demands, and because it tests the alternative explanation: if the agent's margin over a depth-3 stump stays near its imbalanced-task value on these balanced tasks rather than widening, the finding is about the agent and not about the task set.

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
| status | NOT MET as measured |
| clause2_under_strictest_baseline | True |
| leakage_probe_clean | True |
| leakage_probe_verdict | NO_LEAKAGE_DETECTED |
| leakage_probe_tasks | [3, 31, 3913, 3917, 10101] |
| leakage_probe_agent_digest | 14ac971659faff242877908195d42c5bc0530ed1b5a855daeef31cd3b298b2ac |
| leakage_probe_stale_vs_runs | False |
| leakage_probeable_tasks | [3, 31, 3913, 3917, 10101] |
| leakage_min_folds_for_detection | {'31': 1, '3913': 6, '3': 1} |
| leakage_unprobeable_tasks | [] |
| leakage_tasks_cleared | [3, 31, 3913, 3917, 10101] |
| leakage_probeable_but_unprobed | [] |
| leakage_registered_tasks_unaccounted_for | [] |
| leakage_tasks_resolved_by_rank_instrument | [3917, 10101] |
| leakage_records_merged | 2 |
| clause3_conventional_agent_chose_everything | True |
| clause3_strict_no_operator_touched_any_cell | False |
| operator_touched_cells | [[10101, 6]] |
| runs_whose_accuracy_does_not_recompute | [] |
| runs_missing_the_dirty_tree_flag | [] |
| registry_digest_matches_live_baselines_file | True |
| live_baselines_sha256 | 6731b733f4e84696290431ba01ae227bc3168371b6dafa30533c8a035db0ecac |
| escalation_pending_tasks | [] |
| near_line_not_called_tasks | [] |
| n_interventions_total | 0 |
| runs_missing_a_required_field | [] |
| runs_missing_an_agent_digest | [] |
| ledger_present_and_reconciled | True |
| off_registry_runs | False |
| incomplete_runs | [] |
| n_failed_attempts | 0 |
| failed_attempts | [] |
| n_partial_records_excluded_from_the_average | 0 |
| partial_records | [] |
| seeds_present | {'10101': [0, 1, 2, 3, 4, 5, 6, 7], '31': [0, 1, 2, 3, 4, 5, 6, 7], '3913': [0, 1, 2, 3, 4, 5, 6, 7], '3917': [0, 1, 2, 3, 4, 5, 6, 7], '3': [0, 1, 2, 3, 4, 5, 6, 7]} |
| seed_set_required_of_this_run | [0, 1, 2, 3, 4, 5, 6, 7] |
| embarked_on_the_verdict_seed_set | True |
| seeds_registered_but_missing | {} |
| seeds_duplicated | {} |
| seeds_not_in_registered_protocol | {} |
| distinct_registry_digests_across_runs | 1 |
| distinct_agent_source_digests_across_runs | 1 |
| runs_from_a_dirty_agent_tree | [] |
| ledger | {'ledger_present': True, 'role_reconciled': 'confirmatory', 'n_events': 82, 'n_events_in_ledger_all_roles': 138, 'n_started': 40, 'n_completed': 40, 'n_failed': 1, 'attempts_started_but_unresolved': [], 'cells_with_more_starts_than_terminal_records': [], 'cells_started_more_than_once': [[10101, 6, 2]], 'n_killed_events': 1, 'operator_touched_cells': [(10101, 6)], 'completed_but_missing_from_disk': [], 'on_disk_but_not_in_ledger': [], 'failed_attempts_with_records': 0, 'reconciled': True} |
| n_tasks_measured | 5 |
| n_tasks_registered | 5 |

