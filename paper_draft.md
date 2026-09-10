# An autonomous data scientist, and what it takes to believe its score

**Status.** Draft. Accuracy numbers live in `RESULTS.md` and are regenerated
from `runs/*.json` by `scripts/report.py`; none is reproduced here by hand.
Numbers quoted in this document are limited to measurements of the *protocol*
(aggregation checks, the noise floor, the exploit demonstrations), each of which
has a committed run JSON or test behind it.

---

## 1. The claim, and why the interesting part is not the model

The KPI is: *공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 ·
end-to-end 무개입* — reach within 5% of a human baseline on five public
datasets, end to end, with no intervention.

The modelling half of this is not hard, and saying so is the honest starting
point. A six-family tournament (dummy, HistGradientBoosting, random forest,
extra trees, logistic regression, k-NN) with a small random search over the
winner, on classic OpenML-CC18 tabular tasks, is a solved recipe. If the
result were simply "a competent AutoML pipeline lands near the median published
run", the paper would be a footnote.

What is hard, and what this repository is actually about, is **the second half
of the sentence**: making the number mean what it says. Two claims have to
survive contact with an adversary:

1. *±5% of a human baseline* — which baseline, measured how, fixed when?
2. *end-to-end 무개입* — autonomous by what evidence, and how would anyone
   know if it were not?

Both are claims about *provenance*, not about accuracy. Our finding is that
almost all of the available failure modes live there, and that they are
cheap to exploit and cheap to close — but only if they are closed **before**
the first result exists, because afterwards every fix is indistinguishable
from a fix chosen to help.

## 2. Fixing the target before there is a result

### 2.1 Selection

From OpenML-CC18 (study 99) we keep tasks with `n ≤ 20000` and `p ≤ 100` — a
CPU budget, since the agent runs a tournament plus a search on each of ten
outer folds — and take the five with the most published evaluations, because
the median of a larger sample is a better-determined baseline. Neither
criterion can see our accuracy.

This selection is biased and we name the bias rather than removing it: the
most-run CC18 tasks are the oldest and most famous, which are also the small
and clean ones. Selecting for difficulty instead would be choosing tasks to
make a point. A harder five-task set is a *different* measurement, not a
better one.

### 2.2 Four readings of "human baseline"

OpenML stores every `predictive_accuracy` any user ever uploaded for a task,
computed under that task's own estimation procedure — so a published run and
our run are numbers about the same splits. That yields more than one defensible
median, and the choice among them is exactly where a target gets chosen to fit
a result. So all four are fixed in advance and all four are reported:

- `median_run` — over all published runs. **Primary**, because the brief named
  it. Also the weakest: it scores our *selected* model against the distribution
  of *all* human trials, failures and default-parameter sweeps included, and one
  prolific uploader's 5000-point sweep counts 5000 times.
- `median_flow` — over per-flow medians; one number per published method.
- `median_flow_best` — over each flow's best run: every method at its best
  published configuration.
- `median_uploader_best` — over each uploader's best run. The symmetric
  comparison: their selected solution against ours.

An early version of our own documentation asserted that the two "best" readings
are "strictly harder" than `median_run`. **That is false**, and an adversary
caught it: only `median_flow_best ≥ median_flow` holds by construction, because
`median_run` is weighted by how often each flow was submitted. Our own registry
contains the counterexample — one of the five tasks has `median_flow_best`
*below* `median_run`, since de-truncating its evaluation history revealed many
weak flows. Which reading is strictest is therefore an **empirical fact per
task**, recorded in `strictest_baseline`, and the verdict under the strictest
reading is reported beside the primary one rather than computed and ignored.

### 2.3 Three readings of "±5%"

"±5% 이내" is ambiguous between relative and absolute and between one- and
two-sided. The primary is `ours ≥ 0.95 × baseline`: one-sided, because
*도달* ("reach") is a reaching criterion and a two-sided rule would fail a run
for *beating* the baseline by 6%. The two-sided relative and two-sided absolute
readings are reported alongside. Choosing whichever passes, afterwards, would be
the same fraud as choosing the baseline late.

### 2.4 The comparability check, which gates everything else

Evaluating on the task's own folds is necessary for comparability and not
sufficient. OpenML's server publishes one scalar per run, and it matters
whether that scalar is the **pooled** accuracy over all folds' predictions or
the **unweighted mean** of per-fold accuracies. Where folds are unequal these
are different statistics, and comparing one against the other while calling it
one number would invalidate every row in the paper.

This is settled by measurement, not assumption. Joining published per-fold
score vectors against the server's scalar on six tasks — five with unequal
folds, therefore able to discriminate — the pooled reading reproduces the
server value to ~1e-16 while the unweighted mean is off by up to 1.5e-4. All
five discriminating tasks agree. `accuracy_pooled` is therefore the comparable
column (`runs/metric_check.json`). The sixth task has equal folds and is
reported as *non-discriminating* rather than as agreement — a distinction that
matters, because counting it as agreement would have inflated the apparent
evidence by 20%.

### 2.5 A target is not frozen until it is committed, and ours was not

Our first pre-registration commit claimed the baselines were "fetched,
committed and frozen ... before any agent run exists". Two of those three were
true. The registry file was written 33 minutes *after* that commit and was not
tracked by git at all — so nothing prevented it from being regenerated later,
against a result, with no diff for a reviewer to see. A test now asserts that
the registry is tracked, and that once any run exists it is unmodified relative
to `HEAD`.

## 3. The defect that justifies the whole approach

The registry's paging was capped at 300,000 evaluations. The two
highest-ranked tasks — the only two of 51 candidates — hit that cap exactly,
so their "median over all published runs" was in fact the median over an
**oldest-first prefix** ordered by server-assigned `run_id`. Their true
histories hold over 385,000 evaluations each; the capped medians covered
roughly three-quarters of each population, drawn from the oldest end. This is
precisely the bias our own protocol had cited as its reason to page to
exhaustion.

Re-paging to exhaustion is a legitimate fix and not a moved goalpost, for three
reasons that we think are the general test for this situation:

1. **No result existed.** The change cannot have been steered toward our
   accuracy because there was no accuracy.
2. **It moves toward the registered protocol**, not away from it — the protocol
   already said "to exhaustion".
3. **The selection rule is provably stable under it.** Only those two tasks
   were truncated, both at ≥300,000, while rank 3 has 173,611 evaluations. De-
   truncating can only raise counts, so the top five cannot change: two
   baselines are corrected, no task is swapped. Had a *non-selected* task been
   truncated, the ranking itself would have been in question and the honest
   response would have been a full re-fetch of all 51.

And the fix did **not** uniformly help us, which is the part worth recording.
We predicted in writing, before the re-fetch landed, that both affected medians
would rise — targets getting harder. One did, by a small amount. The other's
per-flow median *fell*, by more, because de-truncation more than doubled the
number of visible flows on that task and the newly-visible ones are weak. So
one target moved in our favour by more than the other moved against us. A fix
that happens to help has to be reported as one; the superseded readings are
kept on disk and the deltas are in the commit message and `RESULTS.md`.

## 4. "무개입": autonomy as an enforced property

The agent sees one training fold and nothing else. It profiles the data and
chooses its encoding, its inner-CV scheme, its candidate families and its
search budget from *measured properties* — `n`, `p/n`, cardinality,
missingness, class imbalance. Every choice is written to a decision log
together with the profile quantity that drove it, and the log refuses a
decision carrying no evidence. An intervention is counted, never removed: a run
with `n_interventions > 0` is a failed run rather than a run to be edited.

Rules keyed on measured properties are the entire point; rules keyed on which
dataset it is would be how this clause gets faked. Enforcing that turns out to
be subtler than it looks.

Our first enforcement scanned the package for the registered task ids as string
literals. Two of the five ids are `31` and `3`, and the agent legitimately
contains `max_leaf_nodes=[15, 31, 63]` (31 is scikit-learn's default) and an
inner-fold count of 3. **The guard had been failing on hyperparameters since it
was written** — always red, therefore carrying no information. Literal scanning
is now restricted to ids where a match is not plausibly a hyperparameter, and
the property the scan stood in for is tested directly: relabelling every column
to an opaque name and reversing the column order leaves the chosen family, the
logged decisions and **100% of predictions** unchanged. That closes the
schema-fingerprint route. It does not rule out a rule keyed on feature
*values*, and the test says so rather than implying a stronger guarantee.

## 5. The adversary, and the question to ask it

We ran an independent CLI agent against the design **before any result
existed**, and the phrasing of the question changed the answer completely.
Asked "what is wrong with this code", such critics return defects. Asked
**"how would you fake this KPI without editing a number by hand and without
failing any existing test?"**, it returned eight ranked attacks and the
following diagnosis, which we think is the correct general statement of what
consistency checks buy:

> "I would attack the evidence-selection process, not the arithmetic. The
> design verifies that retained artifacts agree with one another; it does not
> establish that they represent an untouched, prospectively executed
> experiment."

Four attacks were real and are closed. One we had found independently the same
day.

**Duplicated seeds.** The reported accuracy was a mean over run *files*, while
completeness was checked against the *set* of seed values in those files. A set
cannot see a duplicate, so copying the best-scoring seed's record to a second
matching filename reweights the mean with no number edited and no test failing.
On a three-seed screen this moved the mean by nearly a full percentage point —
larger than several gaps the benchmark has to decide. Run files are now required
to be a bijection with the registered seeds.

**An escalation that was registered and never enforced.** The protocol had said
from the outset that a task near the 5% line gets eight seeds "and an exact
test". Nothing enforced it, and "an exact test" named no test. This was the
worst of the eight because exploiting it required no action at all: run the
three screen seeds and report. The test is an exact one-sided sign test of
H₀: median ≤ 0.95 × baseline, with *k* of *n* seeds above the threshold and
*p* = P(Binomial(*n*, ½) ≥ *k*). At the registered eight seeds, 8/8 gives
p = 0.0039 and 6/8 gives p = 0.1445 — so a task clearing the line on a bare
majority does not get called.

**Our first version of that fix was itself the same defect.** We made the test
*conditional*: a task counted as "near the line" when the distance from its
mean to the threshold was smaller than its own observed seed range, and only
near-line tasks got the test. That reads as principled — the noise floor rather
than a hand-chosen number — and it is biased, in the flattering direction, by
construction. The expected range of *n* i.i.d. draws is 1.69σ at *n* = 3
against 2.85σ at *n* = 8, and the range sits in the *denominator*: running
fewer seeds shrinks it and makes "comfortably clear, no test needed" easier to
enter. **A gate that is easier to clear on less evidence is not a gate.** On
our five-task screen every margin exceeded its seed range by 4.5× to 68×, so
the pre-registered test ran on none of the five and the clause passed on point
estimates alone. The test is now unconditional at the full registered seed set,
which makes a three-seed screen uncallable arithmetically: *p* ≥ 1/2ⁿ, so
*n* = 3 cannot reach α = 0.05 at any margin. The published rule and the
enforced rule are now the same rule.

**And the test itself was anti-conservative.** We cited the textbook sign test
and implemented it faithfully — ties dropped, fair binomial over the survivors
— without noticing that the textbook version assumes a *continuous*
distribution. Accuracy is *k* correct out of a fixed *n*, so a tie at the
threshold has real probability, and for a discrete distribution "the median is
*T*" does not imply the non-ties split evenly. An adversary supplied the
counterexample and we recomputed it: with P(A = T) = 0.6 and P(A > T) = 0.4 the
median is exactly *T*, so H₀ is true, and the eight-seed gate rejected it
**17.4%** of the time at a nominal 5%. Counting ties as non-wins and keeping
them in the denominator puts that at **0.9%**. No number in this paper moves —
no seed of ours lands on a threshold — but the guarantee the *p*-value
advertises was not the guarantee it had.

**Post-hoc rule changes need their own ledger, and the direction is the test of
them.** All three fixes above were made *after* the screen was read, and the
registry still carried the rule they replaced. The chronology objection is
real: a rule chosen after seeing a result is not pre-registered, whatever its
merits. But the symmetric error is to treat every post-hoc change as
illegitimate, which would forbid fixing a defect once you can see it. The
distinction we settled on, and now enforce, is **direction**: every amendment
is recorded with what was registered, what replaced it, whether results had
been seen, and what it did to the claim as it stood — and a test asserts that
no amendment is recorded as making a clause *easier*. Amendment 1's defence is
not that it was planned; it is that it took the project's status from `PASS` to
`RUNNING` on identical run data. An amendment that withdraws a claim needs no
chronological alibi. One that grants a claim cannot have one. The three
functions that compute the verdict are fingerprinted in the same file, so a
rule change that is not declared fails a test.

What that record makes visible, rather than hides: three of the eight seeds in
our confirmatory set were inspected before the amendment. A fully clean
confirmatory reading would freeze the amended rule and then draw a disjoint
seed set. Ours is disjoint from seed 3 onward, and the report says so.

**Development on the confirmatory tasks.** Recording `git rev-parse HEAD` does
not record the tree that ran. Every run now carries a digest over the agent
package plus a dirty-vs-`HEAD` flag, and disagreement or dirtiness sinks the
autonomy clause. This fired immediately on an uncommitted change to the agent;
the content was benign, which is exactly why the check cannot be left to
judgement.

**Deleting unfavourable attempts.** Moving the bad run files out of the results
directory leaves every check passing on what remains, and the regeneration test
faithfully reproduces the curated collection. Each attempt is now appended to a
ledger *before* its outcome is known and reconciled against the files on disk.
This is append-only by convention, not by permission, and we say so: it converts
a one-command deletion into two coordinated edits and puts the reconciliation
where a reader sees it.

**The one check whose evidence comes from outside the repository.** A sha256
over rows we produced proves the file has not changed since we wrote it, not
that the rows are what OpenML published — and since every other check reads the
same file, lowering a baseline in the cache and re-pinning would pass all of
them. That half is unanswerable internally, so we ask the server: a seeded
random sample of `run_id`s is drawn *before* any fetch, the seed and the rule
are recorded, and each sampled row is re-fetched and compared value for value.
125 rows across the five tasks, zero mismatches. Alongside it, the
contamination routes that *are* answerable from the file: a duplicate `run_id`
counts one submission twice in a median, a row from another task imports
another task's difficulty, and a row carrying another metric blends it into an
accuracy median — the same layout holds AUC and f-measure rows on the server.
The output records what a spot check cannot prove, including the probability
that tampering of a given fraction went undetected, and a test asserts those
figures are present, so "verified" is not a word a reader has to take on trust.

**Not closed, and named rather than glossed:** a rule keyed on feature values
that encodes dataset identity while passing the invariance test; and label
leakage, since the evaluator holds the full labelled frame in the same process
that calls the agent. The second is architectural and would need subprocess
isolation. Cache provenance is now *partly* closed — the remaining gap is that
a spot check of *k* rows would very likely miss a single edited row, which only
a dated, independently acquired snapshot answers.

## 6. The noise floor, and why three seeds is a screen

The outer folds are the task's own and identical across seeds, so the only
variance is the agent's: the inner-CV shuffle, the selection subsample and the
random search. An early version pinned every estimator at a fixed
`random_state` regardless of the agent's seed, which would have reported a
spread far below the truth — and would have been believed. A test now fails if
that regresses.

Three seeds are reported as a **screen, not a verdict**, and that sentence is
now enforced rather than promised: because *p* ≥ 1/2ⁿ, a three-seed run cannot
reach α = 0.05 in either direction, so no three-seed screen can produce a
verdict however wide its margin looks. This follows a lesson paid for elsewhere
in this workspace: two identical invocations of one configuration differed by
more than most of the effects that had been reported as findings.

The *p*-floor argument stops at *n* = 5, where an all-above run gives
p = 0.031 and would reject — so stopping at the first rejection would report a
sequentially monitored *p*-value as if it were fixed-sample. A task is
therefore called only at the full registered seed count, whichever way the test
comes out.

**What eight seeds do not measure.** They re-draw the agent's own randomness on
the *same* examples and the same outer folds, so they estimate algorithmic
variance conditional on this data — not uncertainty about new data. We do not
convert the eighty fold scores into eighty observations to buy power: their
training sets overlap, and treating dependent folds as independent understates
the variance (Bengio & Grandvalet, JMLR 2004).

**A per-task verdict is not an end-to-end verdict**, which matters for a KPI
whose second clause is *무개입*. Five tasks each failing on a *different* seed
would pass every per-task test and never once produce a clean sweep. So we also
report, per seed, whether **one unattended run cleared all five** — the reading
a reader of "autonomous" actually wants — beside the per-task gate rather than
instead of it. Testing five tasks at α = 0.05 needs no multiplicity correction,
because the claim requires all five to reject: that is an intersection-union
test, and a false global pass needs at least one true task null rejected.

### 6.1 Two noise sources, and we spent our effort on the smaller one

The seed test constrains the agent's own randomness with the data held fixed.
The other axis — which fold you happen to be scored on — is measurable from the
same run records, since each one carries its ten outer-fold accuracies. Putting
a one-sided 95% lower bound on the mean per-fold relative margin against the
pre-registered −0.05 tolerance, non-inferiority holds on all five tasks, under
a naive `s/√K` bound and under the Nadeau & Bengio (2003) variance inflation
`(1/K + n_test/n_train)·s²` alike. That is a stronger result than the sign
test in the sense that matters: it uses the magnitudes the sign test discards,
and it lies on the axis generalisation actually depends on.

Both bounds are reported because the honest one is not obvious. The folds have
disjoint test sets and heavily overlapping training sets, so `s/√K` understates
the variance — the flattering direction — while the correction is *derived* for
repeated random subsampling and is applied here as a conservative adjustment.
We do not pool the eighty fold-by-seed scores as eighty independent
observations; that buys power by assuming away exactly the dependence at issue.

The number that matters in this subsection is not the interval. It is the
ratio: **the fold-to-fold standard deviation of the relative margin exceeds the
seed-to-seed range by 1.6× to 4.4×.** We had rewritten the gate, logged three
protocol amendments, added seven tests and spent two hours of compute
escalating from three seeds to eight — all on the axis that is measurably the
smaller of the two. The seed work was not wrong; the old gate really was biased
at small *n* and the tie handling really did inflate Type I error to 17.4%. It
was misallocated, and the check that would have said so cost nothing and used
data already on disk.

Stated as a prediction and then checked: if fold variance dominates, adding
seeds 3–6 should move the per-task means by far less than the fold spread. It
moved them by +0.13%, +0.18%, +0.11%, −0.05% and +0.15% relative, against fold
standard deviations of 0.37%–4.09% — about an order of magnitude less.

## 6.5 The audit that none of the above performs: is the target demanding?

Everything to this point asks whether *our* number is honest. It is the wrong
question to stop on, and an adversary asked the right one only when the
question put to it changed from "what is wrong with this" to "how would you
make this pass more defensibly". Its first answer was not about our number:

> "On imbalanced datasets like blood-transfusion (majority share 76.2%) or kc1
> (84.5%), the 0.95 × median_run threshold is 72.5% and 80.9%. A dumb
> `DummyClassifier(strategy='prior')` or single-split decision tree clears the
> primary threshold on 4 of the 5 tasks without learning anything. Does passing
> this clause actually prove data science competence, or is the bar
> unfalsifiable?"

The first half of that is checkable from the registry with no run at all: **on
four of the five registered tasks, `0.95 × median_run` sits at or below the
task's own majority-class rate.** We then measured it, running two frozen and
deliberately incapable procedures — a majority-class dummy, and an untuned
depth-3 decision tree — through the *same* outer folds, the same pooled
statistic and the same pre-registered baselines as the agent. Both clear the
primary reading on four of five tasks. The tree clears the strictest reading on
four of five.

Only the one *balanced* task discriminates on its own. What survives is the
**joint** criterion: neither control clears all five, because both collapse
where the majority class is not a strategy. The defensible reading of a PASS is
therefore "clears five tasks including one where triviality fails" — not "beat
a human five times", which is what the KPI's wording invites.

Two consequences we did not expect and report because they are what the runs
say.

**The agent's margin over the trivial tree is small wherever the clause is
weak.** Per task: +0.0093, −0.0209, +0.0010, +0.0921, +0.0094. One is inside
that task's own seed range and so is not a difference this experiment can
resolve; on one, the tree *wins*. Six model families plus a random search over
the winner buys a large margin on the balanced task and close to nothing over
three splits of a tree on the four imbalanced ones — and this was invisible in
every comparison against the human baseline, because the human baseline is also
below the majority rate on those tasks. A weak baseline hides a weak method as
efficiently as it flatters a strong one.

**The binding constraint on this KPI is the task set, not the agent.** §2.1
named the selection bias — the most-published CC18 tasks are the oldest and
most famous, hence the small, clean and imbalanced ones — and treated it as a
limitation of coverage. It is worse than that: it partly dissolves the clause.
Ranking by number of published evaluations was chosen because the median of a
larger sample is better determined, which is true, and it selects for exactly
the tasks whose medians sit near triviality.

The general lesson, and the reason this section exists at all: **provenance
machinery has a blind spot precisely where it is strongest.** Every check in
§4 and §5 constrains the relationship between our claim and our evidence. None
of them constrains the relationship between the *target* and the *difficulty of
the problem*, so a target can be pre-registered, hash-pinned, externally
re-fetched, amendment-logged, and vacuous. A negative control is the cheapest
check in this repository — a quarter of an hour — and it moved the
interpretation of the headline more than anything else built this weekend.

### 6.6 The failure mode this paper kept committing

§6.1 and §6.5 are the same mistake twice, and we would rather name it once than
present two coincidences. In both cases we invested heavily in rigour **on the
axis we were already looking at**, and the axis that dominated went unmeasured
until an adversary or a cheap diagnostic pointed at it:

- seed variance was constrained with a rewritten gate, an amendment ledger and
  a compute escalation, while split variance — 1.6–4.4× larger — sat
  uncomputed in run records already on disk;
- the *provenance* of our claim was constrained by nine layers of checks, while
  the *difficulty of the target* went unexamined until a dummy classifier
  cleared four of five thresholds;
- and, a third time, having found that the target was weak, we asked whether our
  *selection rule* caused it with an exact hypergeometric over the five selected
  tasks — a test which, at n = 5 against a ~50% base rate, can reject only on
  the single most extreme draw. It returned p = 0.022 under the class-prior
  floor and we published that in four documents. Under the procedure floor it
  returns p = 0.187, and the properly powered test — the same association
  measured over all 51 candidates, n = 51 rather than 5 — gives
  ρ = +0.281 with a one-sided permutation p = 0.024. The conclusion survived;
  **the evidence we had given for it did not**, and the test we had reached for
  was one whose power we never checked. A p-value from a test that can fire in
  one outcome out of six is not evidence about a rule, and a criterion that
  makes one's own control fail by construction will also make one's own
  significance test look good by construction, because both are downstream of
  the same too-easy floor.

The corrective is one line and it is not "be more careful": **measure the sizes
of the things you are choosing between, before choosing which to constrain** —
and that includes the power of the test you are about to quote.
Both diagnostics were available at the start, both were cheap — one needed no
runs at all and the other a quarter of an hour — and both would have reordered
the work. Rigour is not free of opportunity cost, and effort spent proving a
small term is indistinguishable, from the outside, from effort spent hiding a
large one.

## 7. What would make this a stronger paper

- **Undisclosed confirmatory tasks.** Every defence in §4 and §5 is weaker than
  simply evaluating on tasks whose identity the agent's author has never seen.
  This is the single highest-value change and it needs a second party.
- **Subprocess isolation** of the agent from the evaluator, which converts
  "무개입" from reviewed to enforced.
- **An externally anchored baseline snapshot** — a dated, independently
  acquired evaluation dump with unique run ids — which is the only real answer
  to cache provenance.
- **A task set that the negative controls fail**, selected by a rule that
  cannot see our accuracy and does not select for popularity — for instance the
  CC18 tasks whose majority-class rate is below `0.95 ×` their published median,
  which is a property of the *published baselines* and so still cannot see us.
  This is now the highest-value experiment on the list, ahead of everything
  else here, because §6.5 shows the current set cannot separate competence from
  triviality on four of five tasks. The distinguishing prediction: on a balanced
  set the agent's margin over the depth-3 tree should look like the kr-vs-kp
  margin (≈0.09) rather than the imbalanced-task margin (≈0.01), and the dummy
  should clear nothing. It is a **different measurement** and must be reported
  as one, never swapped in for this KPI.
- **Fold-level non-inferiority with a resampling correction** (Nadeau &
  Bengio 2003) over the ten outer-fold accuracies already logged per run, to
  put a confidence interval on the relative margin. Our seed test measures
  algorithmic variance conditional on fixed folds; this is the reading that
  speaks to split variance, and the two are not substitutes.
