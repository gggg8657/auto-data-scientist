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
three screen seeds and report. Both halves are now specified. A task is *near
the line* when the distance from its mean to the threshold is smaller than its
own observed seed range — our noise floor, not a hand-chosen number — and such a
task cannot be called on the screen. The test is an exact one-sided sign test of
H₀: median ≤ 0.95 × baseline, with *k* of *n* seeds above the threshold and
*p* = P(Binomial(*n*, ½) ≥ *k*). At the registered eight seeds, 8/8 gives
p = 0.0039 and 6/8 gives p = 0.1445 — so a task clearing the line on a bare
majority does not get called.

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

**Not closed, and named rather than glossed:** a rule keyed on feature values
that encodes dataset identity while passing the invariance test; the provenance
of the evaluation cache itself, where a hash computed over rows we control
proves consistency rather than authentic acquisition (the cheap half — no
repeated `run_id` across 3.4M cached evaluations — is checked; independent
re-acquisition against a dated external snapshot is not built); and label
leakage, since the evaluator holds the full labelled frame in the same process
that calls the agent. The third is architectural and would need subprocess
isolation.

## 6. The noise floor, and why three seeds is a screen

The outer folds are the task's own and identical across seeds, so the only
variance is the agent's: the inner-CV shuffle, the selection subsample and the
random search. An early version pinned every estimator at a fixed
`random_state` regardless of the agent's seed, which would have reported a
spread far below the truth — and would have been believed. A test now fails if
that regresses.

Three seeds are reported as a **screen, not a verdict**. Escalation to eight
seeds and the exact test above is required for any task whose margin is inside
its own spread. This follows a lesson paid for elsewhere in this workspace:
two identical invocations of one configuration differed by more than most of
the effects that had been reported as findings.

## 7. What would make this a stronger paper

- **Undisclosed confirmatory tasks.** Every defence in §4 and §5 is weaker than
  simply evaluating on tasks whose identity the agent's author has never seen.
  This is the single highest-value change and it needs a second party.
- **Subprocess isolation** of the agent from the evaluator, which converts
  "무개입" from reviewed to enforced.
- **An externally anchored baseline snapshot** — a dated, independently
  acquired evaluation dump with unique run ids — which is the only real answer
  to cache provenance.
- **A harder task set**, selected by a rule that cannot see our accuracy and
  does not select for age and cleanliness, reported as a *different*
  measurement rather than as this KPI.
