# WEEKEND.md — auto-data-scientist (B5), track γ

**KPI (verbatim):** 공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 · end-to-end 무개입

## Headline

<!-- HEADLINE:BEGIN -->
<!-- HEADLINE:END -->

**Friday:** the repository did not exist.
**Before this turn:** the pipeline and the pre-registered target existed and
there was **no accuracy number at all** — clauses 2 and 3 read
`[not measured]`, which was the honest state and was reported as such.
**Now:** the first accuracy numbers exist, the project's own gate withdrew the
first `PASS` it computed, and a cheap check nobody had run showed that **most
of the KPI's own thresholds cannot distinguish competence from triviality.**
Read the next section before the numbers.

---

## Read this first: 4 of the 5 thresholds are below the majority-class rate

Everything this repository has built over nine turns asks whether *our* number
is honest. None of it asked whether the *target* is demanding. It mostly is
not.

A frozen `DummyClassifier(strategy="prior")` — predicts the commonest label,
ignores the features entirely — run through the **same** outer folds, the same
pooled statistic and the same pre-registered baselines as the agent:

- clears the primary `±5%` clause on **4 of the 5 tasks**;
- an untuned depth-3 decision tree clears **4 of 5** on the primary reading and
  **4 of 5** on each task's *strictest* baseline;
- **4 of the 5 thresholds sit at or below the task's own majority-class rate**,
  which needs no run at all to check — it is arithmetic on the registry.

Only `kr-vs-kp` (majority 0.5222, threshold 0.9120) discriminates on its own.

**And it was not bad luck.** Of the 51 candidates that passed the registered
size filter, **36** have a threshold above their majority rate. The rule
selected **1** of them. A random five would be expected to contain 3.53; exact
lower-tail hypergeometric **p = 0.0222**. Ranking by number of published
evaluations — chosen because the median of a larger sample is better
determined, which is true — selects at better than the 5% level for tasks whose
±5% band sits beneath triviality.

**What survives:** the *joint* five-task criterion. Neither control clears all
five, because both collapse where the majority class is not a strategy. So a
`PASS` here means **"clears five tasks including one where triviality fails"**,
not "beat a human five times". That sentence is in `RESULTS.md` beside the
result, and a test asserts that if a control ever does clear all five, the
reassuring paragraph is false and must be rewritten rather than the test
relaxed.

**The agent comes out of it worse than expected too.** Margins over the untuned
depth-3 tree: credit-g +0.0093, blood-transfusion **−0.0209**, kc2 +0.0010,
kr-vs-kp +0.0921, kc1 +0.0094. One is inside that task's own seed range; on one
the stump *wins*. Six model families plus a random search buys a lot on the
balanced task and almost nothing over three splits of a tree on the imbalanced
ones — invisible against the human baseline, because that baseline is *also*
below the majority rate there. **A weak baseline hides a weak method as
efficiently as it flatters a strong one.**

The transferable lesson: I pre-registered a selection rule that was blind to my
accuracy and treated that as the whole requirement. A rule has to be blind to
your result **and** blind to the difficulty of the target. Only the first was
designed for.

---

## The other thing that happened: the first PASS was withdrawn

The 3-seed screen finished and `scripts/report.py` printed **`Status: PASS`**.
Withdrawn in the same turn, before it went anywhere. Not because a number was
wrong — all five tasks land above their median published run — but because the
gate that granted the clause was the wrong test.

The gate short-circuited: a task whose margin to the 5% line exceeded its own
observed seed range was called *without running the test*. All five cleared by
4.5×–68× their seed range, so **the pre-registered exact test ran on none of
them** and the clause passed on point estimates.

Why that is wrong in the flattering direction: the expected range of *n* i.i.d.
draws is 1.69σ at *n* = 3 against 2.85σ at *n* = 8, and the range sits in the
**denominator**. Fewer seeds ⇒ smaller range ⇒ easier to enter the "no test
needed" branch. **A gate that is easier to clear on less evidence is not a
gate.** (Measured on our own runs, as seeds landed: the observed range grew
1.00×–2.25× per task going from 3 seeds to 6.)

The exact test is now unconditional at the full 8-seed set. Since *p* ≥ 1/2ⁿ, a
3-seed screen cannot reach α = 0.05 at any margin, so "with three seeds, screen
not verdict" is arithmetic rather than a promise. Same runs, `PASS` →
`RUNNING`.

### What the two adversaries found

`codex`, asked *how to make the clause pass legitimately* rather than what was
wrong, found that **the exact test itself was anti-conservative**: it dropped
ties and ran a fair binomial on the survivors, which is the sign test for a
*continuous* distribution. Accuracy is *k* correct of a fixed *n*. With
P(A = T) = 0.6, P(A > T) = 0.4 the median is exactly *T* — H₀ true — and the
8-seed gate rejected it **17.37%** of the time at a nominal 5%. Recomputed
rather than trusted: 17.367% before, **0.852%** after. It also found the
provenance gate failing open in three places, where a run that *omitted* a
field read as one that *reported it clean*.

`agy` found that **`runs/bench/` and `runs/attempts.jsonl` were never tracked
by git** — so deleting a bad seed's record together with its three ledger lines
left `reconciled: True`, every test green and no diff at all. That invalidates
the stated purpose of the ledger, which existed to make deletion cost two
coordinated edits; the argument silently assumed the file was in the history.
It also found that the OpenML re-fetch probe re-checked the *same* 25 rows every
time (fixed seed), leaving the other 415,087 rows of task 31 editable with
impunity — the sample seed is now derived from the file's own sha256, so editing
any row re-draws the sample.

Every hole either adversary found was **in the flattering direction, and none
was in the arithmetic**. They were in what gets loaded, what gets counted when
a field is absent, which rows get checked, which seed set the bar refers to,
and what a document may say while a job is running.

---

## What I tried that did not work, and what it rules out

- **Conditioning the test on "is this task near the line".** Ruled out: any
  gate whose trigger is estimated from the same small sample it is gating
  inherits that sample's bias, and here the bias favours the claim. Report the
  margin as a *diagnostic*, gate on the test.
- **My own first fix for "the ledger can't reconcile mid-run".** I read
  `/proc` inside `reconcile_ledger` to tell an in-flight attempt from an
  abandoned one. That made the committed document a function of live process
  state, so the freshness test could never hold on a clean checkout and CI
  would have been permanently red. Ruled out: **a committed artefact must be a
  pure function of committed inputs.** The live reading is now a guard in
  `main()` that redirects mid-run output to an untracked interim file.
- **Enumerating test files in CI.** It was running **5 of 9**, and the file it
  omitted included `test_escalation_gate.py` — the test of the gate that
  decides clause 2. Ruled out: a hand-listed CI step list is a list someone has
  to remember to extend. Now discovered by glob with a floor on the count.
- **Treating "not yet run" as "did not clear".** The new per-seed sweep table
  scored the mid-flight seed as having *missed* two tasks that had not started.
  Second time this weekend a partial artefact scored as a bad result rather
  than as no result (the first was `etch-operator-twin`'s partial checkpoints,
  another track). Worth naming as a recurring class: **absence must route to
  `None`, never to a value.**
- **Provenance machinery as a defence against a vacuous target.** Nine turns
  of hash pinning, ledgers, digests, amendment records and external re-fetches,
  and none of it could see that most of the thresholds were beneath the
  majority-class rate. Ruled out: **auditing the link between claim and
  evidence says nothing about the link between target and difficulty.** The
  negative control that found it took a quarter of an hour and should have been
  the *first* thing built, before any of the rest.
- **A hand-typed number surviving in generated prose.** The clean-subset
  paragraph carried "0.0013–0.0077" as literal text; those were the 3-seed
  spreads and were stale by the 6th seed (0.0022–0.0090). Ruled out: a
  generator is not a guarantee — prose inside it needs the same discipline as
  a table cell. Now computed, along with the count in "N of the five sits
  inside seed noise", which said *two* and is *one*.
- **A test writing into the repository's own documents.** `--weekend` was added
  with a default of the real `WEEKEND.md`, so a fixture run over five made-up
  tasks wrote `d0`–`d4` rows with `[not measured]` accuracies into the handover
  file. Ruled out: a script that writes documents must treat an explicit
  `--out` as "the caller is writing elsewhere" and not touch siblings.
  `README.md` had the identical exposure and was safe only by accident.
- **Adversary tooling.** `codex` works. `agy` needs `-p` *and*
  `--dangerously-skip-permissions` and returned nothing twice before that;
  it is worth the trouble — it found the untracked-ledger hole and the
  falsifiability question. `cursor-agent` is unauthenticated on this box and
  needs `agent login` or `CURSOR_API_KEY`. Also: ask **"how would you make
  this pass"**, not "what is wrong" — the single most valuable finding of the
  turn came from the constructive phrasing, and the destructive phrasing had
  been asked twice before without producing it.

---

## Needs a human decision

**1. The KPI's task set cannot separate competence from triviality on 4 of 5
tasks. Which measurement do you want reported?**
This is the decision that matters and it is not really about statistics.
A successor rule that is still blind to our accuracy exists and is recorded in
`runs/target_difficulty.json`: the five most-published candidates whose
threshold exceeds their majority-class rate — `kr-vs-kp`, `qsar-biodeg`,
`wdbc`, `diabetes`, `phoneme`. Ranks 6–15 were reserved by the protocol for
development and **the agent was never run on any of them** (`runs/dev` did not
exist until now), so they are genuinely uninspected.

- *(a)* **Report the registered five as the KPI, with the falsifiability
  section attached.** *This is what is built and what the rules require* — the
  five were pre-registered and swapping in a set chosen after seeing which one
  made the point is exactly the error this repo exists to prevent.
- *(b)* Additionally run the successor five as a clearly-labelled **second
  measurement** and report both. ~2 h of CPU, no code changes, and it answers
  the objection instead of documenting it. **My recommendation** — and it also
  tests the alternative explanation (if the agent's margin over a depth-3 stump
  stays ≈0.01 on balanced tasks rather than widening to ≈0.09, the finding is
  about the agent, not the task set).
- *(c)* Treat B5 as `UNREACHABLE` on the grounds that the clause is not
  meaningful. **I do not recommend this**: the clause is meetable and, read
  jointly, is not vacuous.

**2. Is the amended protocol acceptable, or does it need a fresh seed set?**
Five rule changes were made *after* the 3-seed screen was read, all recorded in
`runs/protocol_amendments.json`, all strictly stricter, and #1 took the status
from `PASS` to `RUNNING` on identical data. But **3 of the 8 verdict seeds
(0, 1, 2) were inspected pre-amendment.**

- *(a)* **Accept the 8-seed set with the disclosure**, plus the section that
  re-runs the exact test on seeds 3–7 alone — a fixed subset named *in* the
  amendment before those seeds were run, so it owes nothing to the inspected
  ones, and n=5 all-clearing reaches p = 0.03125 on its own. *This is what is
  built.*
- *(b)* Run seeds 8–15 as well. ~2 h of CPU. I did **not** do this and the
  reason is measured, not stylistic: seed noise is 0.0022–0.0090 against
  margins of 0.0440–0.0844, so more seeds shrink a quantity already an order of
  magnitude below the effect and do nothing about the uncertainty that binds —
  each task is **one** fixed dataset with **one** fixed set of folds.

**3. Same as last turn, still unresolved and still blocking all tracks: disk.**
`/home/dongjukim` (7.0T) hit 100% / 0 bytes; I reclaimed 40G from `~/.cache/pip`
only. **~5.1T of the 6.6T used is outside this container's view.** Options
unchanged: *(a)* reclaim `~/.cache/huggingface` (267G, but F4 needs Qwen3
weights), *(b)* reclaim `~/.ollama` (278G, no track in my brief uses it),
*(c)* host access to find the 5.1T, *(d)* accept ENOSPC deaths.
**Recommendation: (b) then (c).** Not done unattended: hard to reverse, and not
mine.

**4. Should the agent be isolated from the evaluator's memory?**
`ads/evaluate.py` holds the full labelled frame in the process that calls the
agent. Nothing in `ads/` touches it and the code is short enough to review, but
the architecture permits leakage and no in-process test can rule it out.
*(a)* accept on review (in place), *(b)* subprocess per fold with only the
permitted arrays passed in — about a day, and it makes 무개입 enforced rather
than trusted.

---

## Still running, and how to check it

The **8-seed confirmatory run**, in its own tmux session:

```bash
cd ~/Documents/workspace/auto-data-scientist
tmux ls | grep ads-verdict                  # the session
tail -3 logs/bench_verdict.log              # per-fold progress; ends with EXIT=0
ls runs/bench/*.json | wc -l                # 40 when complete (5 tasks x 8 seeds)
```

Seeds 0–2 were already on disk and were skipped, so this run produces seeds
3–7. When it finishes:

```bash
.venv/bin/python scripts/report.py          # regenerates RESULTS.md + this headline
for f in tests/test_*.py; do python "$f"; done
git add -A && git commit                    # then push, only if green
```

`scripts/report.py` **refuses** to write `RESULTS.md` while a benchmark is
alive — it writes `runs/interim_report.md` instead — so a mid-run document with
an unreconcilable ledger can never be committed as the verdict.

---

## Where the numbers come from

`scripts/report.py` is the only code permitted to write a number into a
document, including the headline block at the top of this file. Every clause is
derived by `verdict()`; `PASS` is returned untouched if the runs earn it. What
it refuses to grant:

- any task not **called** by the exact sign test at the full registered 8 seeds
  (8/8 → p = 0.0039; 6/8 → p = 0.1445, not called; and n = 5, 6, 7 are not
  called *even when they reject*, because stopping at the first rejection is
  optional stopping);
- any run set with a duplicated seed, an unregistered seed, a partial or
  off-registry run, a failed attempt, two `ads/` digests, a dirty tree, a run
  missing a required field, a run with no agent digest, or a ledger that is
  absent or does not reconcile;
- a `NOT MET` off an unfinished measurement — a screen short of the registered
  seed count reads `None` → `RUNNING`, because unfinished is not failed, and
  that error is as real as its opposite.

Long form in `critique_log.md` (every withdrawal, with the arithmetic) and
`paper_draft.md`. Verified against OpenML itself: 125 evaluation rows
re-fetched from the server across the five tasks, 0 mismatches
(`runs/evals_provenance.json`) — the only check here whose evidence comes from
outside this repository.

**Open and named rather than glossed:** a rule keyed on feature *values* that
would encode dataset identity while passing the column-invariance test (#6);
in-process label leakage (#8). Cache provenance (#7) is now partly closed — a
*k*-row spot check would still likely miss a single edited row.
