# WEEKEND.md — auto-data-scientist (B5), track γ

**KPI (verbatim):** 공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 · end-to-end 무개입

## Headline

<!-- HEADLINE:BEGIN -->
<!-- HEADLINE:END -->

**Friday:** the repository did not exist.
**As of the last turn before this one:** the pipeline and the pre-registered
target existed, and there was still **no accuracy number at all** — clauses 2
and 3 read `[not measured]`, which was the honest state and was reported as
such.
**Now:** the numbers above are the first this repository has produced, and the
project's own gate withdrew the first `PASS` it computed. Read the next section
for why that is the point rather than a setback.

---

## The one thing that happened this turn

The 3-seed screen finished and `scripts/report.py` printed **`Status: PASS`**.
That claim was withdrawn within the same turn, before it was pushed anywhere.
Not because any number was wrong — all five tasks land *above* their median
published run, by +0.58% to +4.32%, with a seed spread of 0.0013–0.0077 — but
because the gate that granted the clause was the wrong test.

The gate short-circuited: a task whose margin to the 5% line exceeded its own
observed seed range was called *without running the test*, and only borderline
tasks got the pre-registered exact sign test. Every one of the five cleared its
line by 4.5×–68× its seed range, so **the test ran on none of them** and the
clause passed on point estimates.

Why that condition is wrong, and wrong in the flattering direction: the
expected range of *n* i.i.d. draws is 1.69σ at *n* = 3 against 2.85σ at *n* = 8,
and the range sits in the **denominator**. Fewer seeds ⇒ smaller range ⇒ easier
to enter the "no test needed" branch. **A gate that is easier to clear on less
evidence is not a gate.**

The exact test is now unconditional and required at the full 8-seed set. Since
*p* ≥ 1/2ⁿ, a 3-seed screen cannot reach α = 0.05 at any margin, so "with three
seeds, screen not verdict" is now arithmetic rather than a promise. Same runs,
`PASS` → `RUNNING`. Nothing was loosened and no number moved.

### The adversary earned its keep this time

Asked the *right* question — "how would you make this clause pass legitimately"
rather than "what is wrong" — `codex` found a defect I would not have found:

**the exact test itself was anti-conservative.** It dropped ties and ran a fair
binomial on the survivors, which is the textbook sign test *for a continuous
distribution*. Accuracy is *k* correct of a fixed *n*. With
P(A = T) = 0.6, P(A > T) = 0.4 the median is exactly *T* — H₀ true — and the
8-seed gate rejected it **17.37%** of the time at a nominal 5%. I recomputed it
rather than trusting the figure: 17.367% before, **0.852%** after counting ties
as non-wins in the denominator. No number here moves (no seed of ours sits on a
threshold) but the guarantee the *p*-value advertised was not the one it had.

It also found the provenance gate **failing open** in three places — a missing
`n_interventions` defaulted to 0, an absent agent digest was *discarded* before
the distinct-digest count, an absent ledger was accepted — so a run that
*omitted* a field read as a run that *reported it clean*. All three now sink
clause 3.

And it was right about chronology: all of this was decided **after** the screen
was read. See below.

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
- **`agy` as a second adversary.** Two invocation failures (it reads prompts
  only from `-p`/stdin, and headless mode auto-denies file reads without
  `--dangerously-skip-permissions`). Noted so the next turn does not re-derive
  it.

---

## Needs a human decision

**1. Is the amended protocol acceptable as a confirmatory reading, or does it
need a fresh seed set?**
Three rule changes were made *after* the 3-seed screen was read, all recorded
in `runs/protocol_amendments.json` with what was registered, what replaced it,
and what each did to the claim. All three are strictly stricter, and amendment 1
took the status from `PASS` to `RUNNING` on identical data — an amendment that
*withdraws* a claim needs no chronological alibi. But **3 of the 8 verdict
seeds (0, 1, 2) were inspected before the amendment.**

- *(a)* **Accept the 8-seed set as it stands**, with the disclosure in
  `RESULTS.md` that seeds 0–2 were pre-inspected and 3–7 were not. *This is
  what is built.* The per-seed table lets a reader drop seeds 0–2 and check the
  conclusion on 3–7 alone.
- *(b)* Run seeds 8–15 as a clean confirmatory set under the frozen amended
  rule and report the current 8 as supporting evidence. Costs ~2 h of CPU and
  nothing else. **My recommendation if anyone external will read the claim.**

**2. Same as last turn, still unresolved and still blocking all tracks: disk.**
`/home/dongjukim` (7.0T) hit 100% / 0 bytes; I reclaimed 40G from `~/.cache/pip`
only. **~5.1T of the 6.6T used is outside this container's view.** Options
unchanged: *(a)* reclaim `~/.cache/huggingface` (267G, but F4 needs Qwen3
weights), *(b)* reclaim `~/.ollama` (278G, no track in my brief uses it),
*(c)* host access to find the 5.1T, *(d)* accept ENOSPC deaths.
**Recommendation: (b) then (c).** Not done unattended: hard to reverse, and not
mine.

**3. Should the agent be isolated from the evaluator's memory?**
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
