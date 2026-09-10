# WEEKEND.md — auto-data-scientist (B5), track γ

**KPI (verbatim):** 공개 데이터셋 5개에서 사람 baseline ±5% 이내 자동도달 · end-to-end 무개입

**Status: RUNNING. There is no accuracy number in this repository yet, and
that is the honest headline.** Read the next two lines before anything else.

- **Friday:** nothing — the repo did not exist.
- **Now:** the target is pre-registered and (as of this turn) actually correct,
  the pipeline is built and tested end-to-end, four ways of faking the KPI are
  closed, and **the confirmatory run has not been launched.** Clauses 2 and 3
  read `[not measured]` in `RESULTS.md`, not a placeholder number.

I would rather hand over an empty results table with a trustworthy target than
a filled one measured against a target that was wrong in four ways this morning.

---

## The one thing that happened this turn

**Two of the five pre-registered targets were wrong, and every test that could
have caught it did.** `fetch_baselines.py` paged published OpenML evaluations
with a 300,000-row cap. Tasks **31 (credit-g)** and **10101
(blood-transfusion)** — ranks 1 and 2, the only 2 of 51 — hit it exactly, so
their "median of published runs" was the median of an oldest-first *prefix*.
Probed true sizes: task 31 has **500k–600k** evaluations, task 10101
**400k–500k**. So those medians covered roughly 55% and 70% of the population,
drawn from the 2014-era end — the exact bias the pre-registration named as its
reason for paging to exhaustion.

Superseded readings, kept in `runs/baselines_truncated_superseded.json`:

| task | superseded `median_run` | superseded `median_flow` |
|---|---|---|
| 31 credit-g | 0.7200 | 0.7290 |
| 10101 blood-transfusion | 0.763369 | 0.762032 |

**Why this is a fix and not a moved goalpost:** it was done before any accuracy
of mine existed, so it cannot have been steered toward a result; it moves the
sampling *toward* what the protocol already specified; and the selection rule is
provably stable, because only those two tasks were truncated, both at ≥300,000,
while rank 3 has 173,611 — so de-truncating cannot change which five tasks are
in. The re-fetch is **still running** (see below). When it lands I expect both
medians to come out **higher** (targets get harder), and larger on task 31 than
on 10101. That prediction is on the record in `critique_log.md` so it can be
wrong.

Two smaller defects of the same family: `runs/baselines.json` was **never
tracked by git** (written 09:10, 33 minutes after the 08:37 commit whose message
says it was committed), and it was produced by a script that was **edited
mid-flight at 08:45**, so the committed artifact lacked four baseline readings
and the whole `run_protocol` block that its own committed producer emits.

---

## What I tried that did not work, and what it rules out

- **`df -h /` to diagnose a disk.** It reported 176G free and was the wrong
  filesystem: `/home/dongjukim` is a separate 7.0T ext4 mount and it was at
  **0 bytes**. Rules out "there is space" as an explanation for any ENOSPC on
  this box; always `df` the actual path.
- **Trusting `write_text` for a run record.** The module docstring promised
  "complete or absent, never half-written" and the outage left exactly one
  4096-byte page of a record at the record's own path — and the raised
  exception did not remove it, so the runner's `out.exists()` guard would have
  trusted the stump on the next pass. Rules out exception-raising as sufficient
  for atomicity.
- **A literal scan for dataset identity.** Two registered task ids are `31` and
  `3`, and `agent.py` legitimately contains `max_leaf_nodes=[15, 31, 63]` and
  `INNER_FOLDS_LARGE = 3`, so the guard had been failing on hyperparameters
  since it was written. Rules out bare-integer scanning for small ids; replaced
  with an invariance test that actually passes and actually tests the property
  (renaming every column and reversing their order leaves the chosen family,
  the logged decisions and **100% of predictions** unchanged).
- **Asking the adversary what was wrong.** Asking *"how would you fake this
  KPI without editing a number and without failing a test?"* produced eight
  ranked attacks, four of them real and now closed. The single worst needed no
  action to exploit: run the three screen seeds and report, because the 8-seed
  escalation had been registered since before any result and nothing enforced
  it, and "an exact test" named no test. Both are now specified and enforced.

---

## Needs a human decision

**1. Disk on this box — the one that will bite all three tracks.**
`/home/dongjukim` (7.0T) hit **100% / 0 bytes** and I reclaimed **40G** by
deleting `~/.cache/pip` and nothing else. Only 1.5T of the 6.6T used is under
`/home/dongjukim`; **~5.1T is on the same device outside this container's view**,
so I cannot see or reclaim it. 40G is headroom, not a fix.

- *(a)* Reclaim `~/.cache/huggingface` (**267G**) — but F4 needs the Qwen3
  weights and other tracks may depend on it, so it costs re-downloads.
- *(b)* Reclaim `~/.ollama` (**278G**) — no track in my brief uses it; this is
  the cheapest large win **if** nobody else needs it. I did not touch it
  because it is not mine.
- *(c)* Find the ~5.1T outside the container, which needs host access.
- *(d)* Do nothing and accept that long jobs die at ENOSPC.

**My recommendation: (b), then (c).** I did not do (b) unattended because it is
hard to reverse and outward-facing.

**2. Is `median_run` the baseline you actually want?**
It is what the brief specified and it stays the pre-registered primary — I am
not swapping it after the fact even for something harder. But be clear what a
PASS on it means: it compares *our selected model* against *the distribution of
all human trials, failures and default-parameter sweeps included*, where one
uploader's 5000-point sweep counts 5000 times. codex also correctly caught that
my own documentation **overstated the mathematics**: `median_flow_best ≥
median_flow` is guaranteed, but `median_flow_best ≥ median_run` is *not*.

- *(a)* Keep `median_run` primary, report all four readings and the strictest
  verdict beside it. **This is what is built**, and the strict verdict is now
  binding-and-visible rather than recorded-and-ignored.
- *(b)* Declare a selected-solution reading (`median_uploader_best`) primary in
  a *successor* KPI, labelled as a different and harder claim — never as this
  one.

**3. Should the agent be isolated from the evaluator's memory?**
`ads/evaluate.py` holds the full labelled frame in the same process that calls
the agent (codex attack #8). Nothing in `ads/` does anything with it, and the
code is short enough to review, but the *architecture* permits leakage and no
test can rule it out from inside the process.

- *(a)* Accept it, on code review — cheap, and it is what is in place.
- *(b)* Run each fold's agent in a subprocess with only the permitted arrays
  passed in. A day of work, and it makes "무개입" enforced rather than trusted.

---

## Still running, and how to check it

**The exhaustive baseline re-fetch** (pid was 3264113, started ~12:29, still
paging at ~22 min; deep-offset pagination on this API degrades badly):

```bash
cd ~/Documents/workspace/auto-data-scientist
tail -5 logs/fetch_baselines_exhaustive.log     # per-task completion lines
ls -la runs/evals/task_31.csv.gz runs/evals/task_10101.csv.gz   # appear when done
```

It only re-pages tasks 31 and 10101; the other 49 come from cache in ~45s. When
it finishes, `runs/baselines.json` is rewritten with all four baseline readings,
`strictest_baseline`, and `run_protocol`.

**Exact sequence to resume from (each step gated on the previous):**

```bash
# 1. correct the overstated "strictly harder" wording in the asymmetry_note,
#    then re-run the fetch -- fully cached now, ~1 min
#    (do NOT edit fetch_baselines.py while it is running: that mid-flight edit
#     is precisely what produced the stale registry this morning)
.venv/bin/python scripts/fetch_baselines.py

# 2. write the canonical metric check the tests require
.venv/bin/python scripts/verify_metric.py --task-ids 31 10101 3913 3 3917 11

# 3. commit the registry -- the tree must be CLEAN before the confirmatory run,
#    because runs now carry an ads/ digest and a dirty flag that sinks clause 3
git add runs/baselines.json runs/metric_check.json runs/evals && git commit

# 4. the 3-seed pre-registered screen (seeds come from run_protocol, not a flag)
ADS_N_JOBS=8 nohup .venv/bin/python scripts/run_benchmark.py \
    > logs/bench_screen.log 2>&1 &

# 5. regenerate RESULTS.md; never hand-type a number into it
.venv/bin/python scripts/report.py
```

**Current test state:** 37 pass. 7 fail, and all 7 are the stale/untracked
registry — they are the tests that caught it and they go green when step 3
lands. Nothing is failing for a reason I do not understand.

---

## Where the numbers will come from

`scripts/report.py` is the only code permitted to write a number into a
document, and it reads only `runs/*.json`. Every clause is derived by
`verdict()` from the run records — including `PASS`, which it will return
untouched if the runs earn it. What it now also refuses to grant:

- a task whose margin to the 5% line is smaller than its own seed range,
  unless the registered 8 seeds ran and an exact sign test rejects
  H0: median ≤ 0.95 × baseline (8/8 → p = 0.0039; 6/8 → p = 0.1445, not called);
- any set of runs containing a duplicated seed, a seed outside the registered
  protocol, a partial run, an off-registry run, a failed attempt, two different
  `ads/` digests, a dirty agent tree, or a ledger that does not reconcile
  against the files on disk.

`critique_log.md` is the long form, including the four attacks closed and the
three (#6 identity-as-property, #7 cache provenance, #8 label leakage) that are
**not** closed and are named rather than glossed.
