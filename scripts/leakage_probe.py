"""Sever the agent's only legitimate channel to the labels and see if it still predicts.

Why this exists
---------------
Open item #8 was carried for three turns as "the labelled frame lives in the
same process as the agent, so the test labels are *reachable* from the pandas
views it is handed".  **That premise is false in this environment and I had
asserted it without checking** -- see `critique_log.md`, turn 9.  Measured on
pandas 3.0.2, where Copy-on-Write is mandatory: `X.iloc[tr]` shares no memory
with `X`, `y.iloc[tr].to_numpy()` shares no memory with `y`, and the parent
frame is not reachable from a column selection within four `gc` hops.
`tests/test_no_leakage.py` asserts all of that, so it goes red if a pandas
upgrade reintroduces aliasing.

What survives is the weaker and still worth measuring claim: an import audit
shows no *statically visible* route to the labels, and a structural check shows
no aliasing, but neither excludes a route through a module global, a fitted
object surviving across folds, or anything reached dynamically.  So this
measures the thing itself.

The instrument
--------------
Permute `y_train` before `fit`.  Leave `X_train`, `X_test` and the true test
labels exactly as they are.  Score against the **true** test labels.

- If `y_train` is the agent's only channel, severing it destroys performance:
  a classifier fit on shuffled labels concentrates on the prior, so accuracy
  falls to the test fold's **majority-class rate** -- not `1/C`.
- If information is arriving by some other route (a module global, a re-fetch
  of the task, a fitted object surviving across folds), accuracy stays near the
  intact number, because the permutation never touched that route.

This is the same label-permuted null that found FDR 1.00 in `yield-rca-agent`,
pointed at our own pipeline instead of at a result.

Pre-registered decision rule (see `critique_log.md`, turn 9; fixed before the
first run of this script)
-------------------------------------------------------------------------------
reference   = majority-class rate of the true test labels
noise band  = 2 * sqrt(p_maj * (1 - p_maj) / n_test)
NO_LEAKAGE  <=> max over the k permuted accuracies <= p_maj + noise band
LEAKAGE     <=> any permuted accuracy is closer to intact than to p_maj

Under LEAKAGE every accuracy in this repository is withdrawn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ads.agent import AutoDataScientist          # noqa: E402
from ads.openml_io import get_task, task_splits  # noqa: E402
# A probe is only evidence about the code it ran.  Same digest the benchmark
# records, so `report.py` can refuse a probe produced by a different `ads/`
# than the runs it is being asked to clear.
from scripts.run_benchmark import (agent_source_digest,  # noqa: E402
                                   environment)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "leakage_probe.json"

# Fixed by the pre-registration, not by this run.
K_PERMUTATIONS = 10
NOISE_SIGMAS = 2.0


def detection_threshold(p_maj: float, n_test: int) -> float:
    """The line a shuffled-label fit must not cross.  Pure, so it is testable
    without fitting anything."""
    return p_maj + NOISE_SIGMAS * math.sqrt(p_maj * (1.0 - p_maj) / n_test)


def flags_leakage(permuted_accs, p_maj: float, n_test: int) -> bool:
    """One-sided: leakage can only *raise* the permuted accuracy.  A permuted
    arm below the majority rate is an overfitting classifier, not evidence."""
    return any(a > detection_threshold(p_maj, n_test) for a in permuted_accs)


def _fit_predict(X_tr, y_tr, X_te, random_state: int, make_agent=None):
    """One agent, one fit, one predict.  A fresh agent every time on purpose:
    if a fitted object survived between calls that is itself a leakage route,
    and constructing a new one is what makes the permuted arm honest."""
    agent = (make_agent or AutoDataScientist)(random_state=random_state)
    agent.fit(X_tr, y_tr)
    return np.asarray(agent.predict(X_te)), getattr(agent, "best_name_", "?")


def probe_fold(X, y, tr, te, random_state: int, k: int,
               make_agent=None, verbose: bool = True) -> dict:
    """`make_agent` exists for the positive control.

    codex, asked the constructive question on 2026-09-10, made the point that
    settles it: "an invariant result could simply mean the intervention never
    reached the alleged channel".  A probe that has never been shown to fire on
    a leak it was told about is not an instrument, so
    `tests/test_no_leakage.py` runs this same function against an agent that
    deliberately reads the held-out labels and requires a LEAKAGE verdict."""
    y_te_true = y.iloc[te].to_numpy()
    y_tr_true = y.iloc[tr].to_numpy()

    vals, counts = np.unique(y_te_true, return_counts=True)
    p_maj = float(counts.max() / counts.sum())
    n_te = int(counts.sum())
    band = detection_threshold(p_maj, n_te) - p_maj

    t0 = time.time()
    pred, family = _fit_predict(X.iloc[tr], y_tr_true, X.iloc[te],
                                random_state, make_agent)
    acc_intact = float((pred == y_te_true).mean())
    intact_seconds = round(time.time() - t0, 2)
    if verbose:
        print(f"    intact: acc={acc_intact:.4f} family={family} "
              f"({intact_seconds}s)  p_maj={p_maj:.4f} band=+{band:.4f}",
              flush=True)

    permuted = []
    for j in range(k):
        # Permutation seed is independent of the agent's random_state so the
        # agent's own stochasticity cannot be confounded with the shuffle.
        rng = np.random.default_rng(10_000 + j)
        y_tr_perm = y_tr_true[rng.permutation(len(y_tr_true))]
        t1 = time.time()
        p, fam = _fit_predict(X.iloc[tr], y_tr_perm, X.iloc[te],
                              random_state, make_agent)
        acc = float((p == y_te_true).mean())
        permuted.append({
            "permutation": j, "accuracy": acc, "family": fam,
            "seconds": round(time.time() - t1, 2),
            # How far the permuted arm sits between the two hypotheses.  0 = at
            # the majority rate (no leakage), 1 = at the intact accuracy.
            "position": (None if acc_intact == p_maj
                         else float((acc - p_maj) / (acc_intact - p_maj))),
        })
        if verbose:
            print(f"    perm {j}: acc={acc:.4f} family={fam} "
                  f"({permuted[-1]['seconds']}s)", flush=True)

    accs = np.array([d["accuracy"] for d in permuted])
    # One-sided: leakage can only *raise* the permuted accuracy.  A permuted
    # arm below the majority rate is an overfitting classifier, not evidence.
    exceeds = accs > detection_threshold(p_maj, n_te)
    # Exact permutation p-value for "the intact accuracy is no better than a
    # shuffled-label fit", with the +1 that keeps it from ever reading 0.
    p_value = float((int((accs >= acc_intact).sum()) + 1) / (k + 1))

    return {
        "n_train": int(len(tr)), "n_test": n_te,
        "n_classes": int(len(vals)),
        "accuracy_intact": acc_intact,
        "family_intact": family,
        "majority_rate": p_maj,
        "noise_band_2sigma": float(band),
        "threshold": float(p_maj + band),
        "permuted": permuted,
        "permuted_max": float(accs.max()),
        "permuted_mean": float(accs.mean()),
        "permuted_min": float(accs.min()),
        "n_permutations_exceeding_threshold": int(exceeds.sum()),
        "max_position": float(max(d["position"] for d in permuted))
                        if all(d["position"] is not None for d in permuted) else None,
        "permutation_p_value": p_value,
        "verdict_fold": ("LEAKAGE" if flags_leakage(accs, p_maj, n_te)
                         else "NO_LEAKAGE_DETECTED"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # Default task is the one with the widest gap between an intact fit and the
    # majority rate among the registered five (kr-vs-kp: median published run
    # 0.95995, majority ~0.52), because that gap is the signal this probe has
    # to resolve.  A leak is hardest to hide there.
    # Which tasks, and how many folds each, is not a free choice: it is read
    # from `runs/leakage_power.json`, which measures the smallest fold count at
    # which this instrument can see a complete leak on each task.  Two of the
    # registered five are absent from that map at any fold count.
    ap.add_argument("--task", type=int, action="append", dest="tasks",
                    default=None, help="repeatable; default = every task the "
                                       "power file says is probeable")
    ap.add_argument("--folds", type=int, default=None,
                    help="default = the power file's min_folds_for_detection")
    ap.add_argument("--k", type=int, default=K_PERMUTATIONS)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    power = json.loads((REPO / "runs" / "leakage_power.json").read_text())
    min_folds = {int(k): v for k, v in power["min_folds_for_detection"].items()}
    task_ids = args.tasks if args.tasks else sorted(min_folds)
    unprobeable = [t for t in task_ids if t not in min_folds]
    if unprobeable and not args.folds:
        print(f"refusing tasks {unprobeable}: `runs/leakage_power.json` says "
              f"this instrument cannot resolve a complete leak on them at any "
              f"fold count. Pass --folds to override and the record will say "
              f"the probe was underpowered.", flush=True)
        task_ids = [t for t in task_ids if t in min_folds]
    if not task_ids:
        print("nothing probeable; not writing a record", flush=True)
        return 2

    results = []
    for tid in task_ids:
        n_folds = args.folds or min_folds[tid]
        print(f"leakage probe: task {tid}, {n_folds} fold(s) "
              f"(power file requires >={min_folds.get(tid, '-')}), "
              f"k={args.k} permutations", flush=True)
        task = get_task(tid)
        X, y = task.get_X_and_y(dataset_format="dataframe")
        y = pd.Series(y)

        folds = []
        for r, f, tr, te in task_splits(task):
            if len(folds) >= n_folds:
                break
            print(f"  fold {f} (repeat {r}):", flush=True)
            d = probe_fold(X, y, tr, te, args.random_state, args.k)
            d["repeat"], d["fold"] = int(r), int(f)
            folds.append(d)

        # The probe scores a fold *set*, so the operative reading is the
        # pooled one -- that is the band `leakage_power.py` sized the fold
        # count against.  The per-fold verdicts are kept beside it because on
        # kc2 zero individual folds can see a complete leak and six pooled
        # can, and a reader needs both numbers to know which was used.
        n_pool = sum(d["n_test"] for d in folds)
        acc_pool = sum(d["accuracy_intact"] * d["n_test"] for d in folds) / n_pool
        p_pool = sum(d["majority_rate"] * d["n_test"] for d in folds) / n_pool
        band_pool = NOISE_SIGMAS * math.sqrt(p_pool * (1 - p_pool) / n_pool)
        perm_pool = [
            sum(d["permuted"][j]["accuracy"] * d["n_test"] for d in folds) / n_pool
            for j in range(args.k)]
        pooled_leak = max(perm_pool) > p_pool + band_pool
        results.append({
            "task_id": int(tid),
            "dataset_name": task.get_dataset().name,
            "n_folds_probed": len(folds),
            "min_folds_required_by_power_file": min_folds.get(tid),
            "underpowered_by_fold_count": (
                min_folds.get(tid) is None or len(folds) < min_folds[tid]),
            "folds": folds,
            "pooled": {
                "n_test_total": n_pool,
                "accuracy_intact": acc_pool,
                "majority_rate": p_pool,
                "noise_band_2sigma": band_pool,
                "threshold": p_pool + band_pool,
                "permuted_accuracies": perm_pool,
                "permuted_max": max(perm_pool),
                "permuted_mean": sum(perm_pool) / len(perm_pool),
            },
            "verdict_task": "LEAKAGE" if pooled_leak else "NO_LEAKAGE_DETECTED",
            "n_folds_flagging_individually": sum(
                d["verdict_fold"] == "LEAKAGE" for d in folds),
        })
        print(f"  -> task {tid}: pooled intact {acc_pool:.4f} | majority "
              f"{p_pool:.4f} | permuted max {max(perm_pool):.4f} | threshold "
              f"{p_pool + band_pool:.4f} | "
              f"{results[-1]['verdict_task']}", flush=True)

    leak = [t for t in results if t["verdict_task"] == "LEAKAGE"]
    underpowered = [t["task_id"] for t in results
                    if t["underpowered_by_fold_count"]]
    record = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "empirical test of open item #8, in-process label leakage",
        "instrument": "permute y_train before fit; score against true y_test",
        "pre_registered_rule": {
            "reference": "majority-class rate of the true test labels",
            "noise_band": f"{NOISE_SIGMAS} * sqrt(p_maj*(1-p_maj)/n_test)",
            "no_leakage_iff": "max permuted accuracy <= majority rate + band",
            "consequence_if_leakage": "every accuracy in this repository is withdrawn",
            "fixed_in": "critique_log.md, 2026-09-10 turn 9, before this script ran",
        },
        "task_ids": [int(t) for t in task_ids],
        "k_permutations": int(args.k),
        "random_state": int(args.random_state),
        "ads_n_jobs": os.environ.get("ADS_N_JOBS"),
        "agent_imports_no_io_module": True,  # asserted by tests/test_no_leakage.py
        "env": {**agent_source_digest(), **environment()},
        "power_file_sha256": hashlib.sha256(
            (REPO / "runs" / "leakage_power.json").read_bytes()).hexdigest(),
        "tasks": results,
        "tasks_underpowered_by_fold_count": underpowered,
        "tasks_this_instrument_cannot_probe": sorted(
            power.get("tasks_unpowered_pooled") or []),
        "verdict": "LEAKAGE" if leak else "NO_LEAKAGE_DETECTED",
        "tasks_cleared": sorted(t["task_id"] for t in results
                                if t["verdict_task"] == "NO_LEAKAGE_DETECTED"
                                and not t["underpowered_by_fold_count"]),
        "scope": (
            "This measures the fold(s) and task probed, nothing wider.  It "
            "prices the reachability of the labelled frame through the pandas "
            "views the agent is handed; it cannot prove the absence of a route "
            "that only opens on data this probe did not run."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # .part + fsync + rename, for the reason recorded in critique_log.md turn 2:
    # a truncated write left a 4096-byte stump at the record path and the
    # exception did not remove it, so `exists()` trusted a prefix.
    tmp = args.out.with_suffix(".json.part")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.rename(args.out)

    print(f"\nverdict: {record['verdict']}  ->  {args.out}", flush=True)
    print(f"cleared: {record['tasks_cleared']}", flush=True)
    print(f"unprobeable by this instrument: "
          f"{record['tasks_this_instrument_cannot_probe']}", flush=True)
    return 1 if leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
