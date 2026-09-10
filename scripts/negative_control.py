"""Stage 2b -- is this KPI falsifiable at all?

Every other check in this repository asks whether *our* number is honest. This
one asks whether the *target* is demanding, which is a different question and,
on this evidence, the more damaging one.

`agy`, asked what single addition would most increase a sceptical reader's
belief in a PASS, put this first and it was right to:

    "On imbalanced datasets like blood-transfusion (majority share 76.2%) or
    kc1 (84.5%), the 0.95 x median_run threshold is 72.5% and 80.9%. A dumb
    DummyClassifier(strategy='prior') [...] clears the primary threshold on 4
    of the 5 tasks without learning anything. Does passing this clause
    actually prove data science competence, or is the bar unfalsifiable?"

The arithmetic is checkable before running anything, which is what makes it a
real objection rather than a hypothetical: on four of the five registered
tasks, `0.95 x median_run` sits *below* the majority-class rate.

So: two frozen negative controls, chosen for being obviously incapable rather
than for how they score, evaluated through **the same outer folds, the same
pooled metric and the same baselines** as the agent.

- `prior`: `DummyClassifier(strategy="prior")` -- predicts the training
  majority class, learns nothing from the features at all.
- `stump`: `DecisionTreeClassifier(max_depth=3)` -- three splits, no tuning,
  no preprocessing beyond what is needed to fit.

Neither is seeded meaningfully (`prior` is deterministic; the tree is fixed at
`random_state=0`), so there is no seed distribution to test and none is
claimed: these are point measurements of a fixed procedure, reported as such.

What this can show, and it is a result either way:

- if a control clears the clause on most tasks, the **clause is weak on those
  tasks** and our PASS on them says less than it appears to. That has to be in
  RESULTS.md beside the PASS, not in a footnote;
- if the controls fail the *joint* five-task criterion while the agent clears
  it, then the joint criterion is doing the discriminating work even where
  individual thresholds are soft, and that is the defensible reading.

`report.py` tabulates whatever this finds. Nothing here is tuned; if a control
turned out to beat the agent, that is the headline and the pipeline reports it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

CONTROLS = ("prior", "stump")


def build(name: str):
    """Frozen, deliberately incapable, and not chosen by how it scores."""
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier
    if name == "prior":
        return DummyClassifier(strategy="prior")
    if name == "stump":
        return DecisionTreeClassifier(max_depth=3, random_state=0)
    raise ValueError(name)


def encode(X: pd.DataFrame) -> pd.DataFrame:
    """The minimum that lets a bare sklearn estimator fit at all.

    Deliberately worse than the agent's encoding: ordinal codes for
    categoricals and the column median for missing values. A negative control
    that got the agent's preprocessing would be measuring the preprocessing.
    """
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("object", "category", "bool"):
            out[c] = pd.Categorical(out[c]).codes.astype(float)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def run_control(task_id: int, name: str) -> dict:
    from ads.openml_io import get_task, task_splits

    task = get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    y = pd.Series(y)
    Xe = encode(X)

    folds, n_correct, n_pred = [], 0, 0
    t0 = time.time()
    for r, f, tr, te in task_splits(task):
        est = build(name)
        est.fit(Xe.iloc[tr], y.iloc[tr].to_numpy())
        pred = est.predict(Xe.iloc[te])
        hit = (np.asarray(pred) == y.iloc[te].to_numpy())
        n_correct += int(hit.sum())
        n_pred += int(hit.size)
        folds.append(float(hit.mean()))
    return {
        "task_id": int(task_id), "control": name,
        "dataset_name": task.get_dataset().name,
        # identical statistic to the agent's, so the rows are comparable
        "accuracy_pooled": float(n_correct / n_pred),
        "n_correct": int(n_correct), "n_predictions": int(n_pred),
        "accuracy_folds": folds,
        "n_folds": len(folds),
        "majority_class_rate": float(y.value_counts(normalize=True).max()),
        "seconds": round(time.time() - t0, 2),
        "deterministic": name == "prior",
    }


def write_json_atomic(path: Path, obj) -> None:
    text = json.dumps(obj, indent=2)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(RUNS / "baselines.json"))
    ap.add_argument("--out", default=str(RUNS / "negative_control.json"))
    ap.add_argument("--controls", nargs="*", default=list(CONTROLS))
    args = ap.parse_args()

    base = json.loads(Path(args.baselines).read_text())
    selected = [t for t in base["tasks"] if t["selected"]]
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "metric": base["metric"],
           "why": "Is the KPI falsifiable? Frozen incapable procedures through "
                  "the same outer folds, the same pooled metric and the same "
                  "pre-registered baselines as the agent.",
           "controls": {}}

    for name in args.controls:
        rows = []
        for t in selected:
            rec = run_control(t["task_id"], name)
            thr = 0.95 * t["median_run"]
            thr_strict = 0.95 * t["strictest_baseline_value"]
            rec["threshold_primary"] = float(thr)
            rec["threshold_strictest"] = float(thr_strict)
            rec["clears_primary"] = bool(rec["accuracy_pooled"] >= thr)
            rec["clears_strictest"] = bool(
                rec["accuracy_pooled"] >= thr_strict)
            # the check that does not need a run at all, kept because it is
            # the form of the objection: is the bar under the majority rate?
            rec["threshold_below_majority_rate"] = bool(
                thr <= rec["majority_class_rate"])
            print(f"  {name:6s} task {t['task_id']:6d} "
                  f"{rec['dataset_name']:34s} acc={rec['accuracy_pooled']:.4f} "
                  f"thr={thr:.4f} clears={rec['clears_primary']} "
                  f"strict={rec['clears_strictest']}", flush=True)
            rows.append(rec)
        out["controls"][name] = {
            "tasks": rows,
            "n_tasks": len(rows),
            "n_clearing_primary": sum(r["clears_primary"] for r in rows),
            "n_clearing_strictest": sum(r["clears_strictest"] for r in rows),
            "clears_all_five_primary": all(r["clears_primary"] for r in rows),
            "clears_all_five_strictest": all(
                r["clears_strictest"] for r in rows),
        }

    out["n_thresholds_below_majority_rate"] = sum(
        r["threshold_below_majority_rate"]
        for r in out["controls"][args.controls[0]]["tasks"])
    out["any_control_clears_all_five"] = any(
        c["clears_all_five_primary"] for c in out["controls"].values())
    write_json_atomic(Path(args.out), out)
    print(f"wrote {args.out}")
    for name, c in out["controls"].items():
        print(f"  {name}: clears primary on {c['n_clearing_primary']}/"
              f"{c['n_tasks']}, strictest on {c['n_clearing_strictest']}/"
              f"{c['n_tasks']}, all five: {c['clears_all_five_primary']}")
    print(f"  thresholds at or below the majority-class rate: "
          f"{out['n_thresholds_below_majority_rate']}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
