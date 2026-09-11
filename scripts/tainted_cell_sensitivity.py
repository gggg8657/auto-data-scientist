#!/usr/bin/env python
"""Does clause 2's verdict depend on the operator-touched cell?

Clause 3 fails under its strict reading because one cell -- (10101, seed 6) --
was killed and restarted by an operator (critique_log.md 2026-09-11 turn 11b,
amendment 9).  The honest question that follows is whether the clause-2 verdict
*rests* on that cell.  If the task can still be called without it, the
intervention is a disclosure; if it cannot, it is a hole and the clause-2 PASS
is standing on a record the protocol says should have been counted as a failure.

No new runs: this re-runs the pre-registered exact sign test on the records
already on disk, once with every seed and once with the touched cell removed.
Registered in critique_log.md before it was run, and reported whichever way it
comes out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from report import (exact_sign_test_above, load_bench, read_json,  # noqa: E402
                    reconcile_ledger, tolerance_readings)


def main() -> int:
    base = read_json(REPO / "runs/baselines.json")
    bench, failed, partial = load_bench(REPO / "runs/bench")
    ledger = reconcile_ledger(REPO / "runs/attempts.jsonl", bench, failed)
    touched = [tuple(c) for c in (ledger.get("operator_touched_cells") or [])]

    by_task = {t["task_id"]: t for t in base["tasks"] if t.get("selected")}
    out = []
    for tid, runs in sorted(bench.items()):
        t = by_task.get(tid)
        if not t:
            continue
        thr = 0.95 * t["median_run"]
        dropped = [r for r in runs
                   if (tid, r.get("random_state")) in touched]
        kept = [r for r in runs
                if (tid, r.get("random_state")) not in touched]
        all_accs = [r["accuracy_pooled"] for r in runs]
        kept_accs = [r["accuracy_pooled"] for r in kept]
        full = exact_sign_test_above(all_accs, thr)
        less = exact_sign_test_above(kept_accs, thr)
        out.append({
            "task_id": tid,
            "dataset_name": t["dataset_name"],
            "threshold_primary": thr,
            "seeds_all": sorted(r.get("random_state") for r in runs),
            "seeds_dropped": sorted(r.get("random_state") for r in dropped),
            "with_every_seed": {"n": full["n"], "k": full["k"],
                                "p": full["p"], "called": full["reject_h0"]},
            "without_touched_cells": {"n": less["n"], "k": less["k"],
                                      "p": less["p"],
                                      "called": less["reject_h0"]},
            "verdict_depends_on_a_touched_cell": bool(
                full["reject_h0"] and not less["reject_h0"]),
        })

    rec = {
        "purpose": ("whether clause 2's per-task verdicts rest on any cell an "
                    "operator touched"),
        "registered_in": ("critique_log.md, 2026-09-11 turn 11b, before this "
                          "script was run"),
        "no_new_runs": True,
        "operator_touched_cells": [list(c) for c in touched],
        "protocol": ("the pre-registered one-sided exact sign test of "
                     "H0: median <= 0.95 x median_run, alpha = 0.05, ties "
                     "counted as non-wins; identical to the clause-2 gate, "
                     "applied to a strictly smaller seed set"),
        "floor_note": ("p has a floor of 1/2^n, so dropping a seed costs "
                       "power: n=8 floors at 0.0039 and n=7 at 0.0078, both "
                       "under alpha"),
        "tasks": out,
        "any_verdict_depends_on_a_touched_cell": any(
            t["verdict_depends_on_a_touched_cell"] for t in out),
    }
    p = REPO / "runs/tainted_cell_sensitivity.json"
    p.write_text(json.dumps(rec, indent=2) + "\n")
    for t in out:
        print(f"{t['task_id']:>6} {t['dataset_name']:<34} "
              f"all n={t['with_every_seed']['n']} k={t['with_every_seed']['k']} "
              f"p={t['with_every_seed']['p']:.4f} "
              f"{t['with_every_seed']['called']}   "
              f"| dropped {t['seeds_dropped']} -> "
              f"n={t['without_touched_cells']['n']} "
              f"k={t['without_touched_cells']['k']} "
              f"p={t['without_touched_cells']['p']:.4f} "
              f"{t['without_touched_cells']['called']}")
    print(f"\nany verdict depends on a touched cell: "
          f"{rec['any_verdict_depends_on_a_touched_cell']}  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
