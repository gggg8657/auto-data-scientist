#!/usr/bin/env python
"""Does the untouched `clean` re-run reproduce the confirmatory set, cell by cell?

Clause 3 fails under its strict reading ("no operator touched any cell") because
one cell of `runs/bench` -- (10101, seed 6) -- was killed and restarted by an
operator at turn 8.  The `clean` role (turn 11) re-runs the identical registered
measurement into `runs/clean/` without touching it, so that both readings can be
reported side by side.

This script does NOT run any model.  It compares the records already on disk and
writes `runs/clean_reproduction.json`, which `report.py` reads.  Nothing here is
hand-typed into a document.

What counts as a reproduction.  The weak test is "the pooled accuracy agrees".
The strong test is that the agent made the *same decisions* -- the same model
family, the same tournament, the same preprocessing profile, on every fold -- so
the comparison is over a digest of the whole record with only the genuinely
volatile fields removed (wall-clock, environment, role, and the run's own
provenance of where it was written).  A cell that agrees on accuracy but not on
the digest is reported separately and is NOT counted as a reproduction.

On timing: `seconds_total` is recorded per cell for both sides, but a ratio near
1.0 means the two cells ran under a SIMILAR contention regime, not that accuracy
is invariant to contention.  This script deliberately does not make the second
claim; see critique_log.md 2026-09-11 turn 12.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "runs" / "bench"
CLEAN = REPO / "runs" / "clean"
OUT = REPO / "runs" / "clean_reproduction.json"

# Fields that legitimately differ between two runs of the same cell and so must
# not enter the identity digest.  Everything else must match exactly.
VOLATILE_RECORD = {"seconds_total", "env", "role", "off_registry"}
VOLATILE_FOLD = {"seconds"}
# Every decision the agent logs carries a wall-clock stamp `t`.  Two runs of the
# same cell cannot agree on it and it carries no information about what the
# agent decided, so it is stripped recursively before the digest.  It is the
# ONLY key stripped below the fold level; `strip_timestamps` counts what it
# removed so a silent broadening of this rule shows up as a changed count.
VOLATILE_NESTED = {"t"}

CELL_RE = re.compile(r"task_(\d+)_seed(\d+)\.json$")


def cells(d: Path) -> dict[tuple[int, int], Path]:
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("task_*_seed*.json")):
        m = CELL_RE.search(p.name)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = p
    return out


def strip_timestamps(obj, counter: list):
    """Recursively drop VOLATILE_NESTED keys, counting each removal."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in VOLATILE_NESTED:
                counter[0] += 1
                continue
            out[k] = strip_timestamps(v, counter)
        return out
    if isinstance(obj, list):
        return [strip_timestamps(v, counter) for v in obj]
    return obj


def canonical(rec: dict, counter: list | None = None) -> dict:
    """The record with volatile fields stripped, for the identity digest."""
    counter = counter if counter is not None else [0]
    out = {k: v for k, v in rec.items() if k not in VOLATILE_RECORD}
    if isinstance(out.get("per_fold"), list):
        out["per_fold"] = [
            {k: v for k, v in f.items() if k not in VOLATILE_FOLD}
            for f in out["per_fold"]
        ]
    return strip_timestamps(out, counter)


def digest(rec: dict) -> tuple[str, int]:
    counter = [0]
    blob = json.dumps(canonical(rec, counter), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest(), counter[0]


def main(bench_dir: Path | None = None, clean_dir: Path | None = None,
         out_path: Path | None = None) -> int:
    bench_dir = bench_dir if bench_dir is not None else BENCH
    clean_dir = clean_dir if clean_dir is not None else CLEAN
    out_path = out_path if out_path is not None else OUT
    b, c = cells(bench_dir), cells(clean_dir)
    compared, mismatches, accuracy_only = [], [], []

    for cell in sorted(set(b) & set(c)):
        rb = json.loads(b[cell].read_text())
        rc = json.loads(c[cell].read_text())
        # A partial record is not a reproduction of anything.
        if not (rb.get("complete") and rc.get("complete")):
            continue
        db, nb = digest(rb)
        dc, nc = digest(rc)
        acc_same = rb.get("accuracy_pooled") == rc.get("accuracy_pooled")
        row = {
            "task_id": cell[0], "seed": cell[1],
            "identical": db == dc,
            "accuracy_pooled_bench": rb.get("accuracy_pooled"),
            "accuracy_pooled_clean": rc.get("accuracy_pooled"),
            "accuracy_pooled_equal": acc_same,
            "accuracy_folds_equal":
                rb.get("accuracy_folds") == rc.get("accuracy_folds"),
            "families_equal":
                rb.get("families_chosen") == rc.get("families_chosen"),
            "n_correct_equal": rb.get("n_correct") == rc.get("n_correct"),
            "digest_bench": db, "digest_clean": dc,
            "timestamps_stripped_bench": nb,
            "timestamps_stripped_clean": nc,
            "seconds_total_bench": rb.get("seconds_total"),
            "seconds_total_clean": rc.get("seconds_total"),
            "ads_sha256_equal":
                (rb.get("env") or {}).get("ads_sha256")
                == (rc.get("env") or {}).get("ads_sha256"),
            "registry_sha256_equal":
                rb.get("registry_sha256") == rc.get("registry_sha256"),
        }
        st_b, st_c = row["seconds_total_bench"], row["seconds_total_clean"]
        row["seconds_ratio_clean_over_bench"] = (
            (st_c / st_b) if (st_b and st_c) else None)
        compared.append(row)
        if not row["identical"]:
            (accuracy_only if acc_same else mismatches).append(row)

    n_ident = sum(1 for r in compared if r["identical"])
    result = {
        "generated_by": "scripts/clean_reproduction.py",
        "what": "cell-by-cell identity of the untouched `clean` re-run against "
                "the confirmatory set `runs/bench`",
        "n_cells_bench": len(b),
        "n_cells_clean": len(c),
        "n_cells_compared": len(compared),
        "n_identical": n_ident,
        "n_accuracy_equal_digest_differs": len(accuracy_only),
        "n_mismatched": len(mismatches),
        "clean_coverage_of_bench":
            (len(compared) / len(b)) if b else None,
        # The clean set is still being produced; a partial set is reported as
        # partial rather than as a verdict.
        "clean_set_complete": len(compared) == len(b) and len(b) > 0,
        # `clean_set_complete` says only that every confirmatory cell has a
        # counterpart that was compared -- it reads True even when a cell failed
        # to reproduce. Clause 3 must be gated on the conjunction, or a complete
        # set containing a mismatch would read as a clean reproduction. Absence
        # or partiality makes this False, never None-coerced-to-pass.
        "all_cells_reproduced": (
            len(b) > 0 and len(compared) == len(b) and n_ident == len(b)),
        "cells": compared,
        "mismatched_cells": mismatches,
        "accuracy_equal_digest_differs_cells": accuracy_only,
        "caveats": [
            "A seconds ratio near 1.0 says the two cells met a SIMILAR "
            "contention regime. It is not evidence that accuracy is invariant "
            "to thread count or machine load; no run in this repository has "
            "measured that.",
            "Until clean_set_complete is true this is a partial reproduction "
            "and cannot settle clause 3 under either reading.",
        ],
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"compared {len(compared)} cells: {n_ident} identical, "
          f"{len(accuracy_only)} accuracy-equal/digest-differs, "
          f"{len(mismatches)} mismatched")
    print(f"clean covers {len(compared)}/{len(b)} of the confirmatory set "
          f"(complete={result['clean_set_complete']})")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
