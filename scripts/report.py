"""Regenerate RESULTS.md and the README baseline block from `runs/*.json`.

This is the only code in the repository permitted to write a number into a
document.  It reads `runs/baselines.json` (stage 1) and `runs/bench/*.json`
(stage 2) and nothing else; if a stage has not run, its rows read
`[not measured]` rather than being omitted, because an omitted row is how a
report ends up flattering.

The project verdict is **computed** by `verdict()` from the clause rows, never
typed.  A previous project in this workspace shipped a hard-coded "both clauses
are met" string that survived the numbers moving underneath it.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NM = "[not measured]"


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def cell(x):
    return str(x).replace("|", "\\|")


def table(rows, header):
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, n=4):
    return NM if x is None else f"{x:.{n}f}"


# ------------------------------------------------------------------ tolerance
# A result landing exactly on the 5% line must not be decided by float noise:
# |0.95 - 1.00| / 1.00 evaluates to 0.050000000000000044 in IEEE754, which would
# fail a run that is exactly at tolerance. EPS is 1e-12 -- eleven orders of
# magnitude below any accuracy difference this benchmark can resolve, so it can
# only ever rescue an exact tie, never widen the tolerance.
EPS = 1e-12


def tolerance_readings(ours: float, base: float) -> dict:
    """The three readings of '±5% 이내', all fixed in runs/baselines.json."""
    return {
        "rel_one_sided": bool(ours >= 0.95 * base - EPS),     # primary
        "rel_two_sided": bool(abs(ours - base) / base <= 0.05 + EPS),
        "abs_two_sided": bool(abs(ours - base) <= 0.05 + EPS),
        "rel_gap": float((ours - base) / base),
        "abs_gap": float(ours - base),
    }


def load_bench(bench_dir: Path) -> dict[int, list[dict]]:
    by_task: dict[int, list[dict]] = {}
    for f in sorted(bench_dir.glob("task_*_seed*.json")):
        if f.name.endswith(".FAILED.json"):
            continue
        r = read_json(f)
        if r is None:
            continue
        r["_file"] = f.name
        by_task.setdefault(int(r["task_id"]), []).append(r)
    return by_task


def verdict(rows: list[dict], base: dict, bench: dict) -> dict:
    """Derive the project status from the clause rows. Never a typed string."""
    n_tasks = len(base.get("selected_task_ids", [])) if base else 0
    clause1 = n_tasks == 5
    measured = [r for r in rows if r["ours"] is not None]
    clause2 = bool(measured) and len(measured) == n_tasks and all(
        r["primary_pass"] for r in measured)
    interventions = sum(
        run.get("n_interventions", 0) for runs in bench.values() for run in runs)
    off_registry = any(run.get("off_registry") for runs in bench.values()
                       for run in runs)
    incomplete = [run["_file"] for runs in bench.values() for run in runs
                  if not run.get("complete", False)]
    clause3 = bool(measured) and interventions == 0 and not off_registry \
        and not incomplete
    clauses = {
        "1_five_public_datasets": clause1,
        "2_within_5pct_of_human_baseline": clause2 if measured else None,
        "3_end_to_end_no_intervention": clause3 if measured else None,
    }
    if all(v is True for v in clauses.values()):
        status = "PASS"
    elif any(v is None for v in clauses.values()):
        status = "RUNNING (not all clauses measured)"
    else:
        status = "NOT MET as measured"
    return {"status": status, "clauses": clauses,
            "n_interventions_total": interventions,
            "off_registry_runs": off_registry,
            "incomplete_runs": incomplete,
            "n_tasks_measured": len(measured), "n_tasks_registered": n_tasks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=str(REPO / "runs/baselines.json"))
    ap.add_argument("--bench", default=str(REPO / "runs/bench"))
    ap.add_argument("--out", default=str(REPO / "RESULTS.md"))
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    args = ap.parse_args()

    base = read_json(args.baselines)
    bench = load_bench(Path(args.bench)) if Path(args.bench).exists() else {}

    doc: list[str] = ["# Results", ""]
    if base is None:
        doc += [f"`runs/baselines.json` is absent, so every number here is "
                f"`{NM}`. Run `scripts/fetch_baselines.py` first.", ""]
        Path(args.out).write_text("\n".join(doc) + "\n")
        print(f"wrote {args.out} (no baselines yet)")
        return 0

    sel = {t["task_id"]: t for t in base["tasks"] if t["selected"]}
    order = base["selected_task_ids"]

    # ---------------------------------------------------------- baseline block
    bl_rows = []
    for tid in order:
        t = sel[tid]
        bl_rows.append([
            tid, t["dataset_name"], t["n_instances"], t["n_features"],
            t["n_classes"], t["n_published_runs"], t["n_distinct_flows"],
            t["n_distinct_uploaders"], fmt(t["median_run"]),
            fmt(t["median_flow"]), fmt(t["q90"]), fmt(t["max_published"])])
    bl_table = table(bl_rows, [
        "task", "dataset", "n", "p", "classes", "published runs", "flows",
        "uploaders", "**median_run** (target)", "median_flow", "q90 (context)",
        "max (context)"])

    rule = base["selection_rule"]
    bl_block = "\n".join([
        "### The five tasks and their pre-registered human baselines", "",
        f"Fetched by `scripts/fetch_baselines.py` on **{base['generated']}**, "
        f"before any agent run existed. Selected from {rule['source']} by: "
        f"`{rule['filter']}`, then the top {rule['top_k']} of "
        f"{rule['n_with_evals']} by `{rule['rank_by']}`.", "",
        bl_table, "",
        f"Metric is `{base['metric']}` under each task's own estimation "
        "procedure, which is what makes our number and a published run "
        "comparable. Raw evaluations are committed under `runs/evals/` with a "
        "sha256 per file in `runs/baselines.json`.", ""])

    # ------------------------------------------------------------ our results
    rows, res_rows, spread_rows = [], [], []
    for tid in order:
        t = sel[tid]
        runs = bench.get(tid, [])
        accs = [r["accuracy_pooled"] for r in runs]
        ours = statistics.mean(accs) if accs else None
        row = {"task_id": tid, "name": t["dataset_name"], "ours": ours,
               "primary_pass": None}
        if ours is not None:
            tol_run = tolerance_readings(ours, t["median_run"])
            tol_flow = tolerance_readings(ours, t["median_flow"])
            row["primary_pass"] = tol_run["rel_one_sided"]
            res_rows.append([
                tid, t["dataset_name"], len(accs), fmt(ours),
                fmt(t["median_run"]), f"{tol_run['rel_gap']*100:+.2f}%",
                "PASS" if tol_run["rel_one_sided"] else "FAIL",
                "PASS" if tol_run["rel_two_sided"] else "FAIL",
                "PASS" if tol_run["abs_two_sided"] else "FAIL",
                fmt(t["median_flow"]),
                "PASS" if tol_flow["rel_one_sided"] else "FAIL",
                ",".join(sorted({f for r in runs for f in r["families_chosen"]}))])
            if len(accs) >= 2:
                spread_rows.append([
                    tid, t["dataset_name"], len(accs), fmt(min(accs)),
                    fmt(max(accs)), fmt(max(accs) - min(accs)),
                    fmt(statistics.pstdev(accs)),
                    f"{abs(tol_run['rel_gap'])*100:.2f}%"])
        else:
            res_rows.append([tid, t["dataset_name"], 0, NM,
                             fmt(t["median_run"]), NM, NM, NM, NM,
                             fmt(t["median_flow"]), NM, NM])
        rows.append(row)

    v = verdict(rows, base, bench)

    doc += [f"**Status: {v['status']}**", "",
            "Every number below was produced by a run in this repository and is "
            "regenerated from `runs/*.json` by `scripts/report.py`. Nothing is "
            "hand-typed.", "",
            "## KPI, clause by clause", "",
            table([["1", "공개 데이터셋 5개",
                    f"{v['n_tasks_registered']} registered, "
                    f"{v['n_tasks_measured']} measured",
                    str(v["clauses"]["1_five_public_datasets"])],
                   ["2", "사람 baseline ±5% 이내 자동도달",
                    "primary reading: ours >= 0.95 x median_run, on all five",
                    str(v["clauses"]["2_within_5pct_of_human_baseline"])],
                   ["3", "end-to-end 무개입",
                    f"{v['n_interventions_total']} interventions logged; "
                    f"off-registry runs: {v['off_registry_runs']}; "
                    f"incomplete: {len(v['incomplete_runs'])}",
                    str(v["clauses"]["3_end_to_end_no_intervention"])]],
                  ["#", "clause", "measured as", "met"]), "",
            "## Our accuracy against the pre-registered baselines", "",
            table(res_rows, [
                "task", "dataset", "seeds", "ours (pooled)",
                "median_run", "rel gap", "primary (>=0.95x)", "rel 2-sided",
                "abs 2-sided", "median_flow", "vs flow", "families chosen"]),
            "",
            "`rel gap` is `(ours - median_run) / median_run`; positive means we "
            "are above the median published run. The three tolerance columns are "
            "the three readings fixed in `runs/baselines.json` before any run — "
            "all are shown so that none can be picked after the fact.", ""]

    if spread_rows:
        doc += ["## Our own run-to-run spread", "",
                "An effect smaller than this noise floor is not an effect. Each "
                "seed re-draws the agent's inner CV shuffle, its selection "
                "subsample and its random search; the outer folds are the "
                "task's own and are identical across seeds.", "",
                table(spread_rows, ["task", "dataset", "seeds", "min", "max",
                                    "range", "sd", "|gap to baseline|"]), ""]
    else:
        doc += ["## Our own run-to-run spread", "",
                f"{NM} — fewer than two seeds per task so far.", ""]

    doc += [bl_block, "## Provenance", "",
            table([[k, str(vv)] for k, vv in v.items() if k != "clauses"],
                  ["field", "value"]), ""]

    Path(args.out).write_text("\n".join(doc) + "\n")

    # ------------------------------------------------- README baseline block
    rp = Path(args.readme)
    if rp.exists():
        txt = rp.read_text()
        new = re.sub(r"<!-- BASELINES:BEGIN -->.*?<!-- BASELINES:END -->",
                     "<!-- BASELINES:BEGIN -->\n" + bl_block +
                     "<!-- BASELINES:END -->", txt, flags=re.S)
        rp.write_text(new)

    print(f"wrote {args.out} and refreshed the README block")
    print(f"status: {v['status']}")
    for k, val in v["clauses"].items():
        print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
