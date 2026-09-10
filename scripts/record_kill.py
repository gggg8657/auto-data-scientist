"""Record that an attempt was stopped by an operator, with its evidence.

The gap this closes was named in `critique_log.md` two turns before it was
built: the ledger's vocabulary was `started` / `completed` / `failed`, and a
`SIGKILL` leaves a bare `started` — indistinguishable from the deletion the
ledger exists to catch. So "stop the slow cell and re-run it" was not available
as an *honest* act, even though re-running a fixed `(task, seed)` cannot shop
for a better number: the seed fixes the agent's randomness and the folds are
the task's own, so the re-run reproduces the same accuracy.

Why this is information rather than forgiveness, which is the only reason
`reconcile_ledger` counts a `killed` event as terminal:

- it names the pid, and **refuses to run if that pid is still alive**, so it
  cannot be used to write off an attempt that is merely inconvenient;
- it records the evidence that the attempt really ended — the absence of an
  `EXIT=` line in the runner's log, and the run file's absence from disk;
- the cell still has to be re-run, and the re-run's `completed` record is what
  the report counts. A killed cell with no later completion still shows up as
  an unrun cell.

It cannot delete or edit a ledger line, only append one.

First use, and the reason it exists: on 2026-09-10 pid 936115 held task 10101
seed 6 from 18:16:08; the other instance of this project's loop stopped it and
relaunched at 18:36:34 as pid 1493119 with `ADS_N_JOBS=8`. The stop was a
reasonable call on a saturated box and left no trace anywhere in the
repository. This is that trace.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pid", type=int, required=True,
                    help="the pid that held the attempt; must NOT be alive")
    ap.add_argument("--role", default="confirmatory")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--runner-log", default=None,
                    help="the log whose missing EXIT= line is the evidence")
    ap.add_argument("--replaced-by-pid", type=int, default=None)
    args = ap.parse_args()

    if Path(f"/proc/{args.pid}").exists():
        raise SystemExit(
            f"pid {args.pid} is still alive. A kill may only be recorded after "
            "the fact; recording one for a running attempt would let this "
            "script write off a result that is merely inconvenient.")

    ledger = RUNS / "attempts.jsonl"
    if not ledger.exists():
        raise SystemExit(f"{ledger} does not exist; nothing to reconcile")
    lines = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    starts = [e for e in lines
              if e.get("event") == "started" and e.get("task_id") == args.task_id
              and e.get("seed") == args.seed and e.get("pid") == args.pid]
    if not starts:
        raise SystemExit(
            f"no `started` line for task {args.task_id} seed {args.seed} by "
            f"pid {args.pid}; refusing to record the end of an attempt that "
            "was never recorded as beginning")
    already = [e for e in lines
               if e.get("event") in ("completed", "failed", "killed")
               and e.get("task_id") == args.task_id
               and e.get("seed") == args.seed and e.get("pid") == args.pid]
    if already:
        raise SystemExit(
            f"pid {args.pid} already has a terminal record for that cell: "
            f"{already[-1].get('event')}")

    evidence = {"holder_pid_alive": False,
                "started_utc": starts[-1].get("utc")}
    if args.runner_log:
        log = REPO / args.runner_log
        evidence["runner_log"] = args.runner_log
        evidence["runner_log_exists"] = log.exists()
        evidence["runner_log_has_exit_line"] = (
            log.exists() and any(l.startswith("EXIT=")
                                 for l in log.read_text().splitlines()))
    out_file = starts[-1].get("out_file")
    if out_file:
        evidence["out_file"] = out_file
        evidence["out_file_on_disk"] = (
            RUNS / ("bench" if args.role == "confirmatory" else "dev")
            / out_file).exists()

    from scripts.run_benchmark import register_attempt  # noqa: E402
    register_attempt(event="killed", task_id=args.task_id, seed=args.seed,
                     role=args.role, pid=args.pid, reason=args.reason,
                     replaced_by_pid=args.replaced_by_pid,
                     recorded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%S",
                                                   time.gmtime()),
                     recorded_after_the_fact=True, evidence=evidence)
    print(f"recorded killed: task {args.task_id} seed {args.seed} pid "
          f"{args.pid}")
    print(f"  evidence: {json.dumps(evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
