"""A record must be able to say what parallelism produced it.

`runs/clean` reproduced `runs/bench` exactly on three cells, and the obvious
follow-up -- did the operator's thread change at turn 8 move any result? -- was
unanswerable from the artifacts. Every record carried `cpu_count: 192` and
nothing else, so no record in this repository could report the parallelism or
the machine load it actually met. That gap is why the causal question stays
`[not measured]` for the historical records rather than merely unresolved.

These tests pin the shape of the fix. The failure mode being guarded is the one
this repository keeps rediscovering: absence coerced into a value. An unset
`OMP_NUM_THREADS` must record as `None`, not as the core count it would default
to and not as an empty string, because a record that reports a default nobody
set is worse than one that reports nothing.
"""
import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "rb", REPO / "scripts/run_benchmark.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def test_unset_thread_vars_record_as_none_not_as_a_default():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ADS_N_JOBS"):
        os.environ.pop(name, None)
    got = rb.thread_regime()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ADS_N_JOBS"):
        assert got["thread_env"][name] is None, (
            f"{name} was unset and must record as None, got "
            f"{got['thread_env'][name]!r}")
    assert got["thread_env_all_unset"] is True


def test_a_set_thread_var_is_recorded_verbatim():
    os.environ["OMP_NUM_THREADS"] = "7"
    try:
        got = rb.thread_regime()
        assert got["thread_env"]["OMP_NUM_THREADS"] == "7"
        assert got["thread_env_all_unset"] is False
    finally:
        os.environ.pop("OMP_NUM_THREADS", None)


def test_the_machine_state_is_recorded_alongside():
    """A run that cannot report the load it met cannot be compared with one
    that can -- this repo has already seen a fold go 25-40s -> 1341s on
    external contention."""
    got = rb.thread_regime()
    assert isinstance(got["affinity_count"], int) and got["affinity_count"] > 0
    assert isinstance(got["loadavg_1min_at_start"], float)


def test_threadpools_distinguishes_absent_from_empty():
    """`None` means the query failed; `[]` means it ran and found nothing.
    Collapsing those is how 'not measured' becomes 'measured as zero'."""
    got = rb.thread_regime()
    assert got["threadpools"] is None or isinstance(got["threadpools"], list)


def test_environment_carries_the_thread_regime():
    """The gap was that `environment()` recorded cpu_count and nothing else."""
    env = rb.environment()
    assert "thread_env" in env, "environment() does not carry the thread regime"
    assert "affinity_count" in env
    assert "cpu_count" in env, "the old field must not have been dropped"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
