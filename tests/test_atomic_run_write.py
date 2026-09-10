"""A run record must be complete or absent, never a truncated prefix.

`run_benchmark.py` claimed that guarantee in its module docstring and did not
have it.  On 2026-09-10 this volume filled (0 bytes free on a 7.0T ext4 shared
with other work on this box) and `Path.write_text` left a 4096-byte prefix of a
task record at the record's own path.  It parsed as nothing -- `json.loads`
raised -- but the failure mode that matters is the other one: a prefix of a
JSON document can be *shorter but well-formed* after a truncation, and a reader
that tolerates it would have scored a task on however many folds happened to
land inside the first page.  The exception `write_text` raised did not remove
the file, so the next run's `out.exists()` check skipped the task and kept the
stump.

These tests pin the three properties the fix has to have, so a later
refactor cannot quietly drop them.
"""
import errno
import json
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_benchmark as RB  # noqa: E402


def test_successful_write_leaves_the_record_and_no_part_file(tmp_path):
    out = tmp_path / "task_1_seed0.json"
    rec = {"task_id": 1, "accuracy_pooled": 0.5, "per_fold": [{"fold": i} for i in range(10)]}
    RB.write_json_atomic(out, rec)
    assert json.loads(out.read_text()) == rec
    assert list(tmp_path.glob("*.part")) == [], "a .part file survived a successful write"
    print("  record written, parses back identical, no .part left behind")


def test_enospc_leaves_no_file_at_the_record_path(tmp_path):
    """The bug: the partial file survived at the path the reader trusts."""
    out = tmp_path / "task_1_seed0.json"
    with mock.patch.object(RB.os, "fsync",
                           side_effect=OSError(errno.ENOSPC, "No space left on device")):
        try:
            RB.write_json_atomic(out, {"task_id": 1})
        except OSError as e:
            assert e.errno == errno.ENOSPC
        else:
            raise AssertionError("write_json_atomic swallowed ENOSPC")
    assert not out.exists(), (
        "a failed write left a file at the record path; run_benchmark's "
        "out.exists() check would skip the task and trust this stump")
    assert list(tmp_path.glob("*.part")) == [], "the .part file was not cleaned up"
    print("  ENOSPC left neither a record nor a .part file")


def test_enospc_does_not_destroy_an_existing_record(tmp_path):
    """Rename is atomic, so a re-run that fails must leave the old record."""
    out = tmp_path / "task_1_seed0.json"
    good = {"task_id": 1, "accuracy_pooled": 0.77, "n_folds_run": 10}
    RB.write_json_atomic(out, good)
    with mock.patch.object(RB.os, "fsync",
                           side_effect=OSError(errno.ENOSPC, "No space left on device")):
        try:
            RB.write_json_atomic(out, {"task_id": 1, "accuracy_pooled": 0.0})
        except OSError:
            pass
    assert json.loads(out.read_text()) == good, (
        "a failed rewrite corrupted the record that was already on disk")
    print("  a failed rewrite left the previous complete record intact")


def test_a_truncated_record_on_disk_is_not_silently_readable(tmp_path):
    """Documents the shape of the artifact the outage actually produced."""
    out = tmp_path / "task_3913_seed99.json"
    full = json.dumps({"task_id": 3913, "accuracy_pooled": 0.7735849056603774,
                       "per_fold": [{"fold": i, "accuracy": 0.5} for i in range(10)]},
                      indent=2)
    out.write_text(full[:4096] if len(full) > 4096 else full[: len(full) // 2])
    try:
        json.loads(out.read_text())
    except json.JSONDecodeError:
        print("  a truncated record fails json.loads, which is how it was caught")
    else:
        raise AssertionError(
            "a truncated record parsed cleanly -- the report must therefore "
            "validate n_folds_run against the task's fold count, not just parse")


if __name__ == "__main__":
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        with tempfile.TemporaryDirectory() as d:
            f(Path(d))
    print(f"\n{len(fns)} tests passed")
