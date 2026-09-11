"""Every post-registration protocol change must be recorded, and must be strict.

`runs/baselines.json` is frozen and its sha256 is stamped into every run record,
so the registry cannot be edited without orphaning the runs measured against it.
That protects the *targets*. It does not protect the *rules*, which live partly
in `scripts/report.py` and can be rewritten at any time -- and rewriting a rule
after seeing a result is the way this KPI gets faked without any number being
edited.

So: `runs/protocol_amendments.json` is the record, and this test is the gate on
it. The one rule it enforces is the one that matters -- **no amendment may make
a clause easier**. Reporting a strict reading beside a conventional one is
allowed and is not an amendment; silently replacing one with the other is what
this forbids.

The test cannot know whether the file is complete. What it can do is make an
undeclared amendment cost something: `scripts/report.py` is fingerprinted here,
so changing the gate without recording why turns this red.
"""
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AMEND = REPO / "runs/protocol_amendments.json"
REPORT = REPO / "scripts/report.py"

REQUIRED = ("id", "date", "clause", "registered_rule", "amended_rule",
            "direction", "results_already_seen_when_amended", "why")

# Every function that can change what the verdict is, not just the three that
# compute the test. `agy`, asked what could still be faked, found (its #4) that
# the fingerprint covered `exact_sign_test_above`, `escalation_state` and
# `verdict` only -- so `load_bench` could silently skip `.FAILED.json` files,
# `reconcile_ledger` could return a hard-coded reconciled:True, and the
# `ours`/`primary_pass` computation could change, all with the fingerprint
# still matching. Direction: flattering. `main()` is not a function whose body
# can be isolated this way, so `ours` and `primary_pass` are pinned by
# `tests/test_report.py` behaviourally instead, and the readers they depend on
# are fingerprinted here.
GATE_FUNCTIONS = ("def exact_sign_test_above", "def escalation_state",
                  "def verdict", "def load_bench", "def reconcile_ledger",
                  "def tolerance_readings", "def joint_seed_event",
                  "def clean_seed_subset",
                  # added 2026-09-11 turn 11: this decides which leakage
                  # clearances reach `verdict`, so an undeclared edit to it is
                  # an undeclared edit to the gate.
                  "def merge_leakage_records")


def test_the_amendment_file_exists_and_is_wellformed():
    assert AMEND.exists(), (
        "runs/protocol_amendments.json is missing. The exact test that gates "
        "clause 2 was written on 2026-09-10, after the 3-seed screen was read; "
        "the registry still carries the conditional rule it replaced. That is "
        "an amendment and it has to be on the record.")
    a = json.loads(AMEND.read_text())
    assert a["amendments"], "no amendments recorded"
    for e in a["amendments"]:
        for f in REQUIRED:
            assert f in e, f"amendment {e.get('id')} is missing {f!r}"
    ids = [e["id"] for e in a["amendments"]]
    assert ids == sorted(set(ids)), f"ids not unique and ascending: {ids}"
    print(f"  {len(a['amendments'])} amendments recorded, all fields present")


def test_no_amendment_makes_a_clause_easier():
    """The one thing the rules say may never be done."""
    a = json.loads(AMEND.read_text())
    for e in a["amendments"]:
        assert e["direction"] == "stricter", (
            f"amendment {e['id']} on clause {e['clause']} is recorded as "
            f"{e['direction']!r}. Loosening a test after seeing a result is "
            "the one change that is never allowed; report both readings "
            "instead.")
    print(f"  all {len(a['amendments'])} amendments recorded as stricter")


def test_an_amendment_made_after_seeing_results_says_what_it_did_to_the_claim():
    """Chronology is not fatal, but it has to be visible.

    An amendment made after seeing results is defensible when it *withdrew* a
    claim (as #1 did: PASS -> RUNNING on identical data) and indefensible when
    it granted one. So any amendment flagged as post-hoc must state both what
    had been seen and what the amendment did to the claim as it stood.
    """
    a = json.loads(AMEND.read_text())
    post_hoc = [e for e in a["amendments"]
                if e["results_already_seen_when_amended"]]
    for e in post_hoc:
        assert e.get("what_had_been_seen"), e["id"]
        assert e.get("effect_on_the_claim_as_it_stood"), e["id"]
        eff = e["effect_on_the_claim_as_it_stood"].lower()
        assert ("none" in eff or "withdrew" in eff or "->" in eff), (
            f"amendment {e['id']} was made after results were seen and does "
            "not state whether it withdrew or granted a claim")
    print(f"  {len(post_hoc)} post-hoc amendments each state what had been "
          "seen and what the amendment did to the claim")


def test_the_gate_code_is_fingerprinted_so_an_undeclared_change_costs_something():
    """A recorded fingerprint of the functions that decide the verdict.

    This is not tamper-proofing -- anyone can update the number below. It makes
    an *undeclared* rule change fail a test, which is the difference between a
    silent edit and a deliberate one.
    """
    src = REPORT.read_text()
    body = []
    for name in GATE_FUNCTIONS:
        i = src.index(name)
        # to the next top-level def, which is where the function ends
        j = src.find("\ndef ", i + 1)
        chunk = src[i:j if j > 0 else len(src)]
        # comments and docstrings carry the reasoning, not the rule; strip them
        # so that documenting a decision does not read as changing it
        chunk = re.sub(r'""".*?"""', "", chunk, flags=re.S)
        chunk = re.sub(r"^\s*#.*$", "", chunk, flags=re.M)
        chunk = re.sub(r"\s+", " ", chunk)
        body.append(chunk)
    digest = hashlib.sha256("".join(body).encode()).hexdigest()
    a = json.loads(AMEND.read_text())
    recorded = a.get("gate_code_sha256")
    assert recorded == digest, (
        f"the verdict gate's code has changed (sha256 {digest[:16]}, recorded "
        f"{str(recorded)[:16]}). If that was a protocol change, add an entry to "
        "runs/protocol_amendments.json and update gate_code_sha256. If it was "
        "a refactor with no behavioural change, just update gate_code_sha256 "
        "and say so in the commit message.")
    print(f"  gate code matches the recorded fingerprint {digest[:16]}")


def test_the_clean_seed_subset_comes_from_the_amendment_not_from_the_code():
    """The subset must be pre-specified, not chosen after seeing outcomes.

    The whole value of the "reading that owes nothing to the pre-inspected
    seeds" table is that the seeds in it were named *in the amendment*, before
    they had been run. If report.py chose them itself -- or if the amendment
    could be edited to name whichever seeds happened to look best -- the table
    would be a subgroup analysis dressed as a confirmation, and it would bias
    flatteringly.
    """
    import runpy
    R = runpy.run_path(str(REPORT))
    a = json.loads(AMEND.read_text())

    got = R["clean_seed_subset"](AMEND)
    named = set()
    for e in a["amendments"]:
        named |= set(e.get("seeds_uninspected_at_amendment") or [])
    assert got == named and got, (got, named)

    # and they must be disjoint from the seeds the amendment admits were seen
    seen = set()
    for e in a["amendments"]:
        seen |= set(e.get("seeds_already_inspected_at_amendment") or [])
    assert got.isdisjoint(seen), (
        f"seeds {sorted(got & seen)} are listed both as inspected before the "
        "amendment and as clean")

    # an absent record yields an empty set, so the table disappears rather
    # than falling back to a subset this code picked
    assert R["clean_seed_subset"](AMEND.parent / "nope.json") == set()
    print(f"  clean subset {sorted(got)} read from the amendment, disjoint "
          f"from the inspected seeds {sorted(seen)}")


def test_the_concrete_knobs_of_the_gate_cannot_move_silently():
    """"direction: stricter" is a string, so pin the numbers it describes.

    `agy`, asked what could still be faked (its #5), is right that
    `test_no_amendment_makes_a_clause_easier` checks a JSON field and cannot
    verify mathematical strictness: loosen alpha to 0.10, update the
    fingerprint, write "stricter", and every test passes. A test cannot prove
    strictness in general. What it can do is pin the specific knobs a
    loosening would have to turn, so that turning one is a visible diff in a
    test file rather than a one-character edit in a script.

    These are the knobs, and the direction each would move the clause:
    """
    import runpy
    R = runpy.run_path(str(REPORT))

    # alpha: raising it makes rejection easier -> flattering
    assert R["ALPHA"] == 0.05, (
        f"alpha is {R['ALPHA']}, not the 0.05 every p-value in RESULTS.md is "
        "reported against")

    # the full registered seed set is required -> removing this admits
    # optional stopping, which is flattering
    protocol = {"seeds_screen": [0, 1, 2], "seeds_verdict": list(range(8))}
    for n in (5, 6, 7):
        e = R["escalation_state"]([0.90] * n, 0.80, protocol)
        assert e["exact_test"]["reject_h0"] and not e["called"], (n, e)

    # ties are non-wins in the denominator -> dropping them is
    # anti-conservative, hence flattering
    t = R["exact_sign_test_above"]([1.0, 0.5, 0.5, 0.5], 0.5)
    assert t["n"] == 4 and t["k"] == 1, t

    # the tolerance is one-sided at 0.95x and EPS cannot widen it
    assert R["tolerance_readings"](0.95, 1.0)["rel_one_sided"] is True
    assert R["tolerance_readings"](0.95 - 1e-6, 1.0)["rel_one_sided"] is False
    assert R["EPS"] <= 1e-9, R["EPS"]

    # a strictly-clearing 8/8 is called, so the pins above have not made the
    # gate unreachable in the other direction
    e8 = R["escalation_state"]([0.90] * 8, 0.80, protocol)
    assert e8["called"] is True and e8["exact_test"]["p"] == 1 / 256
    print("  alpha=0.05, 8 seeds required, ties conservative, tolerance "
          "one-sided at 0.95x, EPS<=1e-9 -- all pinned")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
