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
    for name in ("def exact_sign_test_above", "def escalation_state",
                 "def verdict"):
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        print(f"{f.__name__} ...")
        f()
    print(f"\n{len(fns)} tests passed")
