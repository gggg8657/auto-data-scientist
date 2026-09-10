"""The decision log.

The KPI's second clause is "end-to-end 무개입" — the agent picks preprocessing,
model and validation itself.  A claim like that is only auditable if every
choice is recorded *with the evidence that drove it*, so `record` refuses a
decision that carries no evidence.  Anything a human had to do to a run is an
`intervention`, and a run with interventions > 0 is a failed run, not a run to
be edited.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict


@dataclass
class Decision:
    stage: str
    choice: str
    rule: str
    evidence: dict
    alternatives: list = field(default_factory=list)
    t: float = field(default_factory=time.time)


class DecisionLog:
    def __init__(self) -> None:
        self.entries: list[Decision] = []
        self.interventions: list[dict] = []

    def record(self, stage: str, choice, rule: str, evidence: dict,
               alternatives=None) -> None:
        if not evidence:
            raise ValueError(
                f"decision {stage}={choice} recorded with no evidence; every "
                "choice must name the profile quantity that drove it")
        self.entries.append(Decision(stage, str(choice), rule, dict(evidence),
                                     list(alternatives or [])))

    def intervene(self, what: str, why: str) -> None:
        """Called only if a human touched the run.  Counted, never removed."""
        self.interventions.append({"what": what, "why": why, "t": time.time()})

    def to_dict(self) -> dict:
        return {
            "n_decisions": len(self.entries),
            "n_interventions": len(self.interventions),
            "decisions": [asdict(d) for d in self.entries],
            "interventions": self.interventions,
        }
