"""Chain-of-Causation prompt schema.

    [Perception: <critical objects>] -> [Causation: <impact analysis>]
                                     -> [Action Plan: <driving intent>]

The schema is deliberately compact and *parseable*: the Action Plan segment is
machine-readable so the consistency loss can compare a stated intent against the
trajectory the diffusion head actually produced. A rationale that cannot be
checked against the action is decoration, not reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical driving intents. Order is the label order used by the consistency
# loss and by the intent classification head.
INTENTS: tuple[str, ...] = (
    "stop",              # come to / remain at a full stop
    "decelerate",        # actively reduce speed
    "keep_speed",        # maintain current speed, straight
    "accelerate",        # increase speed
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
)
INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}

PERCEPTION_TAG = "[Perception:"
CAUSATION_TAG = "[Causation:"
ACTION_TAG = "[Action Plan:"

_ACTION_RE = re.compile(r"\[Action Plan:\s*([a-z_]+)", re.IGNORECASE)


@dataclass
class CoCSample:
    perception: str
    causation: str
    intent: str
    detail: str = ""
    safety_flag: bool = False
    critical_objects: list[str] = field(default_factory=list)

    def render(self) -> str:
        flag = " [SAFETY]" if self.safety_flag else ""
        detail = f" {self.detail}" if self.detail else ""
        return (
            f"{PERCEPTION_TAG} {self.perception}] -> "
            f"{CAUSATION_TAG} {self.causation}] -> "
            f"{ACTION_TAG} {self.intent}{detail}]{flag}"
        )


def parse_intent(text: str) -> tuple[str, int]:
    """Extract the driving intent from a rendered/generated CoC string.

    Falls back to `keep_speed` when the model emits something unparseable,
    which is the neutral option -- it must not silently become `stop`.
    """
    m = _ACTION_RE.search(text or "")
    if m:
        cand = m.group(1).lower()
        if cand in INTENT_TO_ID:
            return cand, INTENT_TO_ID[cand]
    return "keep_speed", INTENT_TO_ID["keep_speed"]


def build_prompt(instruction: str, speed_kmh: float, command: str) -> str:
    return (
        f"<image>\nEgo speed: {speed_kmh:.1f} km/h. Navigation: {command}. "
        f"Instruction: {instruction}\n"
        "Reason step by step as [Perception: ...] -> [Causation: ...] -> [Action Plan: ...].\n"
    )
