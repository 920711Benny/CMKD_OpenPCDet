"""The CarlaVLA CoT primitive vocabulary.

Taken from `cot_dataset_v3_calibrated.json` (634,293 records) and its validation
report, which lists the exact strings and their counts:

    lon: accelerate 125,242 | maintain speed 216,065 | brake 54,301
         remain stopped 218,988 | decelerate 19,697
    lat: go straight 496,480 | drift slightly right 41,639
         drift slightly left 45,934 | turn right 15,089 | turn left 12,549
         change lane to the left 10,960 | change lane to the right 11,642

This is a FACTORED action space -- longitudinal and lateral vary independently --
and that is strictly more informative than collapsing to one label. "brake while
turning left" and "brake while going straight" are different manoeuvres that a
single 8-way intent cannot separate. The two axes are therefore kept apart all
the way through: two classification heads, and a consistency loss that checks
each axis against the dynamics that axis actually controls.

The distribution is severely imbalanced -- `go straight` is 78% of lateral, and
`turn left` 2.0% -- so the class weights below are exposed for anyone training a
head on it. `remain stopped` and `maintain speed` together are 69% of
longitudinal, which is why an unweighted head trivially reaches high accuracy by
predicting them and learns nothing about braking.
"""
from __future__ import annotations

# Order is the label order for the two heads. Do not reorder without retraining.
LON_PRIMITIVES: tuple[str, ...] = (
    "remain stopped", "brake", "decelerate", "maintain speed", "accelerate",
)
LAT_PRIMITIVES: tuple[str, ...] = (
    "go straight", "drift slightly left", "drift slightly right",
    "turn left", "turn right",
    "change lane to the left", "change lane to the right",
)

LON_TO_ID = {p: i for i, p in enumerate(LON_PRIMITIVES)}
LAT_TO_ID = {p: i for i, p in enumerate(LAT_PRIMITIVES)}

# Counts from cot_dataset_v3_calibrated_validation_report.json.
LON_COUNTS = {"accelerate": 125_242, "maintain speed": 216_065, "brake": 54_301,
              "remain stopped": 218_988, "decelerate": 19_697}
LAT_COUNTS = {"go straight": 496_480, "drift slightly right": 41_639,
              "drift slightly left": 45_934, "turn right": 15_089,
              "turn left": 12_549, "change lane to the left": 10_960,
              "change lane to the right": 11_642}


def class_weights(counts: dict[str, int], order: tuple[str, ...],
                  scheme: str = "inverse_sqrt") -> list[float]:
    """Loss weights for an imbalanced head, normalised to mean 1.0.

    `inverse_sqrt` by default rather than plain inverse frequency: the rarest
    lateral class is 45x rarer than the commonest, and weighting by 45x makes
    the loss hostage to a handful of samples per batch.
    """
    total = sum(counts.values())
    n = len(order)
    raw = []
    for name in order:
        c = max(counts.get(name, 1), 1)
        freq = c / total
        raw.append((1.0 / freq) ** (0.5 if scheme == "inverse_sqrt" else 1.0))
    mean = sum(raw) / n
    return [w / mean for w in raw]


# Mapping onto the single-intent vocabulary used elsewhere in this repo, for
# code that has not been migrated to the factored heads. Lossy by construction:
# lateral wins when both axes are non-neutral, because a turn constrains the
# trajectory more visibly than the speed profile does.
_LAT_TO_INTENT = {
    "turn left": "turn_left", "turn right": "turn_right",
    "change lane to the left": "lane_change_left",
    "change lane to the right": "lane_change_right",
}
_LON_TO_INTENT = {
    "remain stopped": "stop", "brake": "decelerate", "decelerate": "decelerate",
    "maintain speed": "keep_speed", "accelerate": "accelerate",
}


def to_single_intent(lon: str, lat: str) -> str:
    """Collapse a (lon, lat) pair to one legacy intent. Prefer the factored
    form; this exists only for interoperability."""
    if lat in _LAT_TO_INTENT:
        # A full stop still outranks a turn: you cannot turn while stopped.
        if lon == "remain stopped":
            return "stop"
        return _LAT_TO_INTENT[lat]
    return _LON_TO_INTENT.get(lon, "keep_speed")


def normalise(primitive: str, axis: str) -> str:
    """Tolerate whitespace and case drift in a generated or hand-edited label."""
    p = " ".join(str(primitive or "").strip().lower().split())
    table = LON_TO_ID if axis == "lon" else LAT_TO_ID
    if p in table:
        return p
    aliases = {"maintain_speed": "maintain speed", "remain_stopped": "remain stopped",
               "go_straight": "go straight", "turn_left": "turn left",
               "turn_right": "turn right"}
    return aliases.get(p, "maintain speed" if axis == "lon" else "go straight")
