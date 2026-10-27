"""Frozen negative-control definitions. This module does not execute controls."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

class ControlId(str, Enum):
    BROKEN_CONSTRAINT = "BROKEN_CONSTRAINT"
    PRODUCT_NO_HISTORY = "PRODUCT_NO_HISTORY"
    WRONG_CONDITIONING = "WRONG_CONDITIONING"
    HIDDEN_EXTERNAL_TIME = "HIDDEN_EXTERNAL_TIME"
    CLOCK_SCRAMBLE = "CLOCK_SCRAMBLE"
    POST_HOC_THRESHOLD_MUTATION = "POST_HOC_THRESHOLD_MUTATION"

@dataclass(frozen=True)
class ControlDefinition:
    control_id: ControlId
    protocol_section: str
    expected_failure_mode: str

CONTROL_DEFINITIONS = tuple(
    ControlDefinition(control, "12", description) for control, description in (
        (ControlId.BROKEN_CONSTRAINT, "constraint violation must be detected"),
        (ControlId.PRODUCT_NO_HISTORY, "full conditional orbit must not be reproduced"),
        (ControlId.WRONG_CONDITIONING, "wrong clock conditioning must fail comparison"),
        (ControlId.HIDDEN_EXTERNAL_TIME, "external-time construction must invalidate the run"),
        (ControlId.CLOCK_SCRAMBLE, "fixed non-cyclic scramble must fail comparison"),
        (ControlId.POST_HOC_THRESHOLD_MUTATION, "post-hoc mutation must invalidate the run"),
    )
)

CLOCK_SCRAMBLE = (1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12)
