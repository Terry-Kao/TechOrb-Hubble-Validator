"""Frozen negative-control definitions. This module does not execute controls."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np

from .conditional_state import clock_phase_state, partial_trace_clock
from .protocol_binding import D_C, D_S, frozen_binding_snapshot

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

class ConstructionOrigin(str, Enum):
    DIRECT_CONSTRAINED_STATE = "DIRECT_CONSTRAINED_STATE"
    EXTERNAL_CLASSICAL_TIME_THEN_PACKAGED = "EXTERNAL_CLASSICAL_TIME_THEN_PACKAGED"

@dataclass(frozen=True)
class ConstructionProvenance:
    origin: ConstructionOrigin
    external_time_loop_used: bool
    conditional_states_precomputed: bool

def broken_constraint_state() -> np.ndarray:
    state = np.zeros(D_C * D_S, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)  # |0,0>
    state[D_S] = 1.0 / np.sqrt(2.0)  # |1,0>
    return state

def product_no_history_state() -> np.ndarray:
    tau_zero = clock_phase_state(D_C, 0)
    psi_zero = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return np.kron(tau_zero, psi_zero).astype(np.complex128, copy=False)

def wrong_conditioning(global_density: np.ndarray) -> np.ndarray:
    """Deliberately wrong k-independent Section-12 control only."""
    reduced = partial_trace_clock(np.asarray(global_density, dtype=np.complex128), D_C, D_S)
    trace = np.trace(global_density)
    if abs(trace) == 0:
        raise ValueError("global density trace cannot be zero")
    return (reduced / trace).astype(np.complex128, copy=False)

def reject_hidden_external_time(provenance: ConstructionProvenance) -> None:
    if (
        provenance.origin is ConstructionOrigin.EXTERNAL_CLASSICAL_TIME_THEN_PACKAGED
        or provenance.external_time_loop_used
        or provenance.conditional_states_precomputed
    ):
        raise ValueError("hidden external-time construction invalidates the artifact")
    if provenance.origin is not ConstructionOrigin.DIRECT_CONSTRAINED_STATE:
        raise ValueError("unregistered construction origin")

def scramble_clock_labels(values: tuple[object, ...]) -> tuple[object, ...]:
    if len(values) != D_C:
        raise ValueError("clock-scramble control requires exactly 16 outcomes")
    return tuple(values[(5*k + 1) % D_C] for k in range(D_C))

def detect_post_hoc_mutation(candidate_binding: dict[str, object]) -> None:
    if candidate_binding != frozen_binding_snapshot():
        raise ValueError("post-hoc protocol/threshold/parameter mutation detected")

CONTROL_IMPLEMENTATIONS = MappingProxyType({
    ControlId.BROKEN_CONSTRAINT: broken_constraint_state,
    ControlId.PRODUCT_NO_HISTORY: product_no_history_state,
    ControlId.WRONG_CONDITIONING: wrong_conditioning,
    ControlId.HIDDEN_EXTERNAL_TIME: reject_hidden_external_time,
    ControlId.CLOCK_SCRAMBLE: scramble_clock_labels,
    ControlId.POST_HOC_THRESHOLD_MUTATION: detect_post_hoc_mutation,
})

if tuple(CONTROL_IMPLEMENTATIONS) != tuple(ControlId):
    raise RuntimeError("frozen control inventory/implementation mismatch")
