"""Direct constrained-state and conditioning primitives, with no external orbit path."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .protocol_binding import (
    CLOCK_B_SHIFT, D_C, D_S, FROZEN_THRESHOLDS, INTERACTION_G,
    SPECTRAL_WINDOW,
)

@dataclass(frozen=True)
class ProjectorDiagnostics:
    eigenvalues: np.ndarray
    selected_rank: int
    projector: np.ndarray
    projector_hermiticity_residual: float
    projector_idempotence_residual: float
    projection_norm: float
    overlap: complex

@dataclass(frozen=True)
class ContinuationResult:
    state: np.ndarray
    diagnostics: ProjectorDiagnostics

def clock_hamiltonian(d_clock: int, omega: float) -> np.ndarray:
    return np.diag(np.arange(d_clock, dtype=np.float64) * omega).astype(np.complex128)

def system_hamiltonian(omega: float) -> np.ndarray:
    return np.diag(np.asarray([0.0, omega], dtype=np.float64)).astype(np.complex128)

def constraint(h_clock: np.ndarray, h_system: np.ndarray) -> np.ndarray:
    return np.kron(h_clock, np.eye(h_system.shape[0])) - np.kron(np.eye(h_clock.shape[0]), h_system)

def correlated_kernel_state(d_clock: int) -> np.ndarray:
    """Construct the correlated energy state directly; no trajectory is accepted."""
    state = np.zeros(d_clock * 2, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2.0)
    state[3] = 1.0 / np.sqrt(2.0)
    return state

def clock_phase_state(d_clock: int, outcome: int) -> np.ndarray:
    n = np.arange(d_clock, dtype=np.float64)
    return np.exp(1j * 2.0 * np.pi * n * outcome / d_clock).astype(np.complex128) / np.sqrt(d_clock)

def clock_povm_element(d_clock: int, outcome: int) -> np.ndarray:
    state = clock_phase_state(d_clock, outcome)
    return np.outer(state, state.conj()).astype(np.complex128, copy=False)

def pure_density(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128)
    return np.outer(vector, vector.conj()).astype(np.complex128, copy=False)

def partial_trace_clock(operator: np.ndarray, d_clock: int, d_system: int) -> np.ndarray:
    matrix = np.asarray(operator, dtype=np.complex128).reshape(d_clock, d_system, d_clock, d_system)
    return np.trace(matrix, axis1=0, axis2=2).astype(np.complex128, copy=False)

def condition_density_matrix(global_density: np.ndarray, clock_effect: np.ndarray, d_system: int) -> tuple[float, np.ndarray]:
    d_clock = clock_effect.shape[0]
    weighted = np.kron(np.asarray(clock_effect, dtype=np.complex128), np.eye(d_system, dtype=np.complex128)) @ np.asarray(global_density, dtype=np.complex128)
    numerator = partial_trace_clock(weighted, d_clock, d_system)
    probability = float(np.trace(numerator).real)
    if probability <= 0.0:
        raise ValueError("conditioning probability must be positive")
    return probability, (numerator / probability).astype(np.complex128, copy=False)

def condition_pure_state(global_state: np.ndarray, clock_state: np.ndarray, d_system: int) -> tuple[float, np.ndarray]:
    matrix = np.asarray(global_state, dtype=np.complex128).reshape(clock_state.size, d_system)
    conditioned = clock_state.conj() @ matrix
    probability = float(np.vdot(conditioned, conditioned).real)
    if probability <= 0.0:
        raise ValueError("conditioning probability must be positive")
    vector = conditioned / np.sqrt(probability)
    return probability, np.outer(vector, vector.conj()).astype(np.complex128, copy=False)

def near_kernel_projector_continuation(operator: np.ndarray, state: np.ndarray) -> ContinuationResult:
    """Frozen QF-0A Option-B continuation; the spectral window is not configurable."""
    matrix = np.asarray(operator, dtype=np.complex128)
    if not np.allclose(matrix, matrix.conj().T, atol=FROZEN_THRESHOLDS["HERMITICITY_TOL"], rtol=0.0):
        raise ValueError("operator is not Hermitian")
    values, vectors = np.linalg.eigh(matrix)
    if np.any(np.minimum(np.abs(values - SPECTRAL_WINDOW[0]), np.abs(values - SPECTRAL_WINDOW[1])) <= FROZEN_THRESHOLDS["WINDOW_BOUNDARY_TOL"]):
        raise ValueError("eigenvalue at/near frozen spectral boundary; protocol version invalidated")
    mask = (values > SPECTRAL_WINDOW[0]) & (values < SPECTRAL_WINDOW[1])
    if int(mask.sum()) != 2:
        raise ValueError("frozen near-kernel selected rank must equal 2")
    selected = vectors[:, mask]
    projector = selected @ selected.conj().T
    hermiticity_residual = float(np.linalg.norm(projector - projector.conj().T))
    idempotence_residual = float(np.linalg.norm(projector @ projector - projector))
    if hermiticity_residual > FROZEN_THRESHOLDS["PROJECTOR_HERMITICITY_TOL"]:
        raise ValueError("projector Hermiticity residual exceeds frozen tolerance")
    if idempotence_residual > FROZEN_THRESHOLDS["PROJECTOR_IDEMPOTENCE_TOL"]:
        raise ValueError("projector idempotence residual exceeds frozen tolerance")
    projected = projector @ np.asarray(state, dtype=np.complex128)
    norm = float(np.linalg.norm(projected))
    if norm < FROZEN_THRESHOLDS["PROJECTION_NORM_MIN"]:
        raise ValueError("projected norm below frozen minimum")
    continued = projected / norm
    overlap = np.vdot(state, continued)
    if abs(overlap) > 0:
        continued *= np.exp(-1j * np.angle(overlap))
    overlap = np.vdot(state, continued)
    diagnostics = ProjectorDiagnostics(values, 2, projector, hermiticity_residual, idempotence_residual, norm, complex(overlap))
    return ContinuationResult(continued.astype(np.complex128, copy=False), diagnostics)

def validate_g0_recovery(reference_density: np.ndarray, continued_density: np.ndarray) -> None:
    delta = np.asarray(reference_density, dtype=np.complex128) - np.asarray(continued_density, dtype=np.complex128)
    trace_distance = 0.5 * float(np.linalg.svd(delta, compute_uv=False).sum())
    if trace_distance > FROZEN_THRESHOLDS["G0_RECOVERY_TRACE_DISTANCE_TOL"]:
        raise ValueError("g=0 recovery exceeds frozen trace-distance tolerance")

def interaction_operator() -> np.ndarray:
    clock_level_one = np.zeros((D_C, D_C), dtype=np.complex128)
    clock_level_one[1, 1] = 1.0
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return np.kron(clock_level_one, pauli_x).astype(np.complex128, copy=False)

def interacting_constraint(k0: np.ndarray, g: float) -> np.ndarray:
    if g not in INTERACTION_G:
        raise ValueError("g is outside the frozen interaction inventory")
    expected_shape = (D_C * D_S, D_C * D_S)
    if np.asarray(k0).shape != expected_shape:
        raise ValueError("K0 shape is incompatible with frozen QF-0A dimensions")
    return np.asarray(k0, dtype=np.complex128) + g * interaction_operator()

def clock_b_phase_state(outcome: int) -> np.ndarray:
    mapped = (outcome + CLOCK_B_SHIFT) % D_C
    phase = np.exp(1j * 2.0 * np.pi * outcome / D_C)
    return (phase * clock_phase_state(D_C, mapped)).astype(np.complex128, copy=False)

def clock_b_registered_map(outcome: int) -> int:
    return (outcome + CLOCK_B_SHIFT) % D_C

def _coarse_povm_for_dimension(d_clock: int) -> tuple[np.ndarray, ...]:
    if d_clock <= 0 or d_clock % 2:
        raise ValueError("coarse clock dimension must be positive and even")
    return tuple(clock_povm_element(d_clock, 2*j) + clock_povm_element(d_clock, 2*j+1) for j in range(d_clock // 2))

def coarse_grained_clock_povm() -> tuple[np.ndarray, ...]:
    """Adjacent-outcome information-loss diagnostic; not Clock-B equivalence."""
    return _coarse_povm_for_dimension(D_C)
