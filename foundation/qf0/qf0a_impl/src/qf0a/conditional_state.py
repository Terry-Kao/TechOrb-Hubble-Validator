"""Direct constrained-state and conditioning primitives, with no external orbit path."""
from __future__ import annotations

import numpy as np

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

def condition_pure_state(global_state: np.ndarray, clock_state: np.ndarray, d_system: int) -> tuple[float, np.ndarray]:
    matrix = np.asarray(global_state, dtype=np.complex128).reshape(clock_state.size, d_system)
    conditioned = clock_state.conj() @ matrix
    probability = float(np.vdot(conditioned, conditioned).real)
    if probability <= 0.0:
        raise ValueError("conditioning probability must be positive")
    vector = conditioned / np.sqrt(probability)
    return probability, np.outer(vector, vector.conj()).astype(np.complex128, copy=False)

def near_kernel_projector_continuation(operator: np.ndarray, state: np.ndarray, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-subspace Option-B continuation; caller supplies the operator and window."""
    matrix = np.asarray(operator, dtype=np.complex128)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("operator is not Hermitian")
    values, vectors = np.linalg.eigh(matrix)
    mask = (values > window[0]) & (values < window[1])
    selected = vectors[:, mask]
    projector = selected @ selected.conj().T
    projected = projector @ np.asarray(state, dtype=np.complex128)
    norm = np.linalg.norm(projected)
    if norm < 1e-12:
        raise ValueError("projected norm below frozen minimum")
    continued = projected / norm
    overlap = np.vdot(state, continued)
    if abs(overlap) > 0:
        continued *= np.exp(-1j * np.angle(overlap))
    return values, projector, continued
