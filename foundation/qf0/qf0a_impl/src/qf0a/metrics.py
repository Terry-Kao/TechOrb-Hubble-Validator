"""Frozen QF-0A v1.1 metrics. No I/O and no result-dependent configuration."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .protocol_binding import D_C, OMEGA

def _tau_grid(d_clock: int, omega: float) -> np.ndarray:
    return (2.0 * np.pi * np.arange(d_clock, dtype=np.float64) / (d_clock * omega)).astype(np.float64)

def registered_tau_grid() -> np.ndarray:
    return _tau_grid(D_C, OMEGA)

def registered_delta_tau() -> float:
    return float(2.0 * np.pi / (D_C * OMEGA))

def trace_normalization_defect(rho: np.ndarray) -> float:
    return float(abs(np.trace(np.asarray(rho, dtype=np.complex128)) - 1.0))

def hermiticity_residual(rho: np.ndarray) -> float:
    matrix = np.asarray(rho, dtype=np.complex128)
    return float(np.linalg.norm(matrix - matrix.conj().T))

def minimum_eigenvalue(rho: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(np.asarray(rho, dtype=np.complex128)).min())

def probability_sum_defect(probabilities: np.ndarray) -> float:
    return float(abs(np.asarray(probabilities, dtype=np.float64).sum() - 1.0))

def povm_completeness_residual(effects: tuple[np.ndarray, ...]) -> float:
    total = np.sum(np.asarray(effects, dtype=np.complex128), axis=0)
    return float(np.linalg.norm(total - np.eye(total.shape[0], dtype=np.complex128)))

def constraint_residual(operator: np.ndarray, state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(operator, dtype=np.complex128) @ np.asarray(state, dtype=np.complex128)))

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    delta = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    if not np.allclose(delta, delta.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("trace distance requires a Hermitian density difference")
    return 0.5 * float(np.abs(np.linalg.eigvalsh(delta)).sum())

def pure_reference_fidelity(rho: np.ndarray, pure_reference_density: np.ndarray) -> float:
    value = np.trace(np.asarray(rho, dtype=np.complex128) @ np.asarray(pure_reference_density, dtype=np.complex128))
    if abs(value.imag) > 1e-12:
        raise ValueError("pure-reference squared fidelity has non-negligible imaginary part")
    return float(value.real)

def registered_step_unitary(system_hamiltonian: np.ndarray) -> np.ndarray:
    return expm(-1j * np.asarray(system_hamiltonian, dtype=np.complex128) * registered_delta_tau()).astype(np.complex128)

def neighboring_step_distances(densities: tuple[np.ndarray, ...], system_hamiltonian: np.ndarray) -> np.ndarray:
    if len(densities) != D_C:
        raise ValueError("adjudicative neighboring-step path requires exactly d_C registered densities")
    unitary = registered_step_unitary(system_hamiltonian)
    values = []
    for k in range(D_C):
        predicted = unitary @ densities[k] @ unitary.conj().T
        values.append(trace_distance(densities[(k + 1) % D_C], predicted))
    return np.asarray(values, dtype=np.float64)

def unitary_norm_defect(unitary: np.ndarray) -> float:
    matrix = np.asarray(unitary, dtype=np.complex128)
    return float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0], dtype=np.complex128)))

def purity(rho: np.ndarray) -> float:
    matrix = np.asarray(rho, dtype=np.complex128)
    return float(np.trace(matrix @ matrix).real)
