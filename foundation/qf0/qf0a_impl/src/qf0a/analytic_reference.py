"""Independent analytic-reference primitives; never imports conditional_state."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

def system_hamiltonian(omega: float) -> np.ndarray:
    return np.asarray([[0.0, 0.0], [0.0, omega]], dtype=np.complex128)

def initial_state() -> np.ndarray:
    return np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

def reference_density(hamiltonian: np.ndarray, tau: float, state: np.ndarray) -> np.ndarray:
    evolved = expm(-1j * np.asarray(hamiltonian, dtype=np.complex128) * tau) @ state
    return np.outer(evolved, evolved.conj()).astype(np.complex128, copy=False)
