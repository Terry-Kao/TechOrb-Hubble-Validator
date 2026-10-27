"""Artificial and static tests only; frozen QF-0A suites are never instantiated."""
from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "foundation" / "qf0" / "qf0a_impl" / "src"
sys.path.insert(0, str(SRC))

from qf0a import SCIENTIFIC_EXECUTION_AUTHORIZED
from qf0a.audit import validate_static_separation
from qf0a.conditional_state import (
    _coarse_povm_for_dimension, clock_povm_element, condition_density_matrix,
    clock_b_phase_state, clock_b_registered_map, condition_pure_state,
    interaction_operator, near_kernel_projector_continuation,
    pure_density, validate_g0_recovery,
)
from qf0a.controls import (
    CLOCK_SCRAMBLE, CONTROL_DEFINITIONS, CONTROL_IMPLEMENTATIONS,
    ConstructionOrigin, ConstructionProvenance, broken_constraint_state,
    detect_post_hoc_mutation, product_no_history_state,
    reject_hidden_external_time, scramble_clock_labels, wrong_conditioning,
)
from qf0a.protocol_binding import (
    CLOCK_B_SHIFT, D_C, D_S, FROZEN_STRUCTURE, FROZEN_THRESHOLDS,
    INTERACTION_G, SPECTRAL_WINDOW, frozen_binding_snapshot,
    verify_protocol_identity,
)
from qf0a.verdict import FoundationVerdict, PipelineVerdict, construct_verdict

class SourceFreezeTests(unittest.TestCase):
    def test_execution_remains_locked(self):
        self.assertFalse(SCIENTIFIC_EXECUTION_AUTHORIZED)

    def test_protocol_binding(self):
        self.assertEqual(verify_protocol_identity(ROOT).version, "QF0A_FROZEN_PROTOCOL_v1.0")

    def test_static_separation(self):
        source = SRC / "qf0a"
        self.assertEqual(validate_static_separation(source), [])
        self.assertNotIn("analytic_reference", (source / "conditional_state.py").read_text())

    def test_six_control_definitions_only(self):
        self.assertEqual(len(CONTROL_DEFINITIONS), 6)
        self.assertEqual(len({item.control_id for item in CONTROL_DEFINITIONS}), 6)
        self.assertEqual(CLOCK_SCRAMBLE, (1,6,11,0,5,10,15,4,9,14,3,8,13,2,7,12))

    def test_artificial_partial_conditioning(self):
        # A deliberately unrelated 2x2 Bell-state primitive, not the frozen dimensions/suite.
        global_state = np.asarray([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
        probability, density = condition_pure_state(global_state, np.asarray([1, 0], dtype=np.complex128), 2)
        self.assertAlmostEqual(probability, 0.5)
        self.assertTrue(np.allclose(density, np.diag([1, 0])))

    def test_artificial_full_projector(self):
        operator = np.diag(np.asarray([-2.0, -0.1, 0.2, 3.0], dtype=np.float64))
        state = np.asarray([0, 1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        result = near_kernel_projector_continuation(operator, state)
        self.assertEqual(result.diagnostics.selected_rank, 2)
        self.assertTrue(np.allclose(result.diagnostics.projector @ result.diagnostics.projector, result.diagnostics.projector))
        self.assertAlmostEqual(np.linalg.norm(result.state), 1.0)
        self.assertGreaterEqual(result.diagnostics.overlap.real, 0.0)

    def test_window_cannot_be_overridden(self):
        self.assertEqual(tuple(inspect.signature(near_kernel_projector_continuation).parameters), ("operator", "state"))
        with self.assertRaises(TypeError):
            near_kernel_projector_continuation(np.eye(4), np.ones(4), (-0.25, 0.25))

    def test_artificial_boundary_and_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            near_kernel_projector_continuation(np.diag([-2.0, 0.0, 0.5, 3.0]), np.asarray([0,1,0,0]))
        with self.assertRaisesRegex(ValueError, "rank"):
            near_kernel_projector_continuation(np.diag([-2.0, 0.0, 2.0, 3.0]), np.asarray([0,1,0,0]))

    def test_artificial_g0_recovery_contract(self):
        density = np.diag(np.asarray([0.25, 0.75], dtype=np.complex128))
        validate_g0_recovery(density, density.copy())
        with self.assertRaisesRegex(ValueError, "g=0"):
            validate_g0_recovery(density, np.diag([0.5, 0.5]))

    def test_full_protocol_binding_inventory(self):
        expected = {
            "TRACE_NORMALIZATION_TOL": 1e-12, "HERMITICITY_TOL": 1e-12,
            "MIN_EIGENVALUE_TOL": -1e-12, "CONSTRAINT_RESIDUAL_TOL": 1e-12,
            "POVM_COMPLETENESS_TOL": 1e-12, "PROBABILITY_SUM_TOL": 1e-12,
            "MIN_OUTCOME_PROBABILITY": 1e-12, "CONDITIONAL_TRACE_TOL": 1e-12,
            "CONDITIONAL_MIN_EIGENVALUE_TOL": -1e-12, "FIDELITY_DEFECT_TOL": 1e-10,
            "TRACE_DISTANCE_TOL": 1e-10, "UNITARY_STEP_TRACE_DISTANCE_TOL": 1e-10,
            "UNITARY_NORM_DEFECT_TOL": 1e-12, "RESOLUTION_DETERIORATION_TOL": 1e-8,
            "WINDOW_BOUNDARY_TOL": 1e-10, "PROJECTOR_HERMITICITY_TOL": 1e-12,
            "PROJECTOR_IDEMPOTENCE_TOL": 1e-12, "PROJECTION_NORM_MIN": 1e-12,
            "G0_RECOVERY_TRACE_DISTANCE_TOL": 1e-12, "DETERMINISM_TRACE_DISTANCE_TOL": 1e-12,
            "CLOCK_MAPPED_TRACE_DISTANCE_TOL": 1e-10, "CLOCK_PROBABILITY_DIFFERENCE_TOL": 1e-12,
        }
        self.assertEqual(dict(FROZEN_THRESHOLDS), expected)
        self.assertEqual(SPECTRAL_WINDOW, (-0.5, 0.5))
        self.assertEqual(INTERACTION_G, (0.0, 1e-4, 1e-3, 1e-2))
        self.assertEqual((D_C, D_S, CLOCK_B_SHIFT), (16, 2, 3))
        self.assertEqual(set(FROZEN_STRUCTURE) >= {"HBAR", "RESOLUTION_D_C", "SEEDS", "CLOCK_SCRAMBLE"}, True)

    def test_interaction_operator_frozen_identity(self):
        interaction = interaction_operator()
        self.assertEqual(interaction.shape, (D_C * D_S, D_C * D_S))
        self.assertEqual(interaction.dtype, np.complex128)
        self.assertEqual(np.count_nonzero(interaction), 2)

    def test_artificial_coarse_povm_structure(self):
        coarse = _coarse_povm_for_dimension(4)
        self.assertEqual(len(coarse), 2)
        self.assertTrue(np.allclose(coarse[0], clock_povm_element(4, 0) + clock_povm_element(4, 1)))

    def test_clock_b_frozen_algebra(self):
        outcome = 2
        expected = np.exp(1j * 2 * np.pi * outcome / D_C) * __import__("qf0a.conditional_state", fromlist=["clock_phase_state"]).clock_phase_state(D_C, 5)
        self.assertEqual(clock_b_registered_map(outcome), 5)
        self.assertTrue(np.allclose(clock_b_phase_state(outcome), expected))

    def test_explicit_density_conditioning_artificial(self):
        vector = np.asarray([1,0,0,1], dtype=np.complex128) / np.sqrt(2)
        probability, density = condition_density_matrix(pure_density(vector), np.diag([1,0]).astype(np.complex128), 2)
        self.assertAlmostEqual(probability, 0.5)
        self.assertTrue(np.allclose(density, np.diag([1,0])))

    def test_exact_control_implementations(self):
        self.assertEqual(len(CONTROL_IMPLEMENTATIONS), 6)
        broken = broken_constraint_state()
        self.assertEqual(tuple(np.flatnonzero(broken)), (0, D_S))
        self.assertEqual(product_no_history_state().shape, (D_C * D_S,))

    def test_wrong_conditioning_is_k_independent(self):
        artificial = np.eye(D_C * D_S, dtype=np.complex128) / (D_C * D_S)
        self.assertTrue(np.allclose(wrong_conditioning(artificial), np.eye(D_S) / D_S))

    def test_hidden_time_detector(self):
        reject_hidden_external_time(ConstructionProvenance(ConstructionOrigin.DIRECT_CONSTRAINED_STATE, False, False))
        with self.assertRaisesRegex(ValueError, "external-time"):
            reject_hidden_external_time(ConstructionProvenance(ConstructionOrigin.EXTERNAL_CLASSICAL_TIME_THEN_PACKAGED, True, True))

    def test_forward_scramble_only(self):
        values = tuple(range(16))
        self.assertEqual(scramble_clock_labels(values), CLOCK_SCRAMBLE)
        self.assertFalse(hasattr(__import__("qf0a.controls", fromlist=["x"]), "inverse_scramble"))

    def test_post_hoc_mutation_detector(self):
        detect_post_hoc_mutation(frozen_binding_snapshot())
        changed = frozen_binding_snapshot()
        changed["thresholds"]["TRACE_DISTANCE_TOL"] = 0.1
        with self.assertRaisesRegex(ValueError, "mutation"):
            detect_post_hoc_mutation(changed)

    def test_verdict_cannot_promote_foundation(self):
        record = construct_verdict("synthetic-run", PipelineVerdict.PASS, FoundationVerdict.NOT_ESTABLISHED, ("SYNTHETIC",))
        self.assertEqual(record.foundation_model_verdict, FoundationVerdict.NOT_ESTABLISHED)

if __name__ == "__main__":
    unittest.main()
