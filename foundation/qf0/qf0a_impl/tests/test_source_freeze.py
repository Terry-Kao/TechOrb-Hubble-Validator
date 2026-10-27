"""Artificial and static tests only; frozen QF-0A suites are never instantiated."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "foundation" / "qf0" / "qf0a_impl" / "src"
sys.path.insert(0, str(SRC))

from qf0a import SCIENTIFIC_EXECUTION_AUTHORIZED
from qf0a.audit import validate_static_separation
from qf0a.conditional_state import condition_pure_state, near_kernel_projector_continuation
from qf0a.controls import CLOCK_SCRAMBLE, CONTROL_DEFINITIONS
from qf0a.protocol_binding import verify_protocol_identity
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
        operator = np.diag(np.asarray([-2.0, 0.1, 0.2, 3.0], dtype=np.float64))
        state = np.asarray([0, 1, 1j, 0], dtype=np.complex128) / np.sqrt(2)
        _, projector, continued = near_kernel_projector_continuation(operator, state, (-0.25, 0.25))
        self.assertTrue(np.allclose(projector @ projector, projector))
        self.assertAlmostEqual(np.linalg.norm(continued), 1.0)

    def test_verdict_cannot_promote_foundation(self):
        record = construct_verdict("synthetic-run", PipelineVerdict.PASS, FoundationVerdict.NOT_ESTABLISHED, ("SYNTHETIC",))
        self.assertEqual(record.foundation_model_verdict, FoundationVerdict.NOT_ESTABLISHED)

if __name__ == "__main__":
    unittest.main()
