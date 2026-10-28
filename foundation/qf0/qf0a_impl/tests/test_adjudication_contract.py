"""Synthetic-only tests for v1.1 metrics, controls, authorization, and precedence."""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "foundation/qf0/qf0a_impl/src"))

from qf0a.adjudication import AdjudicationRecord, ControlObservation, DerivedPipelineVerdict, EXPECTED_CONTROL_REASON, ThresholdResult, derive_pipeline_verdict
from qf0a.controls import ControlId
from qf0a.protocol_binding import FROZEN_THRESHOLDS
from qf0a.metrics import _tau_grid, hermiticity_residual, minimum_eigenvalue, pure_reference_fidelity, trace_distance
from qf0a.runner import FrozenIdentityBundle, STAGE_ORDER, validate_execution_authorization
from qf0a.verdict import derive_machine_verdict

def controls(observation=ControlObservation.EXPECTED_FAILURE):
    return tuple((control, observation, EXPECTED_CONTROL_REASON[control]) for control in ControlId)

def record(**changes):
    results = tuple(ThresholdResult(name, 0.0, "synthetic", limit, True, "SYNTHETIC", f"{name}_FAILED") for name, limit in FROZEN_THRESHOLDS.items())
    data = dict(provenance_contaminated=False, post_hoc_mutation=False, identity_mismatch=False,
                required_capability_complete=True, required_evidence_complete=True,
                required_measurements_complete=True, threshold_results=results,
                control_observations=controls(), leakage_audit_pass=True)
    data.update(changes)
    return AdjudicationRecord(**data)

class AdjudicationContractTests(unittest.TestCase):
    def test_artificial_metrics(self):
        rho = np.diag([0.75, 0.25]).astype(np.complex128)
        sigma = np.diag([0.5, 0.5]).astype(np.complex128)
        self.assertAlmostEqual(trace_distance(rho, sigma), 0.25)
        self.assertAlmostEqual(pure_reference_fidelity(rho, np.diag([1,0])), 0.75)
        self.assertEqual(hermiticity_residual(rho), 0.0)
        self.assertAlmostEqual(minimum_eigenvalue(rho), 0.25)
        self.assertTrue(np.allclose(_tau_grid(4, 2.0), [0, np.pi/4, np.pi/2, 3*np.pi/4]))

    def test_precedence(self):
        self.assertEqual(derive_pipeline_verdict(record(provenance_contaminated=True)).verdict, DerivedPipelineVerdict.INVALIDATED)
        self.assertEqual(derive_pipeline_verdict(record(required_evidence_complete=False)).verdict, DerivedPipelineVerdict.BLOCKED)
        failed = list(record().threshold_results)
        failed[0] = ThresholdResult(failed[0].name, 1.0, "synthetic", failed[0].limit, False, "SYNTHETIC", "TRACE_NORMALIZATION_TOL_FAILED")
        self.assertEqual(derive_pipeline_verdict(record(threshold_results=tuple(failed))).verdict, DerivedPipelineVerdict.FAIL)
        self.assertEqual(derive_pipeline_verdict(record()).verdict, DerivedPipelineVerdict.PASS)

    def test_product_zero_probability_is_expected_failure_not_blocked(self):
        product = tuple((c, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[c]) for c in ControlId)
        decision = derive_pipeline_verdict(record(control_observations=product))
        self.assertEqual(decision.verdict, DerivedPipelineVerdict.PASS)

    def test_pass_cannot_be_caller_supplied(self):
        self.assertNotIn("pipeline", inspect.signature(derive_machine_verdict).parameters)
        self.assertNotIn("verdict", inspect.signature(derive_machine_verdict).parameters)

    def test_runner_stage_order(self):
        self.assertEqual(STAGE_ORDER[1:3], ("B_CONDITIONAL_OUTPUT_AND_HASH", "C_ANALYTIC_COMPARISON"))
        self.assertEqual(STAGE_ORDER[-3:], ("G2_EVIDENCE_VALIDATION", "F2_FINAL_MACHINE_VERDICT", "G3_FINAL_SEALING"))

    def test_all_named_thresholds_required_and_unique(self):
        self.assertEqual({item.name for item in record().threshold_results}, set(FROZEN_THRESHOLDS))
        partial = record().threshold_results[:-1]
        self.assertEqual(derive_pipeline_verdict(record(threshold_results=partial)).verdict, DerivedPipelineVerdict.BLOCKED)
        duplicate = record().threshold_results[:-1] + (record().threshold_results[0],)
        self.assertEqual(derive_pipeline_verdict(record(threshold_results=duplicate)).verdict, DerivedPipelineVerdict.BLOCKED)

    def test_runner_wires_every_frozen_threshold_by_name(self):
        source = inspect.getsource(__import__("qf0a.runner", fromlist=["FrozenQF0ARunner"]).FrozenQF0ARunner._stage_f1_provisional_gate_evaluation)
        for name in FROZEN_THRESHOLDS:
            self.assertEqual(source.count(f'"{name}"'), 1, name)

    def test_synthetic_authorization(self):
        schema = json.loads((ROOT / "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json").read_text())
        expected = FrozenIdentityBundle("3af303a471d99397429d33f77ca512578f40f972","1cf9f7f3a8cfa6a5b0e12a3290d7e545f44586f9359aaa5367754415cdcc5f3a","a"*40,"b"*64,"c"*64,"d"*64,"e"*40,"f"*40)
        document = {"authorization_version":"1.1","run_id":"synthetic-auth","authorized":True,"protocol_blob":expected.protocol_blob,"protocol_sha256":expected.protocol_sha256,"preregistration_blob":expected.preregistration_blob,"preregistration_sha256":expected.preregistration_sha256,"environment_sha256":expected.environment_sha256,"source_manifest_sha256":expected.source_manifest_sha256,"source_baseline_commit":expected.source_baseline_commit,"runner_blob":expected.runner_blob,"issuer":"synthetic-test","governance_record":"synthetic-only","timestamp_utc":"2026-10-28T00:00:00Z"}
        validate_execution_authorization(document, schema, expected)
        document["source_manifest_sha256"] = "0"*64
        with self.assertRaises(RuntimeError): validate_execution_authorization(document, schema, expected)

if __name__ == "__main__": unittest.main()
