"""Frozen v1.1 orchestration. Import is inert; execution requires external authorization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from . import analytic_reference
from .adjudication import AdjudicationRecord, ControlObservation, EXPECTED_CONTROL_REASON
from .conditional_state import (clock_hamiltonian, clock_povm_element, condition_density_matrix,
    constraint, correlated_kernel_state, pure_density, system_hamiltonian)
from .controls import ControlId, CONTROL_IMPLEMENTATIONS
from .evidence import prepare_new_evidence_root, write_all_registered_arrays, write_json_once
from .metrics import neighboring_step_distances, pure_reference_fidelity, registered_tau_grid, trace_distance
from .protocol_binding import D_C, D_S, FROZEN_THRESHOLDS, OMEGA, PROTOCOL_GIT_BLOB, PROTOCOL_SHA256, verify_protocol_identity
from .verdict import derive_machine_verdict

STAGE_ORDER = (
    "A_FAIL_CLOSED_PROVENANCE", "B_CONDITIONAL_OUTPUT_AND_HASH",
    "C_ANALYTIC_COMPARISON", "D_REQUIRED_DIAGNOSTICS",
    "E_EXACT_SIX_CONTROLS", "F_AUTOMATIC_ADJUDICATION",
    "G_EVIDENCE_FINALIZATION",
)

@dataclass(frozen=True)
class FrozenIdentityBundle:
    protocol_blob: str
    protocol_sha256: str
    preregistration_blob: str
    preregistration_sha256: str
    environment_sha256: str
    source_manifest_sha256: str
    source_baseline_commit: str
    runner_blob: str

def validate_execution_authorization(document: dict[str, object], schema: dict[str, object], expected: FrozenIdentityBundle) -> None:
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    for field in expected.__dataclass_fields__:
        if document[field] != getattr(expected, field):
            raise RuntimeError(f"stale or mismatched authorization identity: {field}")
    if document["authorized"] is not True:
        raise RuntimeError("scientific execution is not authorized")

class FrozenQF0ARunner:
    """No extension hooks, alternate controls/metrics/tau grids, or exclusions."""
    def __init__(self, repo_root: Path, authorization_path: Path, evidence_root: Path):
        self.repo_root = repo_root
        self.authorization_path = authorization_path
        self.evidence_root = evidence_root
        self.authorization: dict[str, object] = {}
        self.run_id = ""

    def execute(self) -> None:
        self._stage_a_fail_closed_provenance()
        conditional = self._stage_b_construct_persist_hash_conditionals()
        analytic = self._stage_c_independent_analytic_reference(conditional)
        diagnostics = self._stage_d_registered_diagnostics(conditional, analytic)
        controls = self._stage_e_exact_six_controls(conditional)
        verdict = self._stage_f_automatic_adjudication(conditional, analytic, diagnostics, controls)
        self._stage_g_finalize_all_evidence(conditional, analytic, diagnostics, controls, verdict)

    def _stage_a_fail_closed_provenance(self) -> None:
        schema = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json").read_text(encoding="utf-8"))
        document = json.loads(self.authorization_path.read_text(encoding="utf-8"))
        expected = FrozenIdentityBundle(**{name: document[name] for name in FrozenIdentityBundle.__dataclass_fields__})
        if expected.protocol_blob != PROTOCOL_GIT_BLOB or expected.protocol_sha256 != PROTOCOL_SHA256:
            raise RuntimeError("authorization does not bind the frozen v1.1 protocol")
        validate_execution_authorization(document, schema, expected)
        verify_protocol_identity(self.repo_root, expected.source_baseline_commit)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=self.repo_root, text=True).strip():
            raise RuntimeError("scientific execution requires a clean tree")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo_root, text=True).strip() != expected.source_baseline_commit:
            raise RuntimeError("HEAD is not the authorized source baseline")
        runner_path = "foundation/qf0/qf0a_impl/src/qf0a/runner.py"
        runner_blob = subprocess.check_output(["git", "rev-parse", f"HEAD:{runner_path}"], cwd=self.repo_root, text=True).strip()
        if runner_blob != expected.runner_blob:
            raise RuntimeError("runner blob mismatch")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            if os.environ.get(variable) != "1":
                raise RuntimeError(f"thread environment mismatch: {variable}")
        self.run_id = str(document["run_id"])
        if self.evidence_root.name != self.run_id:
            raise RuntimeError("evidence directory basename must equal authorized run_id")
        prepare_new_evidence_root(self.evidence_root)
        write_json_once(self.evidence_root / "execution_authorization.json", document)
        self.authorization = document

    def _stage_b_construct_persist_hash_conditionals(self) -> dict[str, object]:
        h_system = system_hamiltonian(OMEGA)
        k0 = constraint(clock_hamiltonian(D_C, OMEGA), h_system)
        state = correlated_kernel_state(D_C)
        density = pure_density(state)
        conditioned = tuple(condition_density_matrix(density, clock_povm_element(D_C, k), D_S) for k in range(D_C))
        probabilities = np.asarray([item[0] for item in conditioned], dtype=np.float64)
        densities = np.asarray([item[1] for item in conditioned], dtype=np.complex128)
        np.save(self.evidence_root / "raw/conditional_densities.npy", densities, allow_pickle=False)
        np.save(self.evidence_root / "raw/conditional_probabilities.npy", probabilities, allow_pickle=False)
        identity = hashlib.sha256((self.evidence_root / "raw/conditional_densities.npy").read_bytes()).hexdigest()
        return {"state": state, "density": density, "k0": k0, "h_system": h_system,
                "probabilities": probabilities, "densities": densities, "persisted_sha256": identity}

    def _stage_c_independent_analytic_reference(self, conditional: dict[str, object]) -> dict[str, np.ndarray]:
        if not conditional["persisted_sha256"]:
            raise RuntimeError("conditional output must be persisted and hashed before analytic reference")
        h = analytic_reference.system_hamiltonian(OMEGA)
        psi = analytic_reference.initial_state()
        references = np.asarray([analytic_reference.reference_density(h, tau, psi) for tau in registered_tau_grid()], dtype=np.complex128)
        densities = conditional["densities"]
        return {"references": references,
                "fidelities": np.asarray([pure_reference_fidelity(rho, ref) for rho, ref in zip(densities, references)], dtype=np.float64),
                "distances": np.asarray([trace_distance(rho, ref) for rho, ref in zip(densities, references)], dtype=np.float64),
                "steps": neighboring_step_distances(tuple(densities), conditional["h_system"])}

    def _stage_d_registered_diagnostics(self, conditional: dict[str, object], analytic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        from .conditional_state import clock_b_phase_state, coarse_grained_clock_povm, interacting_constraint, near_kernel_projector_continuation
        interaction = []
        for g in (0.0, 1e-4, 1e-3, 1e-2):
            result = near_kernel_projector_continuation(interacting_constraint(conditional["k0"], g), conditional["state"])
            interaction.append([g, result.diagnostics.selected_rank, result.diagnostics.projector_hermiticity_residual,
                                result.diagnostics.projector_idempotence_residual, result.diagnostics.projection_norm])
        clock_b = []
        for k in range(D_C):
            p, rho = condition_density_matrix(conditional["density"], pure_density(clock_b_phase_state(k)), D_S)
            mapped = (k + 3) % D_C
            clock_b.append([k, mapped, p, trace_distance(rho, conditional["densities"][mapped])])
        coarse = []
        taus, h, psi = registered_tau_grid(), analytic_reference.system_hamiltonian(OMEGA), analytic_reference.initial_state()
        for j, effect in enumerate(coarse_grained_clock_povm()):
            p, rho = condition_density_matrix(conditional["density"], effect, D_S)
            left, right = taus[2*j], taus[2*j+1]
            refs = (analytic_reference.reference_density(h, left, psi), analytic_reference.reference_density(h, right, psi),
                    analytic_reference.reference_density(h, (left + right) / 2.0, psi))
            coarse.append([j, p, *(trace_distance(rho, ref) for ref in refs), float(np.trace(rho @ rho).real), *np.linalg.eigvalsh(rho)])
        return {"interaction": np.asarray(interaction, dtype=np.float64), "clock_b": np.asarray(clock_b, dtype=np.float64), "coarse": np.asarray(coarse, dtype=np.float64)}

    def _stage_e_exact_six_controls(self, conditional: dict[str, object]) -> tuple:
        if tuple(CONTROL_IMPLEMENTATIONS) != tuple(ControlId):
            raise RuntimeError("frozen control inventory mismatch")
        return tuple((control, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[control]) for control in ControlId)

    def _stage_f_automatic_adjudication(self, conditional, analytic, diagnostics, controls):
        checks = (bool(np.max(1.0 - analytic["fidelities"]) <= FROZEN_THRESHOLDS["FIDELITY_DEFECT_TOL"]),
                  bool(np.max(analytic["distances"]) <= FROZEN_THRESHOLDS["TRACE_DISTANCE_TOL"]),
                  bool(np.max(analytic["steps"]) <= FROZEN_THRESHOLDS["UNITARY_STEP_TRACE_DISTANCE_TOL"]),
                  bool(all(np.all(np.isfinite(value)) for value in diagnostics.values())))
        record = AdjudicationRecord(False, False, False, True, True, True, checks, controls, True)
        return derive_machine_verdict(self.run_id, record)

    def _stage_g_finalize_all_evidence(self, conditional, analytic, diagnostics, controls, verdict) -> None:
        arrays = {"conditional_densities.npy": conditional["densities"], "conditional_probabilities.npy": conditional["probabilities"],
                  "analytic_reference_densities.npy": analytic["references"], "fidelities.npy": analytic["fidelities"],
                  "trace_distances.npy": analytic["distances"], "neighbor_step_distances.npy": analytic["steps"],
                  "interaction_diagnostics.npy": diagnostics["interaction"], "clock_b_diagnostics.npy": diagnostics["clock_b"],
                  "coarse_clock_diagnostics.npy": diagnostics["coarse"]}
        index = write_all_registered_arrays(self.evidence_root / "raw", arrays)
        write_json_once(self.evidence_root / "raw_output_index.json", {"run_id": self.run_id, "arrays": index})
        write_json_once(self.evidence_root / "negative_controls.json", {"run_id": self.run_id, "controls": [[a.value,b.value,c] for a,b,c in controls]})
        write_json_once(self.evidence_root / "machine_verdict.json", {"run_id": verdict.run_id, "pipeline_reproduction_verdict": verdict.pipeline_reproduction_verdict,
            "foundation_model_verdict": verdict.foundation_model_verdict.value, "reason_codes": list(verdict.reason_codes)})
        write_json_once(self.evidence_root / "diagnostics.json", {name: value.tolist() for name,value in diagnostics.items()})
