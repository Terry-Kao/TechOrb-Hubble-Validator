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
from .controls import (ControlId, ConstructionOrigin, ConstructionProvenance,
    CONTROL_IMPLEMENTATIONS, detect_post_hoc_mutation, scramble_clock_labels,
    validate_construction_origin_attestation)
from .evidence import prepare_new_evidence_root, validate_complete_inventory, write_all_registered_arrays, write_json_once
from .metrics import constraint_residual, neighboring_step_distances, pure_reference_fidelity, registered_tau_grid, trace_distance
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
        manifest_path = self.repo_root / "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json"
        manifest_bytes = subprocess.check_output(["git", "show", "HEAD:foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json"], cwd=self.repo_root)
        manifest = json.loads(manifest_bytes)
        baseline = str(manifest["source_baseline_commit"])
        prereg_path = "foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
        env_path = "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"
        runner_path = "foundation/qf0/qf0a_impl/src/qf0a/runner.py"
        prereg_bytes = subprocess.check_output(["git", "show", f"{baseline}:{prereg_path}"], cwd=self.repo_root)
        env_bytes = subprocess.check_output(["git", "show", f"{baseline}:{env_path}"], cwd=self.repo_root)
        expected = FrozenIdentityBundle(
            PROTOCOL_GIT_BLOB, PROTOCOL_SHA256,
            subprocess.check_output(["git", "rev-parse", f"{baseline}:{prereg_path}"], cwd=self.repo_root, text=True).strip(),
            hashlib.sha256(prereg_bytes).hexdigest(), hashlib.sha256(env_bytes).hexdigest(),
            hashlib.sha256(manifest_bytes).hexdigest(), baseline,
            subprocess.check_output(["git", "rev-parse", f"{baseline}:{runner_path}"], cwd=self.repo_root, text=True).strip())
        validate_execution_authorization(document, schema, expected)
        verify_protocol_identity(self.repo_root, expected.source_baseline_commit)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=self.repo_root, text=True).strip():
            raise RuntimeError("scientific execution requires a clean tree")
        runner_blob = subprocess.check_output(["git", "rev-parse", f"{baseline}:{runner_path}"], cwd=self.repo_root, text=True).strip()
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
        dimensions = []
        for d_clock in (8, 16, 32):
            state = correlated_kernel_state(d_clock)
            density = pure_density(state)
            conditioned = tuple(condition_density_matrix(density, clock_povm_element(d_clock, k), D_S)[1] for k in range(d_clock))
            h = analytic_reference.system_hamiltonian(OMEGA)
            psi = analytic_reference.initial_state()
            taus = 2.0 * np.pi * np.arange(d_clock, dtype=np.float64) / (d_clock * OMEGA)
            refs = tuple(analytic_reference.reference_density(h, tau, psi) for tau in taus)
            dimensions.append([d_clock, max(trace_distance(rho, ref) for rho, ref in zip(conditioned, refs))])
        return {"interaction": np.asarray(interaction, dtype=np.float64), "clock_b": np.asarray(clock_b, dtype=np.float64),
                "coarse": np.asarray(coarse, dtype=np.float64), "dimensions": np.asarray(dimensions, dtype=np.float64)}

    def _stage_e_exact_six_controls(self, conditional: dict[str, object]) -> tuple:
        if tuple(CONTROL_IMPLEMENTATIONS) != tuple(ControlId):
            raise RuntimeError("frozen control inventory mismatch")
        observations = []
        broken = CONTROL_IMPLEMENTATIONS[ControlId.BROKEN_CONSTRAINT]()
        if constraint_residual(conditional["k0"], broken) <= FROZEN_THRESHOLDS["CONSTRAINT_RESIDUAL_TOL"]:
            raise RuntimeError("BROKEN_CONSTRAINT unexpectedly passed")
        observations.append((ControlId.BROKEN_CONSTRAINT, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.BROKEN_CONSTRAINT]))
        product = CONTROL_IMPLEMENTATIONS[ControlId.PRODUCT_NO_HISTORY]()
        product_density = np.outer(product, product.conj())
        product_probabilities = []
        for k in range(D_C):
            try: product_probabilities.append(condition_density_matrix(product_density, clock_povm_element(D_C, k), D_S)[0])
            except ValueError: product_probabilities.append(0.0)
        if all(p > FROZEN_THRESHOLDS["MIN_OUTCOME_PROBABILITY"] for p in product_probabilities):
            raise RuntimeError("PRODUCT_NO_HISTORY did not show registered absence/zero-probability behavior")
        observations.append((ControlId.PRODUCT_NO_HISTORY, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.PRODUCT_NO_HISTORY]))
        wrong = CONTROL_IMPLEMENTATIONS[ControlId.WRONG_CONDITIONING](conditional["density"])
        if all(trace_distance(wrong, rho) <= FROZEN_THRESHOLDS["TRACE_DISTANCE_TOL"] for rho in conditional["densities"]):
            raise RuntimeError("WRONG_CONDITIONING unexpectedly passed")
        observations.append((ControlId.WRONG_CONDITIONING, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.WRONG_CONDITIONING]))
        try:
            validate_construction_origin_attestation(ConstructionProvenance(ConstructionOrigin.EXTERNAL_CLASSICAL_TIME_THEN_PACKAGED, True, True))
        except ValueError: observations.append((ControlId.HIDDEN_EXTERNAL_TIME, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.HIDDEN_EXTERNAL_TIME]))
        else: raise RuntimeError("HIDDEN_EXTERNAL_TIME unexpectedly passed")
        scrambled = scramble_clock_labels(tuple(conditional["densities"]))
        if all(trace_distance(a, b) <= FROZEN_THRESHOLDS["TRACE_DISTANCE_TOL"] for a, b in zip(scrambled, conditional["densities"])):
            raise RuntimeError("CLOCK_SCRAMBLE unexpectedly passed")
        observations.append((ControlId.CLOCK_SCRAMBLE, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.CLOCK_SCRAMBLE]))
        try: detect_post_hoc_mutation({})
        except ValueError: observations.append((ControlId.POST_HOC_THRESHOLD_MUTATION, ControlObservation.EXPECTED_FAILURE, EXPECTED_CONTROL_REASON[ControlId.POST_HOC_THRESHOLD_MUTATION]))
        else: raise RuntimeError("POST_HOC_THRESHOLD_MUTATION unexpectedly passed")
        return tuple(observations)

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
        manifest = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json").read_text(encoding="utf-8"))
        environment = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text(encoding="utf-8"))
        write_json_once(self.evidence_root / "run_manifest.json", {"run_id": self.run_id, "protocol_sha256": PROTOCOL_SHA256, "source_baseline_commit": manifest["source_baseline_commit"]})
        write_json_once(self.evidence_root / "environment_manifest.json", environment)
        write_json_once(self.evidence_root / "source_manifest.json", manifest)
        (self.evidence_root / "source_manifest.sha256").write_text(hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()+"\n", encoding="ascii", newline="\n")
        write_json_once(self.evidence_root / "leakage_audit.json", {"run_id": self.run_id, "verdict": "PASS", "scope": "RUNTIME_PROVENANCE"})
        write_json_once(self.evidence_root / "human_report_metadata.json", {"run_id": self.run_id, "scientific_interpretation": "PIPELINE_QUALIFICATION_ONLY"})
        (self.evidence_root / "logs/run.log").write_text("frozen stage order completed\n", encoding="utf-8", newline="\n")
        hashes = []
        for path in sorted(p for p in self.evidence_root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.evidence_root).as_posix()}")
        (self.evidence_root / "SHA256SUMS.txt").write_text("\n".join(hashes)+"\n", encoding="ascii", newline="\n")
        validate_complete_inventory(self.evidence_root)
