"""Frozen v1.1 orchestration. Import is inert; execution requires external authorization."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from . import analytic_reference
from .adjudication import AdjudicationRecord, ControlObservation, EXPECTED_CONTROL_REASON, ThresholdResult
from .conditional_state import (clock_hamiltonian, clock_povm_element, condition_density_matrix,
    constraint, correlated_kernel_state, pure_density, system_hamiltonian)
from .controls import (ControlId, ConstructionOrigin, ConstructionProvenance,
    CONTROL_IMPLEMENTATIONS, detect_post_hoc_mutation, scramble_clock_labels,
    validate_construction_origin_attestation)
from .evidence import (prepare_new_evidence_root, validate_complete_inventory, write_array_once,
    write_remaining_registered_arrays, write_json_once, write_source_manifest_with_sha, write_text_once)
from .metrics import (constraint_residual, hermiticity_residual, minimum_eigenvalue,
    neighboring_step_distances, povm_completeness_residual, probability_sum_defect,
    pure_reference_fidelity, registered_step_unitary, registered_tau_grid,
    trace_distance, trace_normalization_defect, unitary_norm_defect)
from .protocol_binding import D_C, D_S, FROZEN_THRESHOLDS, OMEGA, PROTOCOL_GIT_BLOB, PROTOCOL_SHA256, verify_protocol_identity
from .verdict import derive_machine_verdict
from .provenance import ExecutionSourceSeal, verify_execution_head_against_source_manifest

STAGE_ORDER = (
    "A_FAIL_CLOSED_PROVENANCE", "B_CONDITIONAL_OUTPUT_AND_HASH",
    "C_ANALYTIC_COMPARISON", "D_REQUIRED_DIAGNOSTICS",
    "E_EXACT_SIX_CONTROLS", "F1_PROVISIONAL_GATE_EVALUATION",
    "G1_EVIDENCE_MATERIALIZATION", "G2_EVIDENCE_VALIDATION",
    "F2_FINAL_MACHINE_VERDICT", "G3_FINAL_SEALING",
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

def _execution_branch(repo_root: Path, source_seal: ExecutionSourceSeal) -> str:
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if current_head != source_seal.current_execution_head:
        raise RuntimeError("execution HEAD differs from validated source seal")
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root, text=True, capture_output=True, check=False,
    )
    if result.returncode == 0:
        branch = result.stdout.strip()
        if not branch:
            raise RuntimeError("attached execution HEAD has no branch name")
        return branch
    if result.returncode == 1:
        return "DETACHED"
    raise RuntimeError("could not determine execution branch context")

def build_run_manifest(repo_root: Path, run_id: str, manifest: dict[str, object], environment: dict[str, object], source_sha: str, source_seal: ExecutionSourceSeal) -> dict[str, object]:
    """Build provenance only; safe for synthetic schema tests and performs no science."""
    now = datetime.now(timezone.utc); local = now.astimezone(ZoneInfo("Asia/Taipei"))
    host = environment["host"]
    return {"schema_version":"1.1","run_id":run_id,
      "protocol":{"version":"QF0A_FROZEN_PROTOCOL_v1.1","path":"foundation/qf0/QF0A_FROZEN_PROTOCOL_v1.1.md","git_blob_sha":PROTOCOL_GIT_BLOB,"file_sha256":PROTOCOL_SHA256,"authoritative_main_sha":source_seal.current_execution_head},
      "implementation":{"commit_sha":source_seal.current_execution_head,"current_execution_head":source_seal.current_execution_head,"source_baseline_commit":source_seal.source_baseline_commit,"branch":_execution_branch(repo_root, source_seal),"dirty":False,"source_manifest_path":"source_manifest.json","source_manifest_sha256":source_sha},
      "environment":{"os":host["os"],"kernel":host["version"],"cpu_architecture":host["architecture"],"cpu_model":host["cpu"],"python":f"CPython {environment['python']['version']}","packages_lock_sha256":environment["dependency_lock_canonical_sha256"],"numpy_version":environment["packages"][0]["version"],"scipy_version":environment["packages"][1]["version"],"blas_lapack":{"vendor":"scipy-openblas","version_or_build":environment["blas_lapack"]["scipy"]},"locale":os.environ.get("LANG","UNSET"),"timezone":host["timezone"],"numeric_dtypes":environment["numeric_dtypes"]},
      "execution":{"timestamp_utc":now.isoformat(),"timestamp_asia_taipei":local.isoformat(),"python_executable":sys.executable,"command_line":sys.argv or ["qf0a"],"cwd":str(repo_root)},
      "determinism":{"seeds":{"primary":104729,"repeat_1":130363,"repeat_2":155921},"thread_environment":environment["threads"],"eigensolver_policy":"frozen-full-spectral-projector","tolerance_policy":"frozen-protocol-only"},
      "evidence":{"environment_manifest":"environment_manifest.json","raw_output_index":"raw_output_index.json","negative_controls":"negative_controls.json","leakage_audit":"leakage_audit.json","machine_verdict":"machine_verdict.json","human_report":"human_report.md","hash_manifest":"SHA256SUMS.txt"}}

def validate_execution_authorization(document: dict[str, object], schema: dict[str, object], expected: FrozenIdentityBundle) -> None:
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    for field in expected.__dataclass_fields__:
        if document[field] != getattr(expected, field):
            raise RuntimeError(f"stale or mismatched authorization identity: {field}")
    if document["authorized"] is not True:
        raise RuntimeError("scientific execution is not authorized")

class FrozenQF0ARunner:
    """No extension hooks, alternate controls/metrics/tau grids, or exclusions."""
    def __init__(self, repo_root: Path, authorization_path: Path, evidence_root: Path, provenance_seal: ExecutionSourceSeal):
        self.repo_root = repo_root
        self.authorization_path = authorization_path
        self.evidence_root = evidence_root
        self.authorization: dict[str, object] = {}
        self.run_id = ""
        self.provenance_seal = provenance_seal

    def execute(self) -> None:
        self._stage_a_fail_closed_provenance()
        conditional = self._stage_b_construct_persist_hash_conditionals()
        analytic = self._stage_c_independent_analytic_reference(conditional)
        diagnostics = self._stage_d_registered_diagnostics(conditional, analytic)
        controls = self._stage_e_exact_six_controls(conditional)
        provisional = self._stage_f1_provisional_gate_evaluation(conditional, analytic, diagnostics, controls)
        self._stage_g1_materialize_evidence(conditional, analytic, diagnostics, controls, provisional)
        complete = self._stage_g2_validate_evidence(conditional, provisional)
        verdict = derive_machine_verdict(self.run_id, complete)
        self._stage_g3_final_sealing(verdict)

    def _stage_a_fail_closed_provenance(self) -> None:
        schema = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/schemas/QF0A_EXECUTION_AUTHORIZATION.schema.json").read_text(encoding="utf-8"))
        document = json.loads(self.authorization_path.read_text(encoding="utf-8"))
        current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo_root, text=True).strip()
        if current_head != self.provenance_seal.current_execution_head:
            raise RuntimeError("execution HEAD changed after pre-import provenance closure")
        prereg_path = "foundation/qf0/QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md"
        env_path = "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json"
        runner_path = "foundation/qf0/qf0a_impl/src/qf0a/runner.py"
        expected = FrozenIdentityBundle(
            str(document["protocol_blob"]), str(document["protocol_sha256"]), str(document["preregistration_blob"]),
            str(document["preregistration_sha256"]), str(document["environment_sha256"]),
            str(document["source_manifest_sha256"]), str(document["source_baseline_commit"]), str(document["runner_blob"]))
        validate_execution_authorization(document, schema, expected)
        seal = verify_execution_head_against_source_manifest(self.repo_root, document)
        if seal != self.provenance_seal:
            raise RuntimeError("pre-import and Stage-A provenance seals differ")
        verify_protocol_identity(self.repo_root, current_head)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=self.repo_root, text=True).strip():
            raise RuntimeError("scientific execution requires a clean tree")
        runner_blob = subprocess.check_output(["git", "rev-parse", f"{current_head}:{runner_path}"], cwd=self.repo_root, text=True).strip()
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
        identities = {
            "conditional_densities.npy": write_array_once(self.evidence_root / "raw", "conditional_densities.npy", densities),
            "conditional_probabilities.npy": write_array_once(self.evidence_root / "raw", "conditional_probabilities.npy", probabilities),
        }
        return {"state": state, "density": density, "k0": k0, "h_system": h_system,
                "probabilities": probabilities, "densities": densities, "stage_b_identities": identities}

    def _stage_c_independent_analytic_reference(self, conditional: dict[str, object]) -> dict[str, np.ndarray]:
        if set(conditional["stage_b_identities"]) != {"conditional_densities.npy", "conditional_probabilities.npy"}:
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
            eig = result.diagnostics.eigenvalues
            continued = pure_density(result.state)
            repeated = near_kernel_projector_continuation(interacting_constraint(conditional["k0"], g), conditional["state"])
            interaction.append([g, int(np.count_nonzero(eig == 0.0)), int(np.count_nonzero(np.abs(eig) <= 1e-12)),
                result.diagnostics.selected_rank, float(np.min(np.minimum(np.abs(eig + .5), np.abs(eig - .5)))),
                result.diagnostics.projector_hermiticity_residual, result.diagnostics.projector_idempotence_residual,
                result.diagnostics.projection_norm, constraint_residual(interacting_constraint(conditional["k0"], g), result.state),
                float(result.diagnostics.overlap.real), float(result.diagnostics.overlap.imag),
                trace_distance(continued, conditional["density"]), float(np.min(np.abs(eig))),
                trace_distance(continued, pure_density(repeated.state))])
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

    @staticmethod
    def _threshold(name, observed, relation, passed, scope):
        return ThresholdResult(name, float(observed) if observed is not None else None, relation,
            FROZEN_THRESHOLDS[name], bool(passed), scope, f"{name}_FAILED")

    def _stage_f1_provisional_gate_evaluation(self, conditional, analytic, diagnostics, controls):
        effects = tuple(clock_povm_element(D_C, k) for k in range(D_C))
        rhos = tuple(conditional["densities"])
        interaction, clock_b, dimensions = diagnostics["interaction"], diagnostics["clock_b"], diagnostics["dimensions"]
        unitary_defect = unitary_norm_defect(registered_step_unitary(conditional["h_system"]))
        resolution = max(0.0, float(np.max(dimensions[:,1]) - np.min(dimensions[:,1])))
        values = {
          "TRACE_NORMALIZATION_TOL": (trace_normalization_defect(conditional["density"]), "<=", "PRIMARY", lambda x,t:x<=t),
          "HERMITICITY_TOL": (hermiticity_residual(conditional["density"]), "<=", "PRIMARY", lambda x,t:x<=t),
          "MIN_EIGENVALUE_TOL": (minimum_eigenvalue(conditional["density"]), ">=", "PRIMARY", lambda x,t:x>=t),
          "CONSTRAINT_RESIDUAL_TOL": (constraint_residual(conditional["k0"], conditional["state"]), "<=", "PRIMARY", lambda x,t:x<=t),
          "POVM_COMPLETENESS_TOL": (povm_completeness_residual(effects), "<=", "PRIMARY", lambda x,t:x<=t),
          "PROBABILITY_SUM_TOL": (probability_sum_defect(conditional["probabilities"]), "<=", "PRIMARY", lambda x,t:x<=t),
          "MIN_OUTCOME_PROBABILITY": (float(np.min(conditional["probabilities"])), ">=", "PRIMARY", lambda x,t:x>=t),
          "CONDITIONAL_TRACE_TOL": (max(trace_normalization_defect(r) for r in rhos), "<=", "PRIMARY", lambda x,t:x<=t),
          "CONDITIONAL_MIN_EIGENVALUE_TOL": (min(minimum_eigenvalue(r) for r in rhos), ">=", "PRIMARY", lambda x,t:x>=t),
          "FIDELITY_DEFECT_TOL": (float(np.max(1-analytic["fidelities"])), "<=", "PRIMARY", lambda x,t:x<=t),
          "TRACE_DISTANCE_TOL": (float(np.max(analytic["distances"])), "<=", "PRIMARY", lambda x,t:x<=t),
          "UNITARY_STEP_TRACE_DISTANCE_TOL": (float(np.max(analytic["steps"])), "<=", "PRIMARY", lambda x,t:x<=t),
          "UNITARY_NORM_DEFECT_TOL": (unitary_defect, "<=", "PRIMARY", lambda x,t:x<=t),
          "RESOLUTION_DETERIORATION_TOL": (resolution, "<=", "FINITE_DIMENSION", lambda x,t:x<=t),
          "WINDOW_BOUNDARY_TOL": (float(np.min(interaction[:,4])), ">", "SECTION_9", lambda x,t:x>t),
          "PROJECTOR_HERMITICITY_TOL": (float(np.max(interaction[:,5])), "<=", "SECTION_9", lambda x,t:x<=t),
          "PROJECTOR_IDEMPOTENCE_TOL": (float(np.max(interaction[:,6])), "<=", "SECTION_9", lambda x,t:x<=t),
          "PROJECTION_NORM_MIN": (float(np.min(interaction[:,7])), ">=", "SECTION_9", lambda x,t:x>=t),
          "G0_RECOVERY_TRACE_DISTANCE_TOL": (float(interaction[0,11]), "<=", "SECTION_9", lambda x,t:x<=t),
          "DETERMINISM_TRACE_DISTANCE_TOL": (float(np.max(interaction[:,13])), "<=", "DETERMINISM", lambda x,t:x<=t),
          "CLOCK_MAPPED_TRACE_DISTANCE_TOL": (float(np.max(clock_b[:,3])), "<=", "CLOCK_B", lambda x,t:x<=t),
          "CLOCK_PROBABILITY_DIFFERENCE_TOL": (float(np.max(np.abs(clock_b[:,2]-conditional["probabilities"]))), "<=", "CLOCK_B", lambda x,t:x<=t),
        }
        results = tuple(self._threshold(n, v[0], v[1], v[3](v[0], FROZEN_THRESHOLDS[n]) if v[0] is not None else False, v[2]) for n,v in values.items())
        finite = all(np.all(np.isfinite(value)) for value in diagnostics.values()) and set(dimensions[:,0]) == {8.0,16.0,32.0}
        return AdjudicationRecord(False, False, False, True, False, finite, results, controls, True)

    def _stage_g1_materialize_evidence(self, conditional, analytic, diagnostics, controls, provisional) -> None:
        arrays = {"conditional_densities.npy": conditional["densities"], "conditional_probabilities.npy": conditional["probabilities"],
                  "analytic_reference_densities.npy": analytic["references"], "fidelities.npy": analytic["fidelities"],
                  "trace_distances.npy": analytic["distances"], "neighbor_step_distances.npy": analytic["steps"],
                  "interaction_diagnostics.npy": diagnostics["interaction"], "clock_b_diagnostics.npy": diagnostics["clock_b"],
                  "coarse_clock_diagnostics.npy": diagnostics["coarse"]}
        arrays.pop("conditional_densities.npy"); arrays.pop("conditional_probabilities.npy")
        index = write_remaining_registered_arrays(self.evidence_root / "raw", arrays, conditional["stage_b_identities"])
        write_json_once(self.evidence_root / "raw_output_index.json", {"run_id": self.run_id, "arrays": index})
        write_json_once(self.evidence_root / "negative_controls.json", {"run_id": self.run_id, "controls": [[a.value,b.value,c] for a,b,c in controls]})
        write_json_once(self.evidence_root / "diagnostics.json", {"section9_fields":["g","exact_kernel_dimension","numerical_kernel_dimension","selected_rank","boundary_distance","projector_hermiticity_residual","projector_idempotence_residual","projection_norm","continued_state_constraint_residual","overlap_real","overlap_imag","g0_recovery_trace_distance","minimum_absolute_eigenvalue","determinism_trace_distance"], "values":{name: value.tolist() for name,value in diagnostics.items()}, "threshold_results":[item.__dict__ for item in provisional.threshold_results]})
        manifest = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json").read_text(encoding="utf-8"))
        environment = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/environment/QF0A_ENVIRONMENT_FREEZE_v1.1.json").read_text(encoding="utf-8"))
        write_json_once(self.evidence_root / "environment_manifest.json", environment)
        source_sha = write_source_manifest_with_sha(self.evidence_root, manifest)
        run_manifest = build_run_manifest(self.repo_root, self.run_id, manifest, environment, source_sha, self.provenance_seal)
        write_json_once(self.evidence_root / "run_manifest.json", run_manifest)
        write_json_once(self.evidence_root / "leakage_audit.json", {"run_id": self.run_id, "verdict": "PASS", "scope": "RUNTIME_PROVENANCE"})
        write_text_once(self.evidence_root / "human_report.md", f"# QF-0A v1.1 qualification report\n\nRun: `{self.run_id}`\n\nMachine-generated pipeline qualification only. No post-hoc scientific interpretation.\n")
        write_text_once(self.evidence_root / "logs/run.log", "frozen pre-finalization stages completed\n")

    def _stage_g2_validate_evidence(self, conditional, provisional):
        for name, meta in conditional["stage_b_identities"].items():
            actual = hashlib.sha256((self.evidence_root / "raw" / name).read_bytes()).hexdigest()
            if actual != meta["sha256"]: return replace(provisional, post_hoc_mutation=True)
        schema = json.loads((self.repo_root / "foundation/qf0/qf0a_impl/schemas/QF0A_RUN_MANIFEST.schema.json").read_text(encoding="utf-8"))
        document = json.loads((self.evidence_root / "run_manifest.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
        required = set(expected for expected in ("execution_authorization.json","run_manifest.json","environment_manifest.json","source_manifest.json","source_manifest.sha256","raw_output_index.json","negative_controls.json","leakage_audit.json","diagnostics.json","human_report.md","logs/run.log"))
        if not all((self.evidence_root / item).is_file() for item in required): return provisional
        return replace(provisional, required_evidence_complete=True)

    def _stage_g3_final_sealing(self, verdict) -> None:
        write_json_once(self.evidence_root / "machine_verdict.json", {"run_id": verdict.run_id, "pipeline_reproduction_verdict": verdict.pipeline_reproduction_verdict,
            "foundation_model_verdict": verdict.foundation_model_verdict.value, "reason_codes": list(verdict.reason_codes)})
        hashes = []
        for path in sorted(p for p in self.evidence_root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(self.evidence_root).as_posix()}")
        write_text_once(self.evidence_root / "SHA256SUMS.txt", "\n".join(hashes)+"\n")
        validate_complete_inventory(self.evidence_root)
