# Decision Log

## D-2026-10-20-01 — Governance Reset

**Status:** Approved by merged PR #2 (4134c4b0e95c28975e4a64e97157ffc9c7b2f843)  
**Decision:** Separate Tech-Orb mother hypothesis from all validators; establish Foundation F0/F1/F2, an epistemic/evidence ledger, bridge register, falsification matrix, and HEM v27 R0–R4 protocol.

**Reason:** The prior repository conflated specific HIA/Hubble implementations, numerical target attainment, and broad scientific conclusions. The reset makes claim scope and evidence state auditable.

**Preserved:** All pre-reset HIA/RMP code, paper sources, plots, and audit records remain available as historical evidence.

## D-2026-10-20-02 — Bound the late-time negative result

**Status:** Approved by merged PR #2  
**Decision:** Reclassify the old “comprehensive No-Go theorem” as a historical exclusion study of tested late-time subclasses. Universal wording is superseded.

**Reason:** The repository contains numerical investigations of particular kinematic, screened-gravity, and disformal implementations, not a proof covering all smooth late-time theories.

## D-2026-10-20-03 — Classify HEM v26.4

**Status:** Approved by merged PR #2  
**Decision:** Record v26.4 only as a CLASS-level sensitivity/degeneracy experiment with evidence status `REPRO-BLOCKED` before R0.

**Not established:** a data-preferred posterior, Planck acceptance, full multi-probe viability, physical PMF/magnetogenesis derivation, or Foundation validation.

## D-2026-10-20-04 — Freeze HEM v27 gates

**Status:** Approved by merged PR #2  
**Decision:** Canonical order is R0 Reproducibility Freeze → R1 Baseline Definition Audit → R2 Published Model Reproduction → R3 Full Multi-Probe Likelihood → R4 Physicality Benchmark.

**Rules:** Pipeline PASS is separate from model survival; `H0 > 71` is forbidden as a gate criterion; BBN has only the scoped early-universe consistency role recorded in the gate protocol.

## D-2026-10-20-05 — Freeze candidate R0 recovery; retain BLOCKED

**Status:** Proposed in R0 recovery draft PR; no gate promotion  
**Decision owner:** Repository maintainer through PR review  
**Previous state:** R0 NOT PASSED; v26.4 REPRO-BLOCKED.  
**Proposed state:** R0 BLOCKED / NOT PASSED; model NOT APPLICABLE. Evidence status remains REPRO-BLOCKED.

**Evidence:** [recovery package](../validators/hem_v27/r0/recovery-2026-10-20/README.md), [manifest](../validators/hem_v27/r0/recovery-2026-10-20/MANIFEST.json), [issue #3](https://github.com/Terry-Kao/TechOrb-Hubble-Validator/issues/3).

**Scope:** Freeze inspected candidate runner/notebook, patched RECFAST source, complete tracked diff, small historical summary and source/output hashes. Record that N_ncdm=0 is inherited from CLASS defaults and distinguish the 0.06 eV neutrino comparison. Larger outputs remain local and indexed by checksum.

**Unresolved:** ENV-02 (available Docker engine stores container data on unauthorized C:; sole existing WSL distro lacks toolchain), PROV-01 (historical build/runtime identity), RAW-01 (deleted INI/raw outputs), and BASE-01 (baseline conventions). Source-to-output attribution and independent clean-room reproduction remain unproved. ENV-01 is closed after the reboot; TOL-01 is closed by the pre-run contract.

**Promotion condition:** A separate auditable clean-room gate report must resolve these blockers and satisfy GATE_PROTOCOL.md before any R0 PASS decision. Any recovered mismatch remains visible and prevents promotion.

## D-2026-10-20-06 — Freeze first-rerun tolerances; retain BLOCKED after reboot

**Status:** Proposed in R0 recovery draft PR; no gate promotion
**Decision:** Freeze `TOLERANCE_CONTRACT.md` before any numerical rerun. Close ENV-01 because Docker Desktop and its Linux engine became reachable after reboot. Open ENV-02 because Docker's data VHDX is on C: and the only existing WSL distro lacks the required toolchain.

**Execution consequence:** Do not create a container, image, distro, build, or numerical output until a pre-existing clean Linux runtime can keep all task artifacts, metadata, caches, temporary files, logs, and outputs within the authorized H: tree.

**Gate consequence:** R0 remains BLOCKED / NOT PASSED; model verdict remains NOT APPLICABLE; R1-R4 remain NOT STARTED.

## Decision Entry Template

- ID / date
- status and decision owner
- changed claim or gate
- previous state
- new state
- evidence links and exact versions
- scope and unresolved objections
- rollback/supersession condition

