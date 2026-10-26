# QF-0A Frozen Protocol v1.0 — Relational Time Known-Result Reproduction

**Freeze date:** 2026-10-26 (Asia/Taipei)  
**Protocol status:** FROZEN BEFORE IMPLEMENTATION OR RESULT INSPECTION  
**Scientific status:** NOT EXECUTED; NO RESULT; NO NEW-PHYSICS CLAIM  
**Registered claim:** `QF-TIME-001`

## 1. Objective and scope

Reproduce a finite-dimensional Page–Wootters-type known relational-time result: a globally constrained clock–system state yields normalized conditional system states approximating the reference Schrödinger orbit when conditioned on internal clock outcomes.

This gate tests whether a future implementation correctly reproduces a known toy mechanism under a frozen contract. PASS would be a **reproduction/pipeline verdict only**. It would not be Tech-Orb evidence, would not establish the ontology, would not show that time in nature emerges this way, and would not advance QF-0B, QF-0C, QF-0D, QF-1, QF-2, or QF-3.

No numerical implementation or scientific execution is part of this work package. A schema/link/logic validation of this document is permitted and is not a scientific run.

## 2. Factorization and ansatz boundary

For this reproduction only, use

\[
\mathcal H=\mathcal H_C\otimes\mathcal H_S.
\]

This clock/system factorization, the selected clock Hamiltonian, the absence/presence of an interaction term, the clock POVM, and the finite truncation are all `ANSATZ`. They are not facts about nature. No external/meta-time may appear in the global physical-state definition or be used to generate the conditional orbit.

## 3. Frozen mathematical definitions

### 3.1 Clock space and Hamiltonian

Primary clock dimension: `d_C = 16`. Let `|n>_C`, `n=0,...,d_C-1`, be the clock energy basis and define

\[
H_C=\sum_{n=0}^{d_C-1}\omega n\,|n\rangle\langle n|,
\qquad \omega=1,
\qquad \hbar=1.
\]

Clock phase states are

\[
|\tau_k\rangle_C=\frac{1}{\sqrt{d_C}}\sum_{n=0}^{d_C-1}
e^{+i2\pi nk/d_C}|n\rangle_C,
\quad \tau_k=\frac{2\pi k}{d_C\omega}.
\]

The primary clock POVM is `E_k=|tau_k><tau_k|`; for this discrete orthonormal phase basis it is projective. The label `tau_k` is an internal clock reading, not an external time coordinate.

### 3.2 System space and Hamiltonian

Primary system dimension: `d_S = 2`, with

\[
H_S=\frac{\omega}{2}(I-\sigma_z),
\]

whose energies are `0` and `omega`. The reference initial state is

\[
|\psi_0\rangle_S=\frac{|0\rangle+|1\rangle}{\sqrt2}.
\]

The analytic reference orbit is

\[
|\psi_{\rm ref}(\tau_k)\rangle=e^{-iH_S\tau_k}|\psi_0\rangle.
\]

### 3.3 Global constraint and physical state

The non-interacting primary constraint is

\[
K_0=H_C\otimes I_S-I_C\otimes H_S,
\qquad K_0|\Psi\rangle=0.
\]

The sign convention is frozen and pairs equal clock/system energies. The physical history state must be constructed by solving/projecting onto the kernel of `K_0`, not by first generating an externally time-indexed trajectory and packaging it afterward.

The frozen primary history state is the normalized equal-amplitude correlated energy state

\[
|\Psi\rangle=\frac{|0\rangle_C|0\rangle_S+|1\rangle_C|1\rangle_S}{\sqrt2},
\]

embedded in the `d_C=16` clock space. Its constraint residual must be computed directly.

### 3.4 Conditional state

For global density operator `rho=|Psi><Psi|`, outcome probability and conditional state are

\[
p_k=\operatorname{Tr}[(E_k\otimes I_S)\rho],
\]

\[
\rho_{S|k}=\frac{\operatorname{Tr}_C[(E_k\otimes I_S)\rho]}{p_k}
\quad\text{for }p_k>0.
\]

Pure conditional vectors may be reported only after phase-invariant comparison. Density matrices are the adjudicative representation.

## 4. Analytic reference target

For the frozen primary state, the conditional state must equal the analytic reference orbit up to a global phase. With the frozen positive phase in `|tau_k>` and `K_0=H_C-H_S`, projection by `<tau_k|` produces the negative phase in `exp(-i H_S tau_k)`. The implementation must derive this convention algebraically before comparison and must not change the sign after viewing numeric results.

Primary adjudicative quantities:

- constraint residual `epsilon_K = ||K_0|Psi>||_2`;
- outcome probabilities `p_k`;
- conditional trace/positivity;
- state fidelity `F_k = F(rho_S|k, rho_ref(tau_k))`;
- trace distance `D_k = 0.5 ||rho_S|k-rho_ref(tau_k)||_1`;
- step propagator consistency between neighboring clock readings.

## 5. Numerical reproduction plan (future authorized run)

1. Build `H_C`, `H_S`, `K_0`, clock states/POVM, and the frozen physical state directly from the definitions.
2. Verify Hermiticity, dimensions, POVM completeness, global-state normalization, and constraint residual.
3. Compute `p_k` and every `rho_S|k` by partial trace/conditioning.
4. Compute the analytic reference density matrices independently from matrix exponentiation of `H_S`.
5. Compare by fidelity and trace distance; never compare raw vector phase.
6. Test approximate unitarity using a single system step operator `U_delta=exp(-i H_S Delta_tau)` and compare `rho_S|k+1` with `U_delta rho_S|k U_delta^dagger`.
7. Execute the finite-resolution ideal-clock, interaction-sensitivity, clock-choice, and negative-control suites without changing this protocol.
8. Write raw arrays, environment/package identity, exact command, seed, hashes, and a machine verdict before any interpretation.

No parameter fitting is allowed in the primary reproduction.

## 6. Normalization, positivity, and constraint checks

Required exact/near-exact checks:

- `|Tr(rho)-1| <= 1e-12`;
- `||rho-rho^dagger||_F <= 1e-12`;
- minimum eigenvalue of `rho >= -1e-12`;
- `epsilon_K <= 1e-12` for the primary ideal state;
- `||sum_k E_k-I_C||_F <= 1e-12`;
- `|sum_k p_k-1| <= 1e-12`;
- all adjudicated `p_k >= 1e-12`;
- for every conditional state, `|Tr(rho_S|k)-1| <= 1e-12` and minimum eigenvalue `>= -1e-12`.

Any threshold violation is FAIL, not a warning.

## 7. Approximate unitarity checks

For the primary ideal case:

- minimum state fidelity across all clock readings: `min_k F_k >= 1-1e-10`;
- maximum trace distance: `max_k D_k <= 1e-10`;
- maximum neighboring-step trace distance from the analytic unitary update: `<= 1e-10`;
- norm defect of `U_delta`: `||U_delta^dagger U_delta-I||_F <= 1e-12`.

These are numerical reproduction tolerances, not physical accuracy claims.

## 8. Finite-resolution ideal-clock diagnostic

The preregistered finite-resolution summary is

\[
E_{clock}=\max_k D(\rho_{S|k},\rho_{ref}(\tau_k)),
\]

with secondary reports of median trace distance and worst-case normalization error. Future execution must include `d_C in {8,16,32}` while keeping the embedded two-level energy support and the same analytic target.

This suite changes only ideal discrete-clock dimension/resolution and sampling. It does **not** test a realistic imperfect clock. Smeared/nonorthogonal clock POVMs, clock dephasing, finite coherence, an imperfect clock Hamiltonian, and clock-system coupling are future extensions requiring a separately frozen protocol.

Expected ideal discretization behavior is not itself a PASS condition. The suite FAILS if error is non-finite, normalization fails, or increasing clock resolution produces an unexplained deterioration larger than `1e-8` in the ideal construction. Any interpretation of scaling must be descriptive and may not be upgraded to evidence for nature.

## 9. Interaction sensitivity

Use the frozen family

\[
K(g)=K_0+g\,V,\qquad V=|1\rangle\langle1|_C\otimes\sigma_x,
\]

with `g in {0, 1e-4, 1e-3, 1e-2}`. This diagnostic retains `V` and uses the following basis-independent continuation rule; it does not select an individual eigenvector.

Let the frozen primary history state be `|Psi_0>=(|0,0>+|1,1>)/sqrt(2)`. Define the fixed near-kernel spectral window

\[
W=(-0.5,0.5)
\]

and the orthogonal spectral projector

\[
P_g=\mathbf 1_W(K(g)).
\]

The continued diagnostic state is uniquely defined by

\[
|\Psi_g\rangle=\frac{P_g|\Psi_0\rangle}{\|P_g|\Psi_0\rangle\|}.
\]

Fix its global phase by requiring `<Psi_0|Psi_g>` to be real and nonnegative. At `g=0`, `P_0` projects onto `ker(K_0)=span{|0,0>,|1,1>}`, so this rule recovers `|Psi_0>` exactly.

The unperturbed nonzero spectrum is separated from zero by a gap of `1`, `||V||_2=1`, and the frozen suite has `|g|<=0.01`. Weyl's bound therefore keeps the two eigenvalues continuing from the zero eigenspace within `[-0.01,0.01]` and all other eigenvalues outside `[-0.99,0.99]`; the fixed window selects exactly the two-dimensional relevant spectral subspace throughout the suite. The exact stationary zero mode `|0,0>` may remain inside that subspace, but it cannot hijack selection because no lowest-absolute-eigenvalue eigenvector is chosen: the full projector acts on the frozen history state.

Deterministic numerical construction: perform a Hermitian eigendecomposition of `K(g)` in `complex128`; include every eigenvector whose eigenvalue satisfies strict `-0.5 < lambda < 0.5`; construct `P_g` as the sum of their outer products; project and normalize `|Psi_0>`; then apply the phase rule. Rotating the eigensolver basis inside a degenerate subspace leaves the summed projector unchanged. No returned-eigenvector index or tie rule enters the definition.

The interaction diagnostic is BLOCKED/FAIL if the selected subspace rank is not exactly `2`, any eigenvalue lies within `1e-10` of a window boundary, `||P_g|Psi_0>|| < 1e-12`, projector Hermiticity/idempotence exceeds `1e-12`, the `g=0` state differs from `|Psi_0>` by trace distance above `1e-12`, or repeated allowed seeds/platform runs change the continued density matrix by trace distance above `1e-12`. An eigenvalue crossing at or through the fixed window boundary invalidates this protocol version; the window may not be moved after inspection.

Report `E_clock(g)`, spectral-window eigenvalues/rank, projector residuals, constraint residual, projection norm, and overlap with the primary state. This continued state is a near-kernel spectral-subspace diagnostic, not a claim of an exact interacting physical state. The suite is not part of the primary known-result PASS and cannot be promoted to Tech-Orb evidence. It FAILS if results are non-deterministic, the projector/continuation record is incomplete, or `g=0` does not reproduce the primary case. No post-hoc alternate interaction, spectral window, projector, or selection rule may replace the frozen definitions in v1.0.

## 10. Two clock-choice comparison

Clock choices are frozen as:

1. **Clock A:** the phase basis `|tau_k>` above.
2. **Clock B:** a relabelled/rephased operationally equivalent clock `|tilde_tau_k>=exp(i phi_k)|tau_{(k+3) mod d_C}>`, with `phi_k=2*pi*k/d_C`.

Because POVM elements discard the rephasing, Clock B must match Clock A after the registered cyclic outcome map `k -> (k+3) mod d_C`. Compare conditional density matrices after this map.

Acceptance: maximum mapped trace distance `<= 1e-10` and probability difference `<= 1e-12`. Failure demonstrates a basis/relabel implementation defect. It does not establish general multiple-clock equivalence.

A genuinely inequivalent coarse-grained clock is included as a diagnostic: pair adjacent Clock A outcomes, condition on `E'_j=E_{2j}+E_{2j+1}`, and report the resulting mixed-state error. No equivalence PASS is expected; the result must be explained operationally as information loss rather than hidden arbitrariness.

## 11. Randomness and deterministic settings

The primary, resolution, interaction, and registered two-clock suites are deterministic. If a library requires a random seed for eigensolver initialization, use seeds `[104729, 130363, 155921]`, run all three, and require adjudicative quantities to agree within `1e-12` or FAIL determinism.

No other seed may replace these after result inspection. Record language, package versions, BLAS/LAPACK identity, platform, and numeric dtype (`float64`/`complex128` minimum).

## 12. Negative controls

All controls must be run in the same implementation without modifying thresholds:

1. **Broken constraint:** replace the physical state by `( |0,0> + |1,0> )/sqrt(2)`. It must fail `epsilon_K <= 1e-12` and must not receive PASS.
2. **Product/no-history state:** `|tau_0>_C tensor |psi_0>_S`. It must not reproduce the full conditional orbit; unavailable outcomes must not be silently discarded.
3. **Wrong conditioning:** use `rho'_S|k=Tr_C(rho)/Tr(rho)` independent of `k`. It must fail orbit fidelity/step checks.
4. **Hidden external-time construction detector:** reject any artifact whose conditional states were generated first with an external-time loop and only then assembled as the global state, even if numerical values match.
5. **Clock-scramble control:** for `d_C=16`, apply the fixed permutation `pi=[1,6,11,0,5,10,15,4,9,14,3,8,13,2,7,12]`, equivalently `pi(k)=(5k+1) mod 16`, to the clock labels without supplying its inverse to adjudication. This is not any registered cyclic map `k -> k+c`. The machine report must record the exact list and formula. It must fail the reference comparison; it may not be resampled, replaced, inverted, or repaired after inspection. An unexpected PASS blocks the gate.
6. **Post-hoc threshold mutation:** a changed metric, sign convention, tolerance, state, or excluded outcome invalidates the run and yields protocol FAIL.

Control admission requires their expected failure reasons to be visible in the machine report. A control unexpectedly passing blocks the entire gate pending a new protocol version; it may not be explained away inside v1.0.

## 13. Explicit FAIL gates

QF-0A v1.0 is FAIL if any of the following occurs:

1. Effective evolution can be obtained only by secretly adding classical external/meta-time.
2. Conditional state normalization is unstable or violates the frozen thresholds.
3. Results depend wholly on arbitrary clock choice without operational equivalence, a registered transformation, or a controlled explanation.
4. A threshold, metric, sign, exclusion, or parameter must be changed after viewing results to obtain PASS.
5. The primary state/constraint/POVM identity is not exact and auditable.
6. The analytic reference is not independently computed or phase/sign conventions are chosen post hoc.
7. Any required negative control behaves contrary to its frozen expectation.
8. Source, environment, seeds, raw output, or hashes required for reproduction are missing.
9. The report promotes reproduction to Tech-Orb evidence or claims new physics.

Missing execution capability or missing required evidence is `BLOCKED`, not PASS.

## 14. Definition of Done for a future QF-0A execution

QF-0A execution is complete only when:

- this exact protocol version predates all adjudicative outputs;
- implementation source and environment are frozen and hashed;
- the primary analytic reproduction passes every identity, constraint, normalization, positivity, fidelity, trace-distance, and unitary-step threshold;
- finite-resolution ideal-clock, interaction, and both clock-choice suites are reported without post-hoc substitutions;
- every negative control produces its registered failure;
- all seeds/deterministic settings and raw arrays are preserved;
- a machine-readable verdict and human report distinguish pipeline/reproduction verdict from Foundation/model verdict;
- an independent review confirms no external-time or answer leakage;
- the final wording states: **known-result reproduction only; no new physics; not Tech-Orb evidence**.

Until all items are satisfied in a separately authorized execution, the status remains `PROTOCOL FROZEN; REPRODUCTION NOT EXECUTED`.
