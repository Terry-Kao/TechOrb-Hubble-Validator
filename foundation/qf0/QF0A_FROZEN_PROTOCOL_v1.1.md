# QF-0A Frozen Protocol v1.1 — Pre-Execution Repair

**Status:** `PROPOSED / PRE-EXECUTION REPAIR / NOT EXECUTED`  
**Classification:** `SOFTWARE QUALIFICATION / KNOWN-RESULT REPRODUCTION / CONSTRUCTIBILITY DEMONSTRATION`  
**Supersedes for future execution:** v1.0, which remains historical and was never executed.

## 1. Scope and epistemic boundary

This version retains the v1.0 finite-dimensional definitions unless explicitly amended below. For the frozen ideal construction, the primary Page–Wootters-type conditional-state relation is an exact algebraic identity. Numerical execution verifies software, numerical plumbing, evidence completeness, and provenance against that known identity. Any material primary deviation is implementation/numerical/provenance failure, not evidence against relational time as a physical theory.

A pipeline PASS is software qualification only. It is not evidence that time in nature emerges this way, support for Tech-Orb Foundation ontology, proof of no external time, general clock-choice independence, or interaction robustness. `foundation_model_verdict` remains `NOT_APPLICABLE` or `NOT_ESTABLISHED`.

## 2. Frozen mathematical construction

Use `hbar=1`, `d_C=16`, `d_S=2`, `omega=1`,

`H_C=sum_n omega*n |n><n|`, `H_S=(omega/2)(I-sigma_z)`, and

`K_0=H_C tensor I_S-I_C tensor H_S`.

The primary state is `|Psi_0>=(|0,0>+|1,1>)/sqrt(2)`. Clock states and POVM are

`|tau_k>=d_C^(-1/2) sum_n exp(+i*2*pi*n*k/d_C)|n>` and `E_k=|tau_k><tau_k|`, with `tau_k=2*pi*k/(d_C*omega)`.

The conditional density is `rho_S|k=Tr_C[(E_k tensor I)rho]/p_k`. The analytic reference is independently calculated as `rho_ref(tau_k)=|psi_ref><psi_ref|`, `|psi_ref>=exp(-i H_S tau_k)|psi_0>`, `|psi_0>=(|0>+|1>)/sqrt(2)`.

## 3. Frozen metrics and thresholds

Fidelity convention for the pure analytic reference is squared fidelity: `F(rho,rho_ref)=Re Tr(rho*rho_ref)`. Trace distance is `D(rho,sigma)=0.5*||rho-sigma||_1`, evaluated from the singular/eigenvalue spectrum of the Hermitian difference.

Freeze: trace/conditional-trace defect `<=1e-12`; Hermiticity `<=1e-12`; minimum eigenvalue `>=-1e-12`; constraint residual `<=1e-12`; POVM completeness `<=1e-12`; probability-sum defect `<=1e-12`; every adjudicated probability `>=1e-12`; fidelity defect `1-F<=1e-10`; primary trace distance `<=1e-10`; neighboring-step trace distance `<=1e-10`; unitary norm defect `<=1e-12`.

`Delta_tau=2*pi/(d_C*omega)` and `U_delta=exp(-i H_S Delta_tau)`. Neighbor checks include every `k`, including cyclic final `d_C-1 -> 0`, using the periodic registered ideal orbit.

## 4. Cross-dimension implementation consistency

Run `d_C in {8,16,32}` only after authorization. With registered two-level energy support and the exact ideal clock construction, exact agreement is analytically expected at every dimension up to floating-point error. This is a cross-dimension implementation-consistency diagnostic, not physical clock imperfection or nontrivial convergence. The historical deterioration guard `1e-8` remains a software-failure threshold and is not a physical scaling claim.

## 5. Near-kernel interaction diagnostic

Retain `K(g)=K_0+gV`, `V=|1><1|_C tensor sigma_x`, `g in {0,1e-4,1e-3,1e-2}`, `W=(-0.5,0.5)`, `P_g=1_W(K(g))`, and `|Psi_g>=P_g|Psi_0>/||P_g|Psi_0>||` with nonnegative-real overlap phase.

For every registered `g!=0`, the exact kernel is one-dimensional and contains stationary `|0,0>`. The projected continuation is not an exact constrained physical state or interacting Page–Wootters solution. It is only a near-kernel continuation geometry/numerical diagnostic.

Report exact/numerical kernel dimension, minimum absolute eigenvalues, constraint residual, selected rank, boundary distances, projector Hermiticity/idempotence, projection norm, overlap, and registered state comparisons. Rank must be 2; boundary distance must exceed `1e-10`; projector residuals `<=1e-12`; projection norm `>=1e-12`; `g=0` recovery and deterministic density agreement `<=1e-12`. Boundary crossing invalidates the protocol version; the window is immutable.

## 6. Clock B bookkeeping check

Clock B is `|tilde_tau_k>=exp(i*2*pi*k/d_C)|tau_(k+3 mod d_C)>`. Compare after the exact map `k -> (k+3) mod d_C`; mapped trace distance `<=1e-10`, probability difference `<=1e-12`.

PASS establishes only correct registered relabeling, POVM rephasing invariance, and index mapping. It does not establish general clock-choice independence, solve clock ambiguity, or demonstrate temporal quantum-reference-frame equivalence.

## 7. Coarse-grained clock descriptive diagnostic

For `E'_j=E_(2j)+E_(2j+1)`, report without post-hoc target selection:

1. trace distance to `rho_ref(tau_(2j))`;
2. trace distance to `rho_ref(tau_(2j+1))`;
3. trace distance to `rho_ref((tau_(2j)+tau_(2j+1))/2)`;
4. purity `Tr(rho_coarse^2)`;
5. eigenvalues/mixedness diagnostics.

No equivalence PASS is expected and no scalar contributes to primary PASS. Only finite/complete values, normalization, positivity, and complete recording are structural validity gates. This demonstrates registered information loss, not general clock ambiguity.

## 8. Exact negative-control inventory and adjudication

The six controls remain exactly: `BROKEN_CONSTRAINT`, `PRODUCT_NO_HISTORY`, `WRONG_CONDITIONING`, `HIDDEN_EXTERNAL_TIME`, `CLOCK_SCRAMBLE`, `POST_HOC_THRESHOLD_MUTATION`.

- Broken constraint must trigger `CONSTRAINT_RESIDUAL_EXPECTED_FAILURE`.
- Product/no-history permits zero-probability outcomes as `REGISTERED_EXPECTED_CONTROL_FAILURE`, never generic BLOCKED; missing classification is FAIL.
- Wrong conditioning must trigger orbit/step mismatch.
- Hidden external time is a **provenance attestation / construction-origin check**. It cannot infer origin from the final state; bad or missing attestation invalidates provenance.
- Clock scramble uses only `pi(k)=(5k+1) mod 16` and the frozen list; unexpected comparison PASS is FAIL.
- Post-hoc mutation or independent identity mismatch is INVALIDATED.

Execution exception/crash is `BLOCKED` unless it proves contamination/mutation, which is `INVALIDATED`. Missing control output is `BLOCKED`. The control set cannot be caller-supplied.

## 9. Result blindness and construction provenance

The primary analytic answer is known. Blindness means no stored target arrays enter source; conditional code receives no analytic outputs; conditional outputs are persisted and hashed before analytic comparison; thresholds, inclusion, adjudication, controls, and evidence cannot change after inspection.

The history-state representation is mathematically equivalent to the registered analytic history representation. Output alone cannot distinguish direct constrained construction from mathematically equivalent external-time packaging. The provenance attestation is source/process discipline, not an observable-output discriminator.

## 10. Automatic verdict and precedence

The machine derives, never accepts, the pipeline verdict. Precedence is `INVALIDATED > BLOCKED > FAIL > PASS`: contamination/mutation/identity mismatch; missing capability/evidence; completed threshold/control failure; otherwise complete qualification PASS. Foundation verdict cannot be promoted.

## 11. Authorization and evidence

No scientific computation may start until a schema-valid authorization binds run ID, this protocol blob/hash, preregistration blob/hash, environment hash, source manifest hash, source baseline, runner identity, issuer/governance record, timestamp, and `authorized=true`.

The runner must validate provenance and a clean/new evidence root; compute and hash conditional outputs before analytic reference; then diagnostics, exact controls, adjudication, and complete evidence finalization. Raw inventory is predetermined. No `--force`, result exclusion, missing-outcome suppression, or runtime source mutation is allowed.

## 12. Execution status

`PROTOCOL FROZEN; REPRODUCTION NOT EXECUTED`. This Draft repair authorizes no run. The next gate is independent read-only review of the v1.1 pre-execution repair/re-freeze.
