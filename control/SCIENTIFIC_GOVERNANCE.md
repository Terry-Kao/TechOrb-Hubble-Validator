# Scientific Governance

## 1. Authority and Scope

The repository default branch is the source of truth after review and merge. Conversation text, AI review, local work, and unmerged branches are proposals only. Historical files remain evidence of what was previously claimed or computed, not automatic current truth.

## 2. Required Claim Labels

Every material claim must be assigned exactly one primary epistemic class:

- `FACT`: established external physics/observation or directly verified repository fact.
- `DERIVED`: follows from declared assumptions or a reproducible calculation.
- `HYPOTHESIS`: falsifiable physical proposition not yet established.
- `ANSATZ`: chosen exploratory form without first-principles derivation.
- `EXCLUDED`: bounded negative result; must include subtype and scope.
- `OPEN QUESTION`: unresolved mathematical, physical, or evidential question.

Evidence status is independent:

- `VERIFIED`, `SUPPORTED`, `PROVISIONAL`, `REPRO-BLOCKED`, `SUPERSEDED`, or `HISTORICAL`.

`EXCLUDED` must use one subtype:

- `CLAIM`: wording or inference is ruled out, while the implementation may remain useful.
- `IMPLEMENTATION`: a specific code/configuration/model realization fails.
- `SUBCLASS`: a defined family fails within declared assumptions and domain.

### 2.1 Relational Quantum Foundation label profile (effective 2026-10-26)

New QF Foundation claims use exactly one primary label: `ESTABLISHED`, `COMPATIBLE`, `HYPOTHESIS`, `ANSATZ`, or `REJECTED-LITERAL-VERSION`, as defined in `foundation/qf0/FOUNDATION_CANON_v1.0.md`. This bounded profile does not rewrite historical ledger entries.

- `ESTABLISHED` maps to a scoped mature external result or verified repository fact; it must not be applied to the Tech-Orb ontology merely because a prerequisite is established.
- `COMPATIBLE` means absence of a known conflict within a stated scope. It is not evidence or support.
- `REJECTED-LITERAL-VERSION` retires a literal wording/interpretation. It is narrower than a universal `EXCLUDED` result and cannot be used to claim a no-go theorem.

Every new Foundation claim must include a mathematical definition, dependencies, supporting and conflicting evidence, a fail/falsification condition, validator, current verdict, and review date in the QF claim registry. Claims without an executable fail condition remain concept notes and cannot enter Canon.

## 3. Evidence Rules

1. A claim must cite inputs, method, code/ref, outputs, uncertainty/tolerance, and scope.
2. Reproduction requires exact source, dependency/environment, parameter, patch, seed (if relevant), data, and output identity.
3. Theory-to-theory agreement, optimizer convergence, and a geometric lock are not observational likelihoods.
4. `r_s(z_*)` and `r_d = r_s(z_d)` are distinct and must be named explicitly.
5. An AI review can identify defects or propose tests. Agreement among AIs is not independent scientific evidence.
6. Absence of exclusion is not support. A promising numerical target is not a fact.

## 4. Gate Governance

- Gates R0–R4 must be executed in order unless a protocol amendment explains and approves an exception.
- Each gate has a `pipeline verdict` and, where applicable, a separate `model verdict`.
- A correct pipeline may PASS while the model is excluded, disfavored, or returns `H0 ≈ 69`.
- No target `H0`, including `H0 > 71`, may define gate success.
- A failed gate blocks downstream claims but does not automatically exclude the model or Foundation.
- Gate status changes require evidence links and a Decision Log entry.

## 5. Foundation and Bridge Governance

The active Foundation canon is the Relational Quantum Foundation in `foundation/qf0/FOUNDATION_CANON_v1.0.md`. Earlier F0/F1/F2 documents remain historical/provisional evidence and must not be deleted or silently rewritten. A proposed action, mapping, closure, or observable link begins in the Bridge Register as `Candidate` and normally as `ANSATZ` or `HYPOTHESIS`.

Promotion requires:

1. explicit mathematical definition and domain;
2. covariance/gauge and dimensional analysis;
3. recovery of relevant known limits;
4. a unique or discriminating observable consequence;
5. an executable falsification test;
6. a Decision Log entry with supporting evidence.

Any magnetogenesis action remains a Candidate Bridge / ANSATZ until it meets these conditions.

The evidential direction is one-way: Foundation → effective model → validators. HEM, Hubble/CLASS/HyRec results, anomaly targets, or any other validator may not reverse-define or promote Foundation. A PASS at one layer never establishes the next layer.

## 6. Negative Results

Negative results must state tested assumptions, parameter domain, observables, implementation identity, and failure threshold. Phrases such as “all late-time solutions” or “definitive theorem” are prohibited unless the proof truly covers the universal domain.

## 7. Changes and Review

Material changes arrive by branch and pull request. The PR must list scientific claims changed, evidence preserved, gates affected, and unresolved questions. Merge records governance approval, not scientific validation.

