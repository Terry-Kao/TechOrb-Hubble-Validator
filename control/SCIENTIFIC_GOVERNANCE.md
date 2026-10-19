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

Foundation canon contains F0 hypotheses only. F1 and F2 are research modules. A proposed action, mapping, closure, or observable link begins in the Bridge Register as `Candidate` and normally as `ANSATZ` or `HYPOTHESIS`.

Promotion requires:

1. explicit mathematical definition and domain;
2. covariance/gauge and dimensional analysis;
3. recovery of relevant known limits;
4. a unique or discriminating observable consequence;
5. an executable falsification test;
6. a Decision Log entry with supporting evidence.

Any magnetogenesis action remains a Candidate Bridge / ANSATZ until it meets these conditions.

## 6. Negative Results

Negative results must state tested assumptions, parameter domain, observables, implementation identity, and failure threshold. Phrases such as “all late-time solutions” or “definitive theorem” are prohibited unless the proof truly covers the universal domain.

## 7. Changes and Review

Material changes arrive by branch and pull request. The PR must list scientific claims changed, evidence preserved, gates affected, and unresolved questions. Merge records governance approval, not scientific validation.

