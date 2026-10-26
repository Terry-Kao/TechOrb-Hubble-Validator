# QF-0B/C Benchmark Specification Skeleton v1.0

**Status:** STRUCTURE FROZEN; SCIENTIFIC PARAMETERS/THRESHOLDS NOT YET FROZEN  
**Date:** 2026-10-26  
**Execution:** NOT STARTED  
**Claims:** `QF-DIST-001`, `QF-DIM-001`

This document defines the required benchmark architecture only. It contains no full dataset, tuned parameters, scientific output, or PASS verdict. Before execution, each placeholder marked `TBF` must be resolved in a new reviewed protocol version committed before result inspection.

## 1. Shared anti-leakage architecture

The benchmark must use two physically and procedurally separated components:

1. **Hidden generator:** creates coordinates/topology/dimension, relational observations, noise, and ground-truth evaluation files. It writes an opaque instance ID and hashes. Ground truth remains inaccessible to reconstruction.
2. **Reconstruction pipeline:** receives only an instance ID, relational matrix/tensor, masks, and preregistered metadata that do not disclose coordinates, target dimension, topology class, or generator seed.

An adjudicator compares frozen reconstruction outputs with hidden ground truth only after outputs are immutable. Generator and reconstruction source/config hashes, operator identity, access logs, seeds, and timestamps must be preserved. Any coordinate, filename, dimension label, topology label, ordered node index, or generator parameter that reveals the answer is Validator Leakage and invalidates the run.

Data splits must be generated before reconstruction and include calibration/development instances whose answers may be used only under an explicitly frozen rule, plus adjudicative blind instances whose answers cannot be inspected or tuned against.

## 2. QF-0B — Emergent Distance

### 2.1 Required hidden targets

Each family must include noiseless and noisy variants, multiple sizes, and permuted node identities:

- 1D line;
- 1D ring;
- 2D square/rectangular lattice with boundary conditions declared;
- noisy variants with frozen additive, multiplicative, missing-edge, and spurious-edge regimes (`TBF` levels).

The reconstruction input may contain only preregistered relational data. It must not receive coordinates, Euclidean/geodesic distances, generator dimension, topology label, lattice side lengths, or neighbor order.

### 2.2 Preregistered relation-to-distance candidates

At least these two independent candidates must be fully parameterized before execution:

1. **Monotone transform candidate:** `d_ij^(A) = -log((R_ij+epsilon)/(R_max+epsilon))`, followed only by the frozen symmetrization/shortest-path rule. Freeze `epsilon`, clipping, disconnected handling, and any local/global transform.
2. **Diffusion/resistance candidate:** construct a normalized relational graph operator from `R`, then use either diffusion distance at frozen scale(s) or effective resistance/commute-time distance. Freeze graph construction, normalization, scales, regularization, and disconnected handling.

No candidate may be selected, blended, or reweighted after viewing blind results. Additional candidates require a new protocol version.

### 2.3 Metric/pseudometric checks

For every candidate and instance, report:

- finite/nonnegative values;
- symmetry defect;
- identity-of-indiscernibles violations;
- triangle-inequality violation rate and maximum normalized violation;
- connected-component coverage;
- whether the output is a metric, pseudometric, directed/quasimetric object, or FAIL.

The name must match the observed axioms. A pseudometric may not be reported as a metric.

### 2.4 Reconstruction metrics

Freeze aggregation and uncertainty rules for:

- geodesic distortion: stress and worst/quantile relative error after only the allowed scale alignment;
- rank distortion: Spearman/Kendall pair-distance order agreement;
- local-order/neighborhood preservation at preregistered `k` values;
- shortest-path order consistency;
- topology-sensitive summaries appropriate to line, ring, and lattice without leaking the family label to reconstruction.

All metrics, alignments, exclusions, and acceptance thresholds are `TBF` and must be frozen before execution.

### 2.5 Null controls

Required nulls:

- Erdos–Renyi/random weighted graph matched on node/edge count and global weight distribution;
- degree-preserving edge shuffle, with weight reassignment rule frozen;
- node-label permutation sanity control (must not change invariant reconstruction quality);
- relation-value permutation that destroys structural association while preserving the marginal distribution.

The real relational input must beat the registered null distributions under a frozen statistical/effect-size rule (`TBF`). Failing to distinguish nulls blocks any geometry claim.

### 2.6 QF-0B fail gates

FAIL if coordinates/dimension/topology leak; only one candidate survives because alternatives were dropped post hoc; declared metric checks fail without honest reclassification; blind distortion thresholds fail; nulls perform equivalently; hidden-order/file naming reveals structure; or thresholds are chosen after results.

## 3. QF-0C — Emergent Dimension

### 3.1 Blind targets

Generate blind synthetic targets with intrinsic dimensions `1`, `2`, and `3`; do not privilege or assume the answer `3`. Required families include regular lattices/manifolds and at least one nontrivial topology per dimension where feasible. Instance IDs and inputs must not encode dimension.

### 3.2 Independent estimators

At least two estimators with different failure modes are mandatory:

1. **Spectral/diffusion estimator:** return-probability/spectral-dimension scaling over preregistered scale windows.
2. **Local scaling estimator:** correlation dimension, nearest-neighbor maximum likelihood, or another frozen local-volume estimator.

Their preprocessing, scale/window selection, boundary handling, aggregation, and failure behavior must be frozen. Estimator agreement is evaluated; one estimator may not be silently discarded.

### 3.3 Uncertainty calibration

Pre-register bootstrap/subsampling or repeated-generator intervals, confidence/credible level (`TBF`), coverage target (`TBF`), and how correlated samples are handled. Report interval coverage and calibration error on calibration sets before unblinding adjudicative targets.

### 3.4 Stress grid

The frozen execution matrix must independently and jointly vary:

- finite size;
- relational noise level/type;
- sampling density and missingness;
- anisotropy/inhomogeneity where registered;
- topology change (for example line versus ring, open versus periodic lattice, holes/handles where defined);
- reconstruction error inherited from QF-0B versus ground-truth-distance diagnostic input, with these two tracks kept separate.

### 3.5 QF-0C metrics and fail gates

Report point error, interval coverage, estimator disagreement, scale stability, abstention/failure rate, and confusion across 1D/2D/3D. Thresholds are `TBF` until frozen.

FAIL if target dimension leaks; the pipeline assumes 3; either estimator lacks calibrated uncertainty; blind dimensions are not recovered under frozen thresholds; finite/noise/density/topology sensitivity is not mapped; estimator disagreement is hidden; or thresholds/windows are changed after unblinding.

## 4. Shared negative controls

- Feed random/shuffled relations and require abstention or failure rather than confident geometric/dimensional recovery.
- Provide duplicated/indistinguishable nodes to test pseudometric and local-dimension failure handling.
- Provide disconnected components and require the frozen infinity/component policy.
- Supply adversarial filenames/metadata in a development-only test to verify that the loader ignores answer-bearing fields.
- Run an answer-3 detector: search source/config/output for hard-coded target dimension, three-dimensional embedding defaults, or selection rules favoring 3.

## 5. Required artifacts for the future freeze

- generator spec and source hash;
- reconstruction spec and source hash;
- schemas for public input, hidden truth, immutable prediction, and verdict;
- all seeds and deterministic settings;
- full target/noise/size/sampling matrix;
- both QF-0B candidates and both QF-0C estimators;
- metric definitions, alignments, uncertainty procedure, thresholds, and multiplicity rule;
- null and negative controls;
- leakage audit and operator separation record;
- Definition of Done and explicit scope wording.

## 6. Non-skipping rule

QF-0B and QF-0C receive separate verdicts. QF-0B PASS does not establish QF-0C. QF-0C PASS on synthetic targets does not establish real spatial dimension, causal structure, GR, classical spacetime, or Tech-Orb. Neither benchmark may start until a later version resolves every `TBF` item and is frozen before implementation tuning or adjudicative data inspection.
