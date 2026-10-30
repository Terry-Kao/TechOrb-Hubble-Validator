# Host-Satisfiable Main-Lineage Re-Freeze

Candidate for independent re-freeze acceptance; not formally accepted.

PR #23 independently approved head f38bf8561891f820517c873e70960dca200069a8 merged as M4 cb1c44c0b888595ef07bf3bc68db6eb3fe43aa48.
M4 is authoritative main and the exact new source baseline. Parents: 6aaf4a24882fcc527ff82abfd1001413064c42b9 and f38bf8561891f820517c873e70960dca200069a8.
Merge preserved the complete reviewed tree unchanged. Old manifest blob 8a240e7effcdd6b2fe7adf053aea1b5203d2a38a survived the repair merge.

F-14 was confirmed: platform.processor returned AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, differing from the old frozen marketing identity.
Official CPU source is Windows registry HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString, with no platform.processor fallback and fail-closed registry errors.
Whitespace-only normalization yields AMD Ryzen 7 9700X 8-Core Processor.
F-15 positive observation coverage is present. F-16 observes jsonschema 4.23.0 and behaviorally validates date-time support.
F-14/F-15/F-16: CLOSED CANDIDATE pending independent review; no formal governance closure in this branch.

HOST-LOCAL OBSERVATION after merge, exact frozen venv D:\AI_Project\TechOrb\QF0A\env\qf0a-v1.1-py31210\Scripts\python.exe: FULL POSITIVE HOST SATISFIABILITY PASS.
Raw CPU: 'AMD Ryzen 7 9700X 8-Core Processor             '; normalized as above.
CPython 3.12.10 AMD64; Windows 11 Home 10.0.26200; base executable SHA-256 4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a.
NumPy 2.1.3, scipy-openblas 0.3.27 ILP64 DYNAMIC_ARCH Haswell; SciPy 1.14.1, scipy-openblas 0.3.27.dev LP64 DYNAMIC_ARCH Zen.
All five prescribed thread variables set to 1; float64 8 bytes, complex128 16 bytes; jsonschema 4.23.0; valid date-time accepted and invalid rejected. No additional host mismatch.

Manifest regenerated solely from M4 Git committed bytes, git-committed-bytes-v1; scientific_execution NOT_PERFORMED.
Inventory 41 to 41, all paths retained; no additions/removals. Root repair reports remain governance artifacts excluded under existing convention. No self-reference.
Manifest blob f74f1bf4e9fd6e81b2c11c1bf2008198953d1673.
Manifest canonical SHA-256 fb072ab0abeb443baf52f15807a827e94b43524081b89ffb3873da7df9995241.

Re-freeze changes only manifest and this report. Protocol, scientific source, schemas, runtime, environment, preregistration, tests and dependencies unchanged from M4.
QUAL-001 remains IMMUTABLE HISTORICAL ATTEMPT / NUMERICAL GATES PASS / EVIDENCE CONTRACT BLOCKED; Foundation NOT_APPLICABLE. Historical bytes untouched.
QUAL-002 authorization NOT CREATED / UNAUTHORIZED; run NOT STARTED; evidence ABSENT. No launcher, entrypoint, runner execute or scientific execution.

Next gate: Independent narrow read-only review of QF-0A v1.1 Host-Satisfiable Main-Lineage Re-Freeze.
