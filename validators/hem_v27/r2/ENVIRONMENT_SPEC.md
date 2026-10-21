# R2 Environment Specification

Planned workspace: `H:\AI_Products\AI_Factory_Workspace\TechOrb-Hubble-Validator-R2-published-reproduction-20261021`.

No workspace is created until reference data and tolerances are admitted. When admitted:

1. Export the exact author commit as a pristine Git archive; record archive SHA256 and tree identity.
2. Reject and preserve any attempt with unexpected dirty paths before build.
3. Use an immutable container image/digest; Docker/WSL system components may write to C:, while all controllable source, configs, logs, outputs, manifests and reports remain under H:.
4. Record OS/container, compiler, libc, make, Python and dependency versions; build command; binary hash; stdout/stderr; and every run command.
5. Do not transplant the model into CLASS 3.2.0 for the primary paper benchmark. The exact author CLASS 2.9 fork is adjudicative. CLASS 3.2.0/R1 canonical-modern is a separate control only.
6. Network is disabled during builds/runs after all admitted source/runtime inputs are staged.
