# QF-0A implementation boundary

Status: `PREREGISTRATION DRAFT; SCIENTIFIC IMPLEMENTATION NOT CREATED; REPRODUCTION NOT EXECUTED`.

This directory contains contracts only. The binding authority is `../QF0A_IMPLEMENTATION_PREREGISTRATION_v1.0.md`; the frozen scientific protocol is `../QF0A_FROZEN_PROTOCOL_v1.0.md`.

Present in this checkpoint:

- four JSON Schemas under `schemas/`;
- the leakage-audit checklist;
- validation-only dependency pin `jsonschema[format]==4.23.0`;
- a Draft 2020-12/schema/semantic validator and non-adjudicative synthetic fixtures under `tests/`.

Deliberately absent: scientific source modules, environment/result instances, raw arrays, generated outputs, machine verdicts, scientific reports, and evidence runs. The empty future directories `manifests/`, `src/`, and `evidence/` are specified in the preregistration but are not created by this PR.
