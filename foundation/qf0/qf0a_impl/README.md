# QF-0A implementation boundary

Status: `PREREGISTRATION DRAFT; SCIENTIFIC IMPLEMENTATION NOT CREATED; REPRODUCTION NOT EXECUTED`.

This directory contains the historical accepted v1.0 freeze and the proposed v1.1 pre-execution repair. The v1.1 binding authorities are `../QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md` and `../QF0A_FROZEN_PROTOCOL_v1.1.md`. v1.0 is preserved as historical process evidence and is superseded for any future execution.

The v1.1 tree includes scientific primitives, frozen metrics, automatic adjudication, a predetermined evidence writer, an authorization schema, and runner orchestration. Their presence does not authorize or perform a run. Scientific execution remains unauthorized.

Present in this checkpoint:

- four JSON Schemas under `schemas/`;
- the leakage-audit checklist;
- validation-only dependency pin `jsonschema[format]==4.23.0`;
- a Draft 2020-12/schema/semantic validator and non-adjudicative synthetic fixtures under `tests/`.

Deliberately absent: scientific source modules, environment/result instances, raw arrays, generated outputs, machine verdicts, scientific reports, and evidence runs. The empty future directories `manifests/`, `src/`, and `evidence/` are specified in the preregistration but are not created by this PR.
