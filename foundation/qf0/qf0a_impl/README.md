# QF-0A implementation boundary

Status: `PRE-EXECUTION REPAIR / RE-FREEZE; SCIENTIFIC EXECUTION NOT AUTHORIZED; REPRODUCTION NOT EXECUTED`.

This directory contains the historical accepted v1.0 freeze and the proposed v1.1 pre-execution repair. The v1.1 binding authorities are `../QF0A_IMPLEMENTATION_PREREGISTRATION_v1.1.md` and `../QF0A_FROZEN_PROTOCOL_v1.1.md`. v1.0 is preserved as historical process evidence and is superseded for any future execution.

The v1.1 tree includes scientific primitives, frozen metrics, automatic adjudication, a predetermined evidence writer, an authorization schema, and runner orchestration. Their presence does not authorize or perform a run. Scientific execution remains unauthorized.

Present in this checkpoint:

- four JSON Schemas under `schemas/`;
- the leakage-audit checklist;
- validation-only dependency pin `jsonschema[format]==4.23.0`;
- a Draft 2020-12/schema/semantic validator and non-adjudicative synthetic fixtures under `tests/`.

Deliberately absent: environment/result instances, raw arrays, generated outputs, machine verdicts, scientific reports, and evidence runs. The committed source is inert until a separate identity-bound authorization passes Stage A.

The only supported future execution entry point is a fresh process invoking `python -m qf0a.entrypoint`. It validates the authorization-bound committed HEAD manifest and active source inventory before importing `qf0a.runner`. Interactive/notebook invocation and direct construction of the runner without a pre-import `ExecutionSourceSeal` are unsupported and fail closed.
