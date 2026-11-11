"""Read committed governance objects and report what this check covers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ANCHOR = "3ebb37a180d8acb7a663e50d219530d48c9334a7"
M4 = "cb1c44c0b888595ef07bf3bc68db6eb3fe43aa48"
ROOT = "control/r2/v1.0/"
FILES = {ROOT + name for name in (
    "AUTHORIZATION_BINDING.md", "HISTORICAL_QUAL002.md", "INDEX.md", "MODEL_B.md",
    "POLICY_CONTRACT.json", "QUALIFICATION_ROUTE.md", "TECHNICAL_BASELINE.md", "TRUSTED_G_SOURCE.md",
)}
CONTRACT = ROOT + "POLICY_CONTRACT.json"
MANIFEST = "foundation/qf0/qf0a_impl/manifests/QF0A_SOURCE_MANIFEST_v1.1.json"
RECORD = "control/DECISION_LOG.md"
PINS = {
    RECORD: "0637dbc7dfa5572ab654389c4a2159a518675df2",
    "control/PROJECT_STATE.md": "ab16c770e9dd2dfe2cc53febea577c7f48aedcf5",
    "control/EVIDENCE_INDEX.md": "3c9c6d8a49f921d5b8b92a7617175c04f726706c",
    "control/QF0A_QUAL002_AUTHORIZATION_EXECUTION_GATE_SPEC_v1.0.md": "4c629b70d6933733b7aa9298bd735accd8559bb9",
    "control/QF0A_QUAL002_P8_HASH_BINDING_CLARIFICATION_v1.0.md": "bca62cd5d19b32f989bffd6dd579b5f85e2d80cd",
    MANIFEST: "f74f1bf4e9fd6e81b2c11c1bf2008198953d1673",
}
JOB = "techorb-governance-policy-integrity"

class Rejected(Exception):
    pass

def require(value, reason):
    if not value:
        raise Rejected(reason)

def git(repo, *args):
    run = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args], capture_output=True)
    require(run.returncode == 0, "GIT_QUERY_FAILED")
    return run.stdout

def tree(repo, ref):
    entries = {}
    for item in git(repo, "ls-tree", "-r", "-z", ref).split(b"\0")[:-1]:
        meta, path = item.split(b"\t", 1)
        mode, kind, oid = meta.decode("ascii").split()
        entries[path.decode("utf-8")] = (mode, kind, oid)
    return entries

def raw(repo, oid):
    return git(repo, "cat-file", "blob", oid)

def strict_json(data):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "DUPLICATE_JSON_KEY")
            obj[key] = value
        return obj
    def invalid_constant(_):
        raise Rejected("JSON_SYNTAX")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise Rejected("JSON_SYNTAX") from exc

def verify_policy(repo, entries):
    paths = {p for p in entries if p.startswith(ROOT)}
    require(paths == FILES, "R2_INVENTORY")
    require(all(entries[p][:2] == ("100644", "blob") for p in paths), "POLICY_MODE")
    policy = strict_json(raw(repo, entries[CONTRACT][2]))
    fixed = {
        "version": ("1.0-candidate", "CANDIDATE_STATE"),
        "status": ("CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION", "CANDIDATE_STATE"),
        "phase": ("G-POLICY", "PHASE"),
        "formal_dispatch": (False, "FORMAL_DISPATCH"),
        "qual002_state": ("STATE_2_EXECUTION_HOLD", "QUAL002_STATE"),
        "qualification_state": ("R2_UNALLOCATED_HOLD", "R2_STATE"),
        "historical_anchor": (ANCHOR, "HISTORICAL_ANCHOR"),
        "historical_m4": (M4, "HISTORICAL_M4"),
        "decisions": ([f"DECISION-R2-{i:02}" for i in range(1, 6)], "DECISIONS"),
    }
    require(isinstance(policy, dict) and set(policy) == set(fixed) | {"allowed_paths"}, "JSON_SCHEMA_KEYS")
    for key, (value, reason) in fixed.items():
        require(type(policy[key]) is type(value) and policy[key] == value, reason)
    require(isinstance(policy["allowed_paths"], list) and all(isinstance(p,str) for p in policy["allowed_paths"])
            and len(policy["allowed_paths"]) == 8 and set(policy["allowed_paths"]) == FILES, "R2_ALLOWLIST")
    docs = {p: raw(repo, entries[p][2]).decode("utf-8") for p in paths if p.endswith(".md")}
    require(all("\0" not in d for d in docs.values()), "POLICY_TEXT")
    q, h, i = (docs[ROOT+n] for n in ("QUALIFICATION_ROUTE.md","HISTORICAL_QUAL002.md","INDEX.md"))
    require("APPROVED BY TERRY (GOVERNANCE DIRECTION ONLY)" in q and "尚未配置正式 Run ID" in q
            and "待 Terry 裁決" not in q, "DIRECTION_01")
    require("APPROVED BY TERRY (DISPOSITION DIRECTION ONLY)" in h and "正式 supersession event 尚未物化或接受" in h
            and "production dispatch deny 尚未完成驗收" in h and "待 Terry 裁決" not in h
            and not re.search(r"(?im)^\s*(?:status\s*[:=]\s*)?(?:supersession\s+(?:is\s+)?effective|正式 supersession event 已生效)\s*[.。]?\s*$", h),
            "SUPERSESSION_STATE")
    require(all(mark in i for mark in ("P0 DRAFT_CANDIDATE","P1 REVIEWED_CANDIDATE","P2 MERGE_AUTHORIZED",
            "P3 MERGED_NOT_YET_ACCEPTED","P4 ACCEPTED_G_POLICY","G-POLICY provides no G-AUTH",
            "L1–L3 confer no execution authority")), "ACCEPTANCE_GATE")
    require("G-POLICY MUST" in docs[ROOT+"AUTHORIZATION_BINDING.md"]
            and "G-AUTH" in docs[ROOT+"AUTHORIZATION_BINDING.md"], "AUTHORITY_SEPARATION")

def verify_history(repo, entries):
    for path, pin in PINS.items():
        require(entries.get(path)==("100644","blob",pin), "ORIGINAL_AUTHORIZATION_BINDING" if path==RECORD else "HISTORICAL_CONTROL")
    require(re.search(rb"AUTH_SHA \(SHA-256 of actual final authorization JSON bytes\): [\x60][0-9a-f]{64}[\x60]", raw(repo,PINS[RECORD])),
            "ORIGINAL_AUTHORIZATION_BINDING")
    manifest = strict_json(raw(repo,PINS[MANIFEST]))
    require(manifest["source_baseline_commit"]==M4 and len(manifest["files"])==41, "FROZEN_MANIFEST")
    for row in manifest["files"]:
        entry=entries.get(row["path"])
        require(entry==("100644","blob",row["git_blob"]), "FROZEN_SOURCE_DRIFT")
        data=raw(repo,entry[2])
        require(len(data)==row["byte_length"] and hashlib.sha256(data).hexdigest()==row["sha256"], "FROZEN_BYTES")

def run(repo, base, head, pr):
    for ref in (base, head):
        require(re.fullmatch(r"[0-9a-f]{40}",ref) is not None, "REF_FORMAT")
        require(git(repo,"cat-file","-t",ref).strip()==b"commit", "REF_NOT_COMMIT")
    require(git(repo,"merge-base",ANCHOR,base).decode().strip()==ANCHOR, "HISTORICAL_ANCESTRY")
    require(git(repo,"merge-base",M4,ANCHOR).decode().strip()==M4, "M4_ANCESTRY")
    common=git(repo,"merge-base",base,head).decode().strip()
    old,new=tree(repo,common),tree(repo,head)
    changed=sorted(p for p in old.keys() | new.keys() if old.get(p)!=new.get(p))
    require(changed,"EMPTY_DIFF")
    require(all(not any(ord(c)<32 for c in p) for p in changed),"UNSAFE_PATH")
    require(len({p.casefold() for p in new})==len(new),"CASE_COLLISION")
    for p in changed:
        if p in new:
            require(new[p][:2] in {("100644","blob"),("100755","blob")}, "POLICY_MODE" if p.startswith(ROOT) else "UNSAFE_MODE")
        parts=p.casefold().split("/")
        stem=parts[-1]
        require(not (stem.endswith((".authorization.json",".p7m",".pfx",".p12",".key"))
                     or stem.startswith(("formal-execution-session-","formal-execution.lock"))
                     or "evidence" in parts and "qf0a_impl" in parts), "EXECUTION_ARTIFACT_PATH")
    scoped=pr==30 or any(p.casefold().startswith(ROOT) for p in changed)
    if scoped:
        verify_policy(repo,new)
        verify_history(repo,new)
        verify_history(repo,tree(repo,base))
        require(set(changed)<=FILES,"POLICY_SCOPE")
        if pr==30:
            require(set(changed)==FILES and not FILES.intersection(old),"PR30_EXACT_EIGHT_ADDITIONS")
    diff=subprocess.run(["git","--no-replace-objects","-C",str(repo),"diff","--check",common,head,"--"],
                        capture_output=True)
    require(diff.returncode==0,"DIFF_CHECK")
    identities={p:{"mode":new[p][0],"blob":new[p][2],"sha256":hashlib.sha256(raw(repo,new[p][2])).hexdigest()}
                for p in sorted(FILES) if scoped}
    return {"reason":"POLICY_INTEGRITY" if scoped else "COMMON_INTEGRITY",
            "scope":"R2_POLICY" if scoped else "COMMON_ONLY",
            "policy_acceptance_granted":False,"execution_authority":False,
            "additional_gates":None if scoped else "TECHNICAL_PR_ADDITIONAL_GATES_REQUIRED",
            "base":base,"head":head,"merge_base":common,"changed_count":len(changed),"policy_identities":identities}

def require_live_check(check, sha):
    require(check.get("head_sha")==sha and check.get("name")==JOB
            and check.get("app",{}).get("slug")=="github-actions"
            and check.get("status")=="completed" and check.get("conclusion")=="success","LIVE_CHECK_NOT_SUCCESS")

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--repo",type=Path,required=True)
    cli.add_argument("--base",required=True)
    cli.add_argument("--head",required=True)
    cli.add_argument("--pr",type=int,required=True)
    args=cli.parse_args()
    try:
        detail=run(args.repo,args.base,args.head,args.pr)
    except (Rejected,UnicodeError) as exc:
        print(json.dumps({"result":"FAIL","reason":str(exc) if isinstance(exc,Rejected) else "INVALID_UTF8"}))
        return 1
    print(json.dumps({"result":"PASS",**detail},sort_keys=True))
    return 0

if __name__=="__main__":
    sys.exit(main())
