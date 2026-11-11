"""Run CLI tests against disposable Git objects backed by the real history."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_policy import ANCHOR, M4, ROOT, FILES, CONTRACT, RECORD, JOB, Rejected, require_live_check

def command(repo,*args,data=None):
    env={**os.environ,"GIT_AUTHOR_NAME":"Governance CI fixture","GIT_AUTHOR_EMAIL":"fixture@example.invalid",
         "GIT_COMMITTER_NAME":"Governance CI fixture","GIT_COMMITTER_EMAIL":"fixture@example.invalid"}
    return subprocess.check_output(["git","--no-replace-objects","-C",str(repo),*args],input=data,env=env,
                                   stderr=subprocess.PIPE)

def fixtures(origin,repo):
    command(repo,"init","-q")
    packed=command(origin,"pack-objects","--revs","--stdout",data=(ANCHOR+"\n").encode())
    command(repo,"index-pack","--stdin",data=packed)
    policy={"version":"1.0-candidate","status":"CANDIDATE / NOT ACCEPTED / NOT AUTHORIZED FOR FORMAL EXECUTION",
            "phase":"G-POLICY","formal_dispatch":False,"qual002_state":"STATE_2_EXECUTION_HOLD",
            "qualification_state":"R2_UNALLOCATED_HOLD","historical_anchor":ANCHOR,"historical_m4":M4,
            "decisions":[f"DECISION-R2-{i:02}" for i in range(1,6)],"allowed_paths":sorted(FILES)}
    def serialize(d):
        return (json.dumps(d)+"\n").encode()
    docs={p:("100644",b"# Synthetic policy fixture\n") for p in FILES}
    docs.update({
        CONTRACT:("100644",serialize(policy)),
        ROOT+"QUALIFICATION_ROUTE.md":("100644","APPROVED BY TERRY (GOVERNANCE DIRECTION ONLY)\n尚未配置正式 Run ID\n".encode()),
        ROOT+"HISTORICAL_QUAL002.md":("100644",("APPROVED BY TERRY (DISPOSITION DIRECTION ONLY)\n"
            "正式 supersession event 尚未物化或接受\nproduction dispatch deny 尚未完成驗收\n").encode()),
        ROOT+"INDEX.md":("100644",("P0 DRAFT_CANDIDATE\nP1 REVIEWED_CANDIDATE\nP2 MERGE_AUTHORIZED\n"
            "P3 MERGED_NOT_YET_ACCEPTED\nP4 ACCEPTED_G_POLICY\nG-POLICY provides no G-AUTH\n"
            "L1–L3 confer no execution authority\n").encode()),
        ROOT+"AUTHORIZATION_BINDING.md":("100644",b"G-POLICY MUST maintain a separate G-AUTH.\n"),
    })
    def changed(**fields):
        return {CONTRACT:("100644",serialize({**policy,**fields}))}
    def commit(patches,include=True):
        command(repo,"read-tree",ANCHOR)
        for path,item in ({**(docs if include else {}),**patches}).items():
            if item is None:
                command(repo,"update-index","--force-remove","--",path)
                continue
            mode,value=item
            oid=ANCHOR if mode=="160000" else command(repo,"hash-object","-w","--stdin",data=value).decode().strip()
            command(repo,"update-index","--add","--cacheinfo",mode,oid,path)
        tree=command(repo,"write-tree").decode().strip()
        return command(repo,"commit-tree",tree,"-p",ANCHOR,"-m","Synthetic CI input").decode().strip()
    science="foundation/qf0/qf0a_impl/src/qf0a/metrics.py"
    return [
      ("01_VALID_POLICY",{},True,30,"POLICY_INTEGRITY",0),
      ("02_JSON_SYNTAX",{CONTRACT:("100644",b"{ broken\n")},True,30,"JSON_SYNTAX",1),
      ("03_DUPLICATE_KEY",{CONTRACT:("100644",b'{"phase":"G-POLICY","phase":"G-AUTH"}\n')},True,30,"DUPLICATE_JSON_KEY",1),
      ("04_DISPATCH_TRUE",changed(formal_dispatch=True),True,30,"FORMAL_DISPATCH",1),
      ("05_G_AUTH",changed(phase="G-AUTH"),True,30,"PHASE",1),
      ("06_QUAL002_DRIFT",changed(qual002_state="READY"),True,30,"QUAL002_STATE",1),
      ("07_ID_ALLOCATION",changed(qualification_state="ALLOCATED"),True,30,"R2_STATE",1),
      ("08_M4_DRIFT",changed(historical_m4="0"*40),True,30,"HISTORICAL_M4",1),
      ("09_NINTH_FILE",{ROOT+"EXTRA.md":("100644",b"fixture\n")},True,30,"R2_INVENTORY",1),
      ("10_MISSING_FILE",{ROOT+"MODEL_B.md":None},True,30,"R2_INVENTORY",1),
      ("11_EXTRA_JSON_KEY",changed(extra=True),True,30,"JSON_SCHEMA_KEYS",1),
      ("12A_SYMLINK",{ROOT+"MODEL_B.md":("120000",b"elsewhere")},True,30,"POLICY_MODE",1),
      ("12B_SUBMODULE",{ROOT+"MODEL_B.md":("160000",b"")},True,30,"POLICY_MODE",1),
      ("12C_EXECUTABLE",{ROOT+"MODEL_B.md":("100755",b"fixture\n")},True,30,"POLICY_MODE",1),
      ("13_ACCEPTED_IMPERSONATION",changed(status="ACCEPTED"),True,30,"CANDIDATE_STATE",1),
      ("14_SUPERSESSION_FALSE_CLAIM",{ROOT+"HISTORICAL_QUAL002.md":("100644",docs[ROOT+"HISTORICAL_QUAL002.md"][1]+b"Supersession is effective.\n")},True,30,"SUPERSESSION_STATE",1),
      ("15_FROZEN_CHANGE",{science:("100644",b"# fixture only\n")},True,30,"FROZEN_SOURCE_DRIFT",1),
      ("16_AUTH_REFERENCE_REMOVED",{RECORD:("100644",b"# fixture only\n")},True,30,"ORIGINAL_AUTHORIZATION_BINDING",1),
      ("17_UNRELATED_DOC",{"CI_FIXTURE.md":("100644",b"# Fixture documentation\n")},False,44,"COMMON_INTEGRITY",0),
      ("18_PR30_OUT_OF_SCOPE",{"CI_FIXTURE.md":("100644",b"# Fixture documentation\n")},True,30,"POLICY_SCOPE",1),
      ("21_FORMAL_ARTIFACT",{"synthetic.authorization.json":("100644",b"{}\n")},False,44,"EXECUTION_ARTIFACT_PATH",1),
    ],commit


def binding_tests(repo, candidate):
    import copy
    workflow = ".github/workflows/r2-governance-policy-ci.yml"
    old = ANCHOR
    base = "f9040f13d576c31b405d0b82eba4aa7c34f9f6b8"
    head = "c6eb1a44349ca2774d8bd72327078b4f43d1c874"
    merge = "c9c56818cbe03c9c9afaca87c35669f6f1da6523"
    def git(path, *args, data=None):
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(path), *args],
                              input=data, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return result.stdout
    source = git(repo, "show", f"{candidate}:{workflow}").decode()
    inline = source.split("          python3 -I -B - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    inline = "\n".join(line[10:] for line in inline.splitlines())
    boot = {"__name__": "candidate_bootstrap"}
    exec(compile(inline, workflow, "exec"), boot)
    results = []
    repo_id = boot["REPOSITORY_ID"]
    event = {"number": 30, "action": "reopened",
             "repository": {"id": repo_id, "full_name": boot["REPOSITORY"]},
             "pull_request": {"number": 30, "head": {"sha": head},
                              "base": {"sha": old, "ref": "main", "repo": {"id": repo_id}}}}
    with tempfile.TemporaryDirectory(prefix="r2-binding-tests-", dir=repo.resolve().parent) as tmp:
        work = Path(tmp)
        git(work, "init", "--quiet")
        objects = git(repo, "rev-parse", "--path-format=absolute", "--git-path", "objects").decode().strip()
        (work / ".git/objects/info/alternates").write_bytes((objects + "\n").encode())
        git(work, "update-ref", "--no-deref", "HEAD", merge)
        def commit(tree_source, parents):
            tree = git(work, "rev-parse", tree_source + "^{tree}").decode().strip()
            args = [arg for parent in parents for arg in ("-p", parent)]
            return git(work, "-c", "user.name=CI Fixture", "-c", "user.email=fixture@example.invalid",
                       "commit-tree", tree, *args, data=b"CI fixture\n").decode().strip()
        foreign = commit(old, [M4])
        wrong_first = commit(merge, [old, head])
        wrong_second = commit(merge, [base, old])
        # Missing-test fixture retains the accepted validator blob in an actual Git tree.
        item = git(work, "ls-tree", base, "--", boot["SOURCES"][0]).decode().split()
        child = git(work, "mktree", data=f"100644 blob {item[2]}\tvalidate_policy.py\n".encode()).strip()
        gov = git(work, "mktree", data=b"040000 tree " + child + b"\tgovernance\n").strip()
        root = git(work, "mktree", data=b"040000 tree " + gov + b"\t.github\n").strip()
        no_tests = git(work, "-c", "user.name=CI Fixture", "-c", "user.email=fixture@example.invalid",
                       "commit-tree", root.decode(), "-p", old, data=b"Missing tests\n").decode().strip()
        missing_validator = commit(head, [old])
        contract = json.loads(git(work, "show", f"{head}:{CONTRACT}"))
        contract["formal_dispatch"] = True
        blob = git(work, "hash-object", "-w", "--stdin", data=json.dumps(contract).encode()).decode().strip()
        git(work, "read-tree", head)
        git(work, "update-index", "--cacheinfo", f"100644,{blob},{CONTRACT}")
        tree = git(work, "write-tree").decode().strip()
        invalid_policy = git(work, "-c", "user.name=CI Fixture", "-c", "user.email=fixture@example.invalid",
                             "commit-tree", tree, "-p", old, data=b"Invalid R2 policy\n").decode().strip()
        state = (base, head, old)
        def resolve(checkout=merge, before=state, fetched=None, after=None, changes=None,
                    ref="refs/pull/30/merge", actual=None):
            payload = copy.deepcopy(event)
            if changes:
                changes(payload)
            git(work, "update-ref", "--no-deref", "HEAD", actual or checkout)
            return boot["resolve"](work, payload, ref, checkout, before, fetched or (base, checkout),
                                   after or before)
        def stage(source_base=base, mode="TRUSTED_BASE_VALIDATION"):
            destination = Path(tempfile.mkdtemp(dir=work))
            records = boot["stage_sources"](work, {"mode": mode, "base": source_base}, destination)
            return records, destination
        def validate(head_sha=head):
            binding = {**resolve(), "head": head_sha}
            records, destination = stage()
            return {"sources": records, "verdict": boot["validate"](work, binding, destination)}
        def recorded(test, expected, operation):
            import contextlib
            import io
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                try:
                    value = operation()
                    actual, status = (value.get("reason", "PASS") if isinstance(value, dict) else "PASS"), 0
                except boot["Hold"] as error:
                    value, actual, status = None, str(error), 1
            expected_exit = 0 if test.startswith("POS") else 1
            results.append({"id": test, "expected": expected, "actual": actual, "exit": status,
                            "expected_exit": expected_exit, "detail": value, "log": output.getvalue(),
                            "pass": (actual, status) == (expected, expected_exit)})
        def set_base(value):
            return lambda payload: payload["pull_request"]["base"].update(sha=value)
        def original():
            for rev in (old, head):
                result = subprocess.run(["git", "-C", str(work), "show", f"{rev}:{boot['SOURCES'][0]}"],
                                        capture_output=True)
                boot["require"](result.returncode == 128, "ORIGINAL_FAILURE_NOT_REPRODUCED")
                print(result.stderr.decode(), end="")
            return {"original_exit": 128, "binding": resolve()}
        def split_sources():
            # A head with candidate tests cannot replace the accepted Base test blob.
            records, _ = stage()
            head_blob = git(work, "rev-parse", f"{candidate}:{boot['SOURCES'][1]}").decode().strip()
            boot["require"](records["test_policy.py"]["blob"] == head_blob,
                            "HEAD_TEST_SOURCE_REJECTED")
        recorded("POS-01", "TRUSTED_BASE_VALIDATED", lambda: resolve(changes=set_base(base),
                 before=(base, head, base)))
        recorded("POS-02", "EVENT_BASE_STALE_BUT_VERIFIED", resolve)
        recorded("POS-03", "PASS", original)
        recorded("POS-04", "PASS", lambda: stage()[0])
        recorded("POS-05", "PASS", validate)
        recorded("POS-06", "PASS", lambda: {"source_commit": candidate,
                 "bootstrap_sha256": __import__("hashlib").sha256(inline.encode()).hexdigest()})
        negatives = (
            ("TRUSTED_BASE_PARENT_MISMATCH", lambda: resolve(checkout=wrong_first)),
            ("SYNTHETIC_HEAD_PARENT_MISMATCH", lambda: resolve(checkout=wrong_second)),
            ("SYNTHETIC_MERGE_INVALID", lambda: resolve(checkout=head)),
            ("REPOSITORY_ID_MISMATCH", lambda: resolve(changes=lambda p: p["repository"].update(id=1))),
            ("TRUSTED_BASE_REF_DRIFT", lambda: resolve(fetched=(old, merge))),
            ("TRUSTED_BASE_REF_DRIFT", lambda: resolve(after=(old, head, old))),
            ("PR_HEAD_MISMATCH", lambda: resolve(after=(base, old, old))),
            ("TRUSTED_VALIDATOR_MISSING", lambda: stage(missing_validator)),
            ("TRUSTED_TESTS_MISSING", lambda: stage(no_tests)),
            ("HEAD_TEST_SOURCE_REJECTED", split_sources),
            ("TRUSTED_VALIDATOR_MISSING", lambda: stage(old)),
            ("INVALID_SOURCE_SHA", lambda: resolve(changes=set_base(""))),
            ("PR_NUMBER_OR_REF_MISMATCH", lambda: resolve(ref="refs/pull/31/merge")),
            ("REPOSITORY_MISMATCH", lambda: resolve(changes=lambda p: p["repository"].update(full_name="other/repo"))),
            ("GIT_READ_FAILED", lambda: resolve(changes=set_base("a" * 40), before=(base, head, base))),
            ("SYNTHETIC_MERGE_REF_DRIFT", lambda: resolve(fetched=(base, wrong_first))),
            ("EVENT_BASE_NOT_ANCESTOR", lambda: resolve(changes=set_base(foreign), before=(base, head, base))),
            ("TRUSTED_BASE_REF_DRIFT", lambda: resolve(before=(candidate, head, old), after=(candidate, head, old))),
            ("CANDIDATE_SELFTEST_ONLY", lambda: stage(mode="CANDIDATE_REPAIR_SELFTEST")),
            ("VALIDATOR_REJECTED", lambda: validate(invalid_policy)),
        )
        for index, (reason, operation) in enumerate(negatives, 1):
            recorded(f"NEG-{index:02d}", reason, operation)
    print(json.dumps({"mode": "CANDIDATE_REPAIR_SELFTEST", "source_commit": candidate,
                      "fixture": {"base": base, "event_base": old, "head": head, "merge": merge},
                      "pass": sum(row["pass"] for row in results), "tests": results}, sort_keys=True))
    return 0 if all(row["pass"] for row in results) else 1


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--repo",type=Path,required=True)
    cli.add_argument("--binding-candidate")
    args=cli.parse_args()
    if args.binding_candidate:
        return binding_tests(args.repo, args.binding_candidate)
    results=[]
    script=Path(__file__).with_name("validate_policy.py")
    with tempfile.TemporaryDirectory(prefix="techorb-governance-") as tmp:
        repo=Path(tmp)
        cases,commit=fixtures(args.repo,repo)
        valid=None
        for name,patch,include,pr,reason,exit_code in cases:
            head=commit(patch,include)
            valid=valid or head
            argv=[sys.executable,"-I","-B",str(script),"--repo",str(repo),"--base",ANCHOR,"--head",head,"--pr",str(pr)]
            process=subprocess.run(argv,capture_output=True,encoding="utf-8")
            actual=json.loads(process.stdout)
            match=(actual["reason"],process.returncode)==(reason,exit_code)
            if name=="17_UNRELATED_DOC":
                match=match and actual["policy_acceptance_granted"] is False and actual["execution_authority"] is False and actual["additional_gates"]=="TECHNICAL_PR_ADDITIONAL_GATES_REQUIRED"
            results.append({"id":name,"expected":reason,"expected_exit":exit_code,"actual":actual,
                            "exit":process.returncode,"stdout":process.stdout,"stderr":process.stderr,"pass":match})
        skip=subprocess.run([sys.executable,"-I","-B",str(script),"--repo",str(repo),"--base",ANCHOR,
                            "--head",valid,"--pr","30","--skip-tests"],capture_output=True,encoding="utf-8")
        results.append({"id":"19_NO_SKIP_FLAG","expected":"ARGUMENT_REJECTED","expected_exit":2,
                        "exit":skip.returncode,"stdout":skip.stdout,"stderr":skip.stderr,
                        "actual":"ARGUMENT_REJECTED" if skip.returncode==2 and not skip.stdout else "UNEXPECTED",
                        "pass":skip.returncode==2 and not skip.stdout})
        check={"head_sha":valid,"name":JOB,"app":{"slug":"github-actions"},"status":"in_progress","conclusion":None}
        for name,fields in (("20_PENDING_CHECK",{}),("22_STALE_CHECK",{"head_sha":"0"*40,"status":"completed","conclusion":"success"}),
                            ("23_WRONG_APP",{"app":{"slug":"other"},"status":"completed","conclusion":"success"})):
            try:
                require_live_check({**check,**fields},valid)
                actual,code="UNEXPECTED",0
            except Rejected as error:
                actual,code=str(error),1
            results.append({"id":name,"expected":"LIVE_CHECK_NOT_SUCCESS","expected_exit":1,"actual":actual,"exit":code,
                            "fixture":{**check,**fields},"pass":actual=="LIVE_CHECK_NOT_SUCCESS" and code==1})
    print(json.dumps({"tests":results,"pass":sum(row["pass"] for row in results),"fail":sum(not row["pass"] for row in results)},
                     ensure_ascii=True,sort_keys=True))
    return 0 if all(row["pass"] for row in results) else 1

if __name__=="__main__":
    sys.exit(main())
