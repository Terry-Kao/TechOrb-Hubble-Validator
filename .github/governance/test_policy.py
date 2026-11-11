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

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--repo",type=Path,required=True)
    args=cli.parse_args()
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
