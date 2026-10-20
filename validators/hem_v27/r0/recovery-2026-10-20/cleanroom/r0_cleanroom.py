#!/usr/bin/env python3
"""Provenance-preserving HEM v26.4 clean-room rerun.

This intentionally reproduces the frozen runner's numerical algorithm while
retaining every generated INI, CLASS stdout/stderr stream, and raw CLASS file.
It does not modify the frozen tolerance contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/work")
UPSTREAM = ROOT / "source_upstream"
PATCHED = ROOT / "source_patched"
OUT = ROOT / "outputs"
HISTORICAL = ROOT / "frozen" / "hem_class_v26_4_exact_lock_results.csv"
EXPECTED_COMMIT = "aa92943e4ab86b56970953589b4897adf2bd0f99"

COSMO = {
    "h": 0.674,
    "omega_b": 0.022383,
    "omega_cdm": 0.11933,
    "n_s": 0.965,
    "A_s": 2.1e-9,
}

CANDIDATES = [
    ("Mode1_b1.40", "Mode1_Standard_Clumping", 0.0, 1.40),
    ("Mode2_b1.35", "Mode2_LyAlpha_Enhanced", 0.5, 1.35),
    ("Mode2_b1.40", "Mode2_LyAlpha_Enhanced", 0.5, 1.40),
    ("Mode1_b1.60", "Mode1_Standard_Clumping", 0.0, 1.60),
]

TOLS = {
    "z_star": 5e-5,
    "rs_Mpc": 5e-5,
    "rs_reduction_pct": 5e-5,
    "H0_BAO_naive": 5e-4,
    "H0_CMB_exact": 5e-4,
    "theta100_error_ppm": 5e-2,
    "ns_compensated": 5e-6,
    "As_compensated_1e9": 5e-5,
    "rms_residual_naive_pct": 5e-4,
    "rms_residual_theta_only_pct": 5e-4,
    "rms_residual_final_pct": 5e-4,
    "max_residual_final_pct": 5e-4,
    "est_B_nG": 5e-12,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(cmd: list[str], cwd: Path, log_name: str) -> subprocess.CompletedProcess[str]:
    log_dir = OUT / "commands"
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    (log_dir / f"{log_name}.command.json").write_text(
        json.dumps({"argv": cmd, "cwd": str(cwd), "exit_code": proc.returncode}, indent=2) + "\n",
        encoding="utf-8",
    )
    (log_dir / f"{log_name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{log_name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def build(path: Path, label: str) -> None:
    if git_head(path) != EXPECTED_COMMIT:
        raise RuntimeError(f"{label} checkout identity mismatch")
    command(["make", "clean"], path, f"build_{label}_clean")
    command(["make", "class", "-j4"], path, f"build_{label}")
    binary = path / "class"
    if not binary.is_file():
        raise RuntimeError(f"{label} CLASS binary missing")


def write_ini(run_dir: Path, root_prefix: str, h: float, n_s: float, A_s: float,
              explicit_neutrino: bool = False) -> Path:
    ini = run_dir / "input.ini"
    lines = [
        "output = tCl, pCl",
        "l_max_scalars = 2500",
        f"h = {h:.8f}",
        f"omega_b = {COSMO['omega_b']:.8f}",
        f"omega_cdm = {COSMO['omega_cdm']:.8f}",
        f"n_s = {n_s:.8f}",
        f"A_s = {A_s:.10e}",
        "recombination = RECFAST",
        f"root = {root_prefix}",
        "write background = yes",
        "write thermodynamics = yes",
        "input_verbose = 1",
        "background_verbose = 2",
        "thermodynamics_verbose = 2",
    ]
    if explicit_neutrino:
        lines += [
            "N_ur = 2.046",
            "N_ncdm = 1",
            "m_ncdm = 0.06",
            "T_ncdm = 0.7137658555036082",
        ]
    ini.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ini


def run_class(source: Path, track: str, tag: str, b: float = 1.0, eta: float = 0.0,
              h: float = COSMO["h"], n_s: float = COSMO["n_s"],
              A_s: float = COSMO["A_s"], explicit_neutrino: bool = False) -> dict:
    run_dir = OUT / "runs" / track / tag
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    root_prefix = str(run_dir / "class_")
    ini = write_ini(run_dir, root_prefix, h, n_s, A_s, explicit_neutrino)
    env = os.environ.copy()
    env["HIA_B_CLUMPING"] = str(b)
    env["HIA_LYA_EXP"] = str(eta)
    env_record = {
        "HIA_B_CLUMPING": env["HIA_B_CLUMPING"],
        "HIA_LYA_EXP": env["HIA_LYA_EXP"],
        "source": str(source),
        "source_commit": git_head(source),
        "binary_sha256": sha256(source / "class"),
        "ini_sha256": sha256(ini),
        "explicit_neutrino": explicit_neutrino,
    }
    (run_dir / "environment.json").write_text(json.dumps(env_record, indent=2) + "\n", encoding="utf-8")
    proc = subprocess.run([str(source / "class"), str(ini)], cwd=ROOT, env=env, text=True, capture_output=True)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(f"{proc.returncode}\n", encoding="ascii")
    if proc.returncode != 0:
        raise RuntimeError(f"CLASS failed for {track}/{tag}")

    z_match = re.search(r"recombination(?:[^=\n]*)\s+at\s+z\s*=\s*([0-9.]+)", proc.stdout)
    rs_match = re.search(r"with\s+comoving\s+sound\s+horizon\s*=\s*([0-9.]+)\s*Mpc", proc.stdout)
    if not z_match or not rs_match:
        raise RuntimeError(f"failed to parse z_star/r_s for {track}/{tag}")
    z_star = float(z_match.group(1))
    r_s = float(rs_match.group(1))
    # CLASS v3.2.0 prefixes the first output realization with ``00_``.
    bg_file = run_dir / "class_00_background.dat"
    cl_file = run_dir / "class_00_cl.dat"
    thermo_file = run_dir / "class_00_thermodynamics.dat"
    for required in (bg_file, cl_file, thermo_file):
        if not required.is_file():
            raise RuntimeError(f"missing raw output {required}")
    bg = np.loadtxt(bg_file, comments="#")
    cls = np.loadtxt(cl_file, comments="#")
    da = np.interp(z_star, bg[:, 0][::-1], bg[:, 4][::-1])
    theta100 = 100.0 * r_s / da
    result = {
        "z_star": z_star,
        "rs_Mpc": r_s,
        "theta100": theta100,
        "ell": cls[:, 0],
        "tt": cls[:, 1],
        "run_dir": str(run_dir),
    }
    (run_dir / "parsed.json").write_text(
        json.dumps({k: v for k, v in result.items() if k not in ("ell", "tt")}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def compare_control(up: dict, patched: dict) -> dict:
    if not np.array_equal(up["ell"], patched["ell"]):
        return {"pass": False, "reason": "multipole grid mismatch"}
    scale = np.maximum(np.abs(up["tt"]), 1e-30)
    rel = np.abs(patched["tt"] - up["tt"]) / scale
    report = {
        "z_star_abs": abs(patched["z_star"] - up["z_star"]),
        "rs_Mpc_abs": abs(patched["rs_Mpc"] - up["rs_Mpc"]),
        "theta100_abs": abs(patched["theta100"] - up["theta100"]),
        "tt_max_relative": float(np.max(rel)),
        "tt_rms_relative": float(np.sqrt(np.mean(rel ** 2))),
        "rows": int(len(rel)),
    }
    report["pass"] = bool(
        report["z_star_abs"] <= 5e-6
        and report["rs_Mpc_abs"] <= 5e-6
        and report["theta100_abs"] <= 1e-8
        and report["tt_max_relative"] <= 1e-7
        and report["tt_rms_relative"] <= 1e-8
    )
    return report


def run_candidates(base: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (base["ell"] >= 30) & (base["ell"] <= 2500)
    ell_fit = base["ell"][mask]
    cl_base = base["tt"][mask]
    rows: list[dict] = []
    curves = pd.DataFrame({"ell": ell_fit})

    for case_id, mode, eta, b in CANDIDATES:
        fixed = run_class(PATCHED, "historical", f"{case_id}_fixedh", b, eta)
        h_naive = COSMO["h"] * (base["rs_Mpc"] / fixed["rs_Mpc"])
        naive = run_class(PATCHED, "historical", f"{case_id}_naive", b, eta, h=h_naive)
        res_naive = (naive["tt"][mask] - cl_base) / cl_base * 100.0
        rms_naive = float(np.sqrt(np.mean(res_naive ** 2)))

        h_prev, h_curr = 0.700, 0.760
        prev = run_class(PATCHED, "historical", f"{case_id}_iter0", b, eta, h=h_prev)
        curr = run_class(PATCHED, "historical", f"{case_id}_iter1", b, eta, h=h_curr)
        for iteration in range(5):
            if abs(curr["theta100"] - base["theta100"]) < 1e-6:
                break
            denom = curr["theta100"] - prev["theta100"]
            if denom == 0:
                raise RuntimeError(f"secant denominator zero for {case_id}")
            h_next = float(np.clip(h_curr + (base["theta100"] - curr["theta100"]) *
                                   (h_curr - h_prev) / denom, 0.65, 0.88))
            h_prev, prev, h_curr = h_curr, curr, h_next
            curr = run_class(PATCHED, "historical", f"{case_id}_iter{iteration + 2}", b, eta, h=h_curr)

        locked = run_class(PATCHED, "historical", f"{case_id}_locked", b, eta, h=h_curr)
        res_locked = (locked["tt"][mask] - cl_base) / cl_base * 100.0
        rms_locked = float(np.sqrt(np.mean(res_locked ** 2)))

        y = np.log(cl_base / locked["tt"][mask])
        x = np.log(ell_fit / 500.0)
        coeff, _, _, _ = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)
        ln_as_ratio, delta_ns = coeff
        ns_opt = COSMO["n_s"] + delta_ns
        as_opt = COSMO["A_s"] * np.exp(ln_as_ratio) * ((700.0 / 500.0) ** (-delta_ns))
        opt1 = run_class(PATCHED, "historical", f"{case_id}_opt1", b, eta, h=h_curr,
                         n_s=ns_opt, A_s=as_opt)
        as_final = as_opt * float(np.mean(cl_base / opt1["tt"][mask]))
        final = run_class(PATCHED, "historical", f"{case_id}_final", b, eta, h=h_curr,
                          n_s=ns_opt, A_s=as_final)
        res_final = (final["tt"][mask] - cl_base) / cl_base * 100.0
        row = {
            "case_id": case_id,
            "mode": mode,
            "eta_lya": eta,
            "b": b,
            "est_B_nG": 0.45 * math.sqrt(max(0.0, b - 1.0)),
            "z_star": final["z_star"],
            "rs_Mpc": final["rs_Mpc"],
            "rs_reduction_pct": (base["rs_Mpc"] - final["rs_Mpc"]) / base["rs_Mpc"] * 100.0,
            "H0_BAO_naive": h_naive * 100.0,
            "H0_CMB_exact": h_curr * 100.0,
            "theta100_error_ppm": abs(final["theta100"] - base["theta100"]) / base["theta100"] * 1e6,
            "ns_compensated": ns_opt,
            "As_compensated_1e9": as_final * 1e9,
            "rms_residual_naive_pct": rms_naive,
            "rms_residual_theta_only_pct": rms_locked,
            "rms_residual_final_pct": float(np.sqrt(np.mean(res_final ** 2))),
            "max_residual_final_pct": float(np.max(np.abs(res_final))),
        }
        rows.append(row)
        curves[f"{case_id}_naive_pct"] = res_naive
        curves[f"{case_id}_theta_only_pct"] = res_locked
        curves[f"{case_id}_final_pct"] = res_final
        print(f"completed {case_id}", flush=True)
    return pd.DataFrame(rows), curves


def comparison(actual: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, exp in expected.iterrows():
        act = actual.loc[actual["case_id"] == exp["case_id"]]
        if len(act) != 1:
            records.append({"case_id": exp["case_id"], "field": "identity", "status": "FAIL"})
            continue
        act = act.iloc[0]
        for field, tol in TOLS.items():
            delta = abs(float(act[field]) - float(exp[field]))
            records.append({
                "case_id": exp["case_id"], "field": field,
                "historical": exp[field], "clean_room": act[field],
                "absolute_delta": delta, "tolerance": tol,
                "status": "PASS" if delta <= tol else "FAIL",
            })
    return pd.DataFrame(records)


def manifest() -> None:
    entries = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUT / "SHA256_MANIFEST.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    env = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "upstream_commit": git_head(UPSTREAM),
        "patched_commit": git_head(PATCHED),
    }
    for name, cmd in {
        "gcc": ["gcc", "--version"], "make": ["make", "--version"],
        "git": ["git", "--version"], "pip": ["python3", "-m", "pip", "freeze"],
    }.items():
        p = subprocess.run(cmd, text=True, capture_output=True)
        env[name] = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    (OUT / "ENVIRONMENT_MANIFEST.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")

    build(UPSTREAM, "upstream")
    build(PATCHED, "patched")
    up = run_class(UPSTREAM, "control", "unmodified_baseline")
    patched = run_class(PATCHED, "control", "patched_b1_eta0")
    control = compare_control(up, patched)
    (OUT / "UNMODIFIED_CONTROL.json").write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")
    if not control["pass"]:
        manifest()
        raise RuntimeError("unmodified-versus-patched control failed")

    # Separate explicit-neutrino comparison, with no historical PASS target.
    run_class(PATCHED, "standard_comparison", "explicit_sum_mnu_0p06", explicit_neutrino=True)
    actual, curves = run_candidates(patched)
    derived = OUT / "derived"
    derived.mkdir()
    actual.to_csv(derived / "clean_room_results.csv", index=False)
    curves.to_csv(derived / "clean_room_residual_curves.csv", index=False)
    expected = pd.read_csv(HISTORICAL)
    comp = comparison(actual, expected)
    comp.to_csv(derived / "tolerance_comparison.csv", index=False)
    verdict = "PASS" if (comp["status"] == "PASS").all() else "FAIL"
    (OUT / "R0_NUMERICAL_VERDICT.json").write_text(
        json.dumps({"verdict": verdict, "comparison_rows": len(comp),
                    "pass_rows": int((comp["status"] == "PASS").sum()),
                    "fail_rows": int((comp["status"] == "FAIL").sum())}, indent=2) + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for case_id, _, _, _ in CANDIDATES:
        ax.plot(curves["ell"], curves[f"{case_id}_final_pct"], label=case_id)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Multipole ell", ylabel="Final TT residual (%)", xlim=(30, 2500))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(derived / "clean_room_final_residuals.png", dpi=200)
    plt.close(fig)
    manifest()
    print(json.dumps({"numerical_verdict": verdict, "control": control}, indent=2))


if __name__ == "__main__":
    main()
