#!/usr/bin/env python3
"""HEM v27 R2 -- repeatability check, observable extraction and frozen pointwise adjudication.

Frozen rules applied (TOLERANCE_CONTRACT.md / BENCHMARK_SPEC.md):
  * reference z points fixed; model evaluated at reference z by linear interpolation in z
    (numpy.interp on the native CLASS grid); NO extrapolation (out-of-range -> BLOCKED row).
  * Fig.2 x_e linear |delta| <= 0.012 ; Fig.2 g linear |delta| <= 0.00025 Mpc^-1
  * Fig.3 x_e log10 |delta| <= 0.045 dex
  * only frozen `admitted` mask used; excluded rows preserved with frozen exclusion_reason.
No shift / warp / renormalisation / smoothing / fitting is applied.
"""
import csv, hashlib, json, math, os, sys
import numpy as np

R2 = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(R2, *p)

TABLES = ["background.dat", "thermodynamics.dat", "cl.dat", "cl_lensed.dat",
          "pk.dat", "pk_cb.dat", "pk_cb_nl.dat", "pk_nl.dat"]
TOL = {("Fig2", "xe"): 0.012, ("Fig2", "g"): 0.00025, ("Fig3", "xe"): 0.045}
REF_SHA = {"FIG2_REFERENCE.csv": "a079416fdf59e040233f4fecc67f5d8f920ff038a654c8c7758a4c7aab051a05",
           "FIG3_REFERENCE.csv": "10a160e50d5f67b11d678ce7beb709b2c782846eb2f8fff9f39177782c32c719"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return np.loadtxt(p, comments="#")


# ---------------------------------------------------------------- repeatability
def compare_tables(pa, pb):
    ba, bb = open(pa, "rb").read(), open(pb, "rb").read()
    rec = {"file_a": os.path.relpath(pa, R2), "file_b": os.path.relpath(pb, R2),
           "sha256_a": hashlib.sha256(ba).hexdigest(), "sha256_b": hashlib.sha256(bb).hexdigest(),
           "bytes_a": len(ba), "bytes_b": len(bb), "byte_identical": ba == bb}
    A, B = load(pa), load(pb)
    if A.shape != B.shape:
        rec.update(shape_a=list(A.shape), shape_b=list(B.shape), max_abs_delta=None,
                   max_rel_delta=None, rms_delta=None, first_mismatch="SHAPE MISMATCH")
        return rec
    d = np.abs(A - B)
    den = np.maximum(np.abs(A), np.abs(B))
    rel = np.where(den > 0, d / np.where(den > 0, den, 1), 0.0)
    nz = np.argwhere(d > 0)
    rec.update(shape=list(A.shape), max_abs_delta=float(d.max()), max_rel_delta=float(rel.max()),
               rms_delta=float(math.sqrt(float(np.mean(d ** 2)))),
               first_mismatch="NONE" if len(nz) == 0 else f"row {int(nz[0][0])} col {int(nz[0][1])}")
    return rec


pairs = []
for tag, da, pa_, db, pb_ in [
        ("standard_run1_vs_run2", "05_standard_control_run1", "standard_run1_", "06_standard_control_run2", "standard_run2_"),
        ("m3_run1_vs_run2", "07_m3_run1", "m3_run1_", "08_m3_run2", "m3_run2_"),
        ("m3_run1_vs_readonly_diagnostic_build", "07_m3_run1", "m3_run1_", "03_build/diagnostic_zone_dump/out", "diag_m3_")]:
    for t in TABLES:
        r = compare_tables(J(da, pa_ + t), J(db, pb_ + t))
        r["pair"] = tag
        r["table"] = t
        pairs.append(r)

os.makedirs(J("12_manifests"), exist_ok=True)
with open(J("12_manifests", "REPEATABILITY_DATA.json"), "w", newline="\n") as f:
    json.dump(pairs, f, indent=2)

# ---------------------------------------------------------------- observables
def thermo(p):
    T = load(p)
    o = np.argsort(T[:, 0])
    return T[o, 0], T[o, 2], T[o, 5]   # z, x_e, g [Mpc^-1]

z_s, xe_s, g_s = thermo(J("05_standard_control_run1", "standard_run1_thermodynamics.dat"))
z_m, xe_m, g_m = thermo(J("07_m3_run1", "m3_run1_thermodynamics.dat"))
Z = load(J("03_build", "diagnostic_zone_dump", "out", "zone_histories_diagnostic.dat"))
Z = Z[np.argsort(Z[:, 0])]

# curve -> (z grid, values, extraction track, source)
UNMOD = "UNMODIFIED_ADJUDICATIVE"
DIAG = "READONLY_DIAGNOSTIC_DUMP_NON_PREREGISTERED"
src = {
    ("Fig2", "xe_standard"): (z_s, xe_s, UNMOD, "standard_run1_thermodynamics.dat col3 x_e"),
    ("Fig2", "xe_m3"): (z_m, xe_m, UNMOD, "m3_run1_thermodynamics.dat col3 x_e"),
    ("Fig2", "g_standard"): (z_s, g_s, UNMOD, "standard_run1_thermodynamics.dat col6 g"),
    ("Fig2", "g_m3"): (z_m, g_m, UNMOD, "m3_run1_thermodynamics.dat col6 g"),
    ("Fig3", "M3"): (z_m, xe_m, UNMOD, "m3_run1_thermodynamics.dat col3 x_e"),
    ("Fig3", "zero_zone_LCDM"): (z_s, xe_s, UNMOD, "standard_run1_thermodynamics.dat col3 x_e (recfast, Delta=1)"),
    ("Fig3", "minus_zone"): (Z[:, 0], Z[:, 2], DIAG, "zone_histories_diagnostic.dat reco1 (Delta=0.1) freed in pristine code"),
    ("Fig3", "plus_zone"): (Z[:, 0], Z[:, 4], DIAG, "zone_histories_diagnostic.dat reco3 (Delta=8/3) freed in pristine code"),
    ("Fig3", "simple_clumping"): (Z[:, 0], Z[:, 5], DIAG, "zone_histories_diagnostic.dat extra RECFAST run Delta=1+b=2 (not an .ini option)"),
}
# cross-checks of diagnostic M3 / zero-zone columns against unmodified outputs
_w = (Z[:, 0] >= 200) & (Z[:, 0] <= 1700)  # thermodynamics.dat includes reionization at low z; recombination table does not
xcheck = {
    "note": "full-range values differ at z<~30 only because thermodynamics.dat adds reionization; window 200<=z<=1700 is the Fig.3 domain check",
    "diag_xe_m3_vs_unmod_m3_max_abs_fullrange": float(np.max(np.abs(np.interp(Z[:, 0], z_m, xe_m) - Z[:, 1]))),
    "diag_xe_zero_vs_unmod_standard_max_abs_fullrange": float(np.max(np.abs(np.interp(Z[:, 0], z_s, xe_s) - Z[:, 3]))),
    "diag_xe_m3_vs_unmod_m3_max_abs_z200_1700": float(np.max(np.abs(np.interp(Z[_w, 0], z_m, xe_m) - Z[_w, 1]))),
    "diag_xe_zero_vs_unmod_standard_max_abs_z200_1700": float(np.max(np.abs(np.interp(Z[_w, 0], z_s, xe_s) - Z[_w, 3]))),
    "diag_xe_m3_vs_unmod_m3_max_rel_z200_1700": float(np.max(np.abs(np.interp(Z[_w, 0], z_m, xe_m) / Z[_w, 1] - 1))),
}

os.makedirs(J("09_raw_observables"), exist_ok=True)
with open(J("09_raw_observables", "fig2_observables_unmodified.csv"), "w", newline="\n") as f:
    w = csv.writer(f)
    w.writerow(["z", "xe_standard", "g_standard_Mpc^-1", "xe_m3", "g_m3_Mpc^-1"])
    zz = np.arange(250, 1601, 0.5)
    for z in zz:
        w.writerow([f"{z:.1f}", f"{np.interp(z, z_s, xe_s):.12e}", f"{np.interp(z, z_s, g_s):.12e}",
                    f"{np.interp(z, z_m, xe_m):.12e}", f"{np.interp(z, z_m, g_m):.12e}"])
with open(J("09_raw_observables", "fig3_zone_histories.csv"), "w", newline="\n") as f:
    w = csv.writer(f)
    w.writerow(["z", "M3_unmodified", "zero_zone_LCDM_unmodified", "M3_diag", "minus_zone_diag",
                "zero_zone_diag", "plus_zone_diag", "simple_clumping_diag"])
    sel = (Z[:, 0] >= 200) & (Z[:, 0] <= 1700)
    for row in Z[sel]:
        z = row[0]
        w.writerow([f"{z:.6f}", f"{np.interp(z, z_m, xe_m):.12e}", f"{np.interp(z, z_s, xe_s):.12e}"] +
                   [f"{v:.12e}" for v in row[1:]])

# ---------------------------------------------------------------- adjudication
ref_hash = {k: sha(J("00_governance_snapshot", k)) for k in REF_SHA}
for k, v in REF_SHA.items():
    if ref_hash[k] != v:
        raise SystemExit(f"STOP: reference hash mismatch {k}")

rows_out = {"Fig2": [], "Fig3": []}
for fig in ["Fig2", "Fig3"]:
    with open(J("00_governance_snapshot", f"{fig.upper()}_REFERENCE.csv"), newline="") as f:
        for r in csv.DictReader(f):
            key = (r["figure"], r["curve"])
            zg, val, track, source = src[key]
            z = float(r["z"])
            obs_key = "g" if r["observable"].startswith("g") else r["observable"]  # frozen label 'g_Mpc^-1'
            tol = TOL[(r["figure"], obs_key)]
            in_range = zg[0] <= z <= zg[-1]
            node_dist = float(np.min(np.abs(zg - z)))
            rep = float(np.interp(z, zg, val)) if in_range else float("nan")
            refv = float(r["reference_value"])
            o = {"figure": r["figure"], "curve": r["curve"], "observable": r["observable"], "z": r["z"],
                 "reference_value": r["reference_value"], "reproduced_value": f"{rep:.12e}"}
            if fig == "Fig3":
                rl = float(r["reference_log10"]) if r["reference_log10"] else math.log10(refv)
                pl = math.log10(rep) if rep > 0 else float("nan")
                d = pl - rl
                o.update(difference=f"{rep - refv:.12e}", absolute_difference=f"{abs(rep - refv):.12e}",
                         reference_log10=f"{rl:.12f}", reproduced_log10=f"{pl:.12f}", delta_dex=f"{d:.12e}")
                cmp_ = abs(d)
            else:
                d = rep - refv
                o.update(difference=f"{d:.12e}", absolute_difference=f"{abs(d):.12e}")
                cmp_ = abs(d)
            admitted = r["admitted"].strip().upper() == "YES"
            if not admitted:
                st = "EXCLUDED"
            elif not in_range or not math.isfinite(cmp_):
                st = "BLOCKED"
            else:
                st = "PASS" if cmp_ <= tol else "FAIL"
            o.update(tolerance=f"{tol}", comparison_space="log10" if fig == "Fig3" else "linear",
                     admitted=r["admitted"], exclusion_reason=r["exclusion_reason"], status=st,
                     extraction_track=track, adjudicative_eligible="YES" if track == UNMOD else "NO",
                     source=source, nearest_grid_node_dz=f"{node_dist:.3e}",
                     interpolation="numpy.interp linear-in-z, no extrapolation", _cmp=cmp_)
            rows_out[fig].append(o)

cols2 = ["figure", "curve", "observable", "z", "reference_value", "reproduced_value", "difference",
         "absolute_difference", "tolerance", "comparison_space", "admitted", "exclusion_reason", "status",
         "extraction_track", "adjudicative_eligible", "source", "nearest_grid_node_dz", "interpolation"]
cols3 = cols2[:8] + ["reference_log10", "reproduced_log10", "delta_dex"] + cols2[8:]
os.makedirs(J("10_reference_comparison"), exist_ok=True)
for fig, cols in [("Fig2", cols2), ("Fig3", cols3)]:
    with open(J("10_reference_comparison", f"{fig.upper()}_POINTWISE_COMPARISON.csv"), "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for o in rows_out[fig]:
            w.writerow(o)

summary = []
for fig in ["Fig2", "Fig3"]:
    for curve in dict.fromkeys(o["curve"] for o in rows_out[fig]):
        rs = [o for o in rows_out[fig] if o["curve"] == curve]
        ad = [o for o in rs if o["status"] != "EXCLUDED"]
        npass = sum(o["status"] == "PASS" for o in ad)
        nfail = sum(o["status"] == "FAIL" for o in ad)
        nblk = sum(o["status"] == "BLOCKED" for o in ad)
        a = np.array([o["_cmp"] for o in ad])
        imax = int(np.argmax(a))
        track = rs[0]["extraction_track"]
        if nblk:
            v = "BLOCKED"
        elif nfail:
            v = "FAIL"
        else:
            v = "PASS"
        if track != UNMOD:
            v = f"{v}_DIAGNOSTIC_ONLY_NOT_ADJUDICATIVE"
        summary.append({"figure": fig, "curve": curve, "observable": rs[0]["observable"],
                        "comparison_space": rs[0]["comparison_space"], "N_total": len(rs),
                        "N_admitted": len(ad), "N_pass": npass, "N_fail": nfail, "N_blocked": nblk,
                        "max_abs_delta": f"{a.max():.6e}", "z_of_max_delta": ad[imax]["z"],
                        "mean_abs_delta": f"{a.mean():.6e}", "rms_delta": f"{math.sqrt(float(np.mean(a**2))):.6e}",
                        "tolerance": rs[0]["tolerance"], "margin_ratio_max_over_tol": f"{a.max()/float(rs[0]['tolerance']):.4f}",
                        "extraction_track": track, "curve_verdict": v})
with open(J("10_reference_comparison", "R2_COMPARISON_SUMMARY.csv"), "w", newline="\n") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

with open(J("10_reference_comparison", "COMPARISON_RUN_META.json"), "w", newline="\n") as f:
    json.dump({"reference_sha256": ref_hash, "diagnostic_crosschecks": xcheck,
               "numpy": np.__version__, "python": sys.version,
               "grid_dz_standard": float(np.median(np.diff(z_s[(z_s > 200) & (z_s < 1700)]))),
               "summary": summary}, f, indent=2)

# ---------------------------------------------------------------- figures (diagnostic only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs(J("11_figures"), exist_ok=True)
for fig in ["Fig2", "Fig3"]:
    curves = list(dict.fromkeys(o["curve"] for o in rows_out[fig]))
    if fig == "Fig2":
        f1, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for c in curves:
            ax = axs[0] if c.startswith("xe") else axs[1]
            rs = rows_out[fig]
            zg, val, _, _ = src[(fig, c)]
            m = (zg >= 780) & (zg <= 1620)
            l, = ax.plot(zg[m], val[m], lw=1.2, label=f"{c} reproduction")
            pts = [o for o in rs if o["curve"] == c]
            ax.plot([float(o["z"]) for o in pts if o["status"] != "EXCLUDED"],
                    [float(o["reference_value"]) for o in pts if o["status"] != "EXCLUDED"], "o", ms=2.5,
                    color=l.get_color(), mfc="none", label=f"{c} reference (admitted)")
        axs[0].set_ylabel("x_e"); axs[1].set_ylabel("g [Mpc^-1]")
    else:
        f1, ax = plt.subplots(figsize=(8, 5.5)); axs = [ax]
        for c in curves:
            zg, val, tr, _ = src[(fig, c)]
            m = (zg >= 240) & (zg <= 1510)
            l, = ax.semilogy(zg[m], val[m], lw=1.2, ls="-" if tr == UNMOD else "--", label=f"{c} reproduction{'' if tr == UNMOD else ' (diag)'}")
            pts = [o for o in rows_out[fig] if o["curve"] == c and o["status"] != "EXCLUDED"]
            ax.plot([float(o["z"]) for o in pts], [float(o["reference_value"]) for o in pts], "o", ms=2.5, color=l.get_color(), mfc="none")
        ax.set_ylabel("x_e")
    for ax in axs:
        ax.set_xlabel("z"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    f1.suptitle(f"R2 {fig}: published reference vs reproduction (DIAGNOSTIC ONLY; gate = CSV pointwise contract)", fontsize=9)
    f1.tight_layout(); f1.savefig(J("11_figures", f"{fig}_reference_vs_reproduction.png"), dpi=150); plt.close(f1)

    f2, ax = plt.subplots(figsize=(9, 5))
    for c in curves:
        pts = [o for o in rows_out[fig] if o["curve"] == c and o["status"] != "EXCLUDED"]
        tol = float(pts[0]["tolerance"])
        key = "delta_dex" if fig == "Fig3" else "difference"
        ax.plot([float(o["z"]) for o in pts], [float(o[key]) / tol for o in pts], ".-", ms=3, lw=.8,
                label=f"{c} ({pts[0]['extraction_track'][:10]})")
    ax.axhline(1, color="k", ls="--", lw=.8); ax.axhline(-1, color="k", ls="--", lw=.8)
    ax.set_ylim(-1.2, 1.2); ax.set_xlabel("z"); ax.set_ylabel("delta / frozen tolerance")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title(f"R2 {fig} residuals normalised by frozen tolerance (DIAGNOSTIC ONLY)", fontsize=9)
    f2.tight_layout(); f2.savefig(J("11_figures", f"{fig}_residuals.png"), dpi=150); plt.close(f2)

for s in summary:
    print(f"{s['figure']},{s['curve']},{s['N_admitted']},{s['N_pass']},{s['N_fail']},{s['N_blocked']},{s['max_abs_delta']},{s['z_of_max_delta']},{s['tolerance']},{s['curve_verdict']}")
print(json.dumps(xcheck))
for p in pairs:
    print(p["pair"], p["table"], p["byte_identical"], p["max_abs_delta"], p["first_mismatch"])
