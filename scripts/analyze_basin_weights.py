#!/usr/bin/env python3
"""Basin weighting for every basin protocol: unweighted, MD population, 298 K
Boltzmann and scatter-scaled weights, plus two population x scatter hybrids.

For each basin scheme and energy model, reads the existing per-basin interaction
energies (xTB JSON, cuby4 PM6/PM7 logs, MM/GBSA outputs), combines them within
each trajectory under each weight, and reports pool-5 R2, rho, replica spread and
the effective number of basins, 1 / sum(p_i^2). Writes basin_weights.json
and basin_weights.csv.
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from protocol_names import protocol_display_name

from paths import DATA, WORK

REPO = WORK
SCR = REPO / "screening_ff_md_sqm"
ART = REPO / "md"
REPS = ["r1", "r2", "r3", "r4", "r5"]
KT = 0.00198720425864083 * 298.15
E_RE = re.compile(r"Interaction energy:\s*(-?\d+\.\d+)")
DG_RE = re.compile(r"DELTA TOTAL\s+([-\d.]+)")
exp = pd.read_csv(DATA / "experimental_data.csv").set_index("PDB_ID")["Exp_dG"].astype(float)
SCHEMES = ("aa", "bb", "iface_sc", "contacts")
XTB = ("gfn2_gbsa", "gfn2_alpb", "gfn2_cosmo", "gfn1_gbsa", "gfn1_alpb", "gfn1_cosmo")

def snap_dir(tag):
    d = REPO / f"screening_ff_md_sqm_propka_clust_{tag}_gt5"
    return d if d.exists() else REPO / f"screening_ff_md_sqm_propka_clust_{tag}"

weights = {}
for tag in SCHEMES:
    for bf in snap_dir(tag).glob("*/ff14sb/*/basins.json"):
        d = json.loads(bf.read_text())
        for b in d.get("basins", []):
            weights[(tag, d["system"], d["rep"], int(b["snap"]))] = float(b["frac"])
print("weight keys", len(weights), flush=True)

# frames[(tag, method)][(system, rep)] = list of (frame, energy)
frames = defaultdict(lambda: defaultdict(list))

def add(tag, method, system, rep, frame, energy):
    frames[(tag, method)][(system, rep)].append((int(frame), float(energy)))

print("xtb", flush=True)
for tag in SCHEMES:
    root = SCR / f"_xtb_propka_clust_{tag}_gt5"
    for rep in REPS:
        base = root / rep
        if not base.exists():
            continue
        for v in XTB:
            for jf in base.rglob(f"{v}.json"):
                try:
                    d = json.loads(jf.read_text())
                except (OSError, ValueError):
                    continue
                if d.get("status") not in ("ok", "cached") or d.get("energy") is None:
                    continue
                system = d.get("system") or jf.parent.parent.name
                if d.get("frame") is None:
                    continue
                add(tag, v, system, rep, d["frame"], d["energy"])

print("pm6", flush=True)
for tag in SCHEMES:
    root = SCR / f"_qmmm_trunc_propka_clust_{tag}_gt5"
    for rep in REPS:
        base = root / rep
        if not base.exists():
            continue
        for log in base.rglob("cuby_*_sp.log"):
            name = log.name
            txt = log.read_text(errors="ignore")
            if not txt.startswith("EXIT=0"):
                continue
            m = E_RE.search(txt)
            if not m:
                continue
            preset = name[len("cuby_"):-len("_sp.log")]
            # .../rep/SYS/c8/wet/fN/file
            system = log.parts[-5]
            frame = int(log.parts[-2][1:])
            add(tag, preset, system, rep, frame, m.group(1))

print("mm", flush=True)
for tag in SCHEMES:
    root = ART / f"mmgbsa_propka_clust_{tag}_gt5"
    if not root.exists():
        print(" missing", root)
        continue
    for f in root.glob("*/*/f*/FINAL_RESULTS_CLOSEST30"):
        system, rep, fr = f.parts[-4], f.parts[-3], f.parts[-2]
        if rep not in REPS:
            continue
        m = DG_RE.search(f.read_text(errors="ignore"))
        if not m:
            continue
        add(tag, "mmgbsa", system, rep, fr[1:], m.group(1))

print("arms", len(frames), flush=True)

def r2_rho(series):
    common = [s for s in series.dropna().index if s in exp.index]
    if len(common) < 5:
        return np.nan, np.nan, len(common)
    x = series.loc[common].astype(float)
    y = exp.loc[common]
    pr = stats.pearsonr(x, y)[0]
    rho = stats.spearmanr(x, y).correlation
    return float(pr * pr), float(rho), len(common)

def std_or_nan(e):
    if len(e) < 2:
        return np.nan
    s = float(np.std(e, ddof=1))
    return s if s > 1e-8 else np.nan

records = []
identity = []
for (tag, method), groups in frames.items():
    # global scatter scale for this arm: median per-trajectory SD
    stds = []
    prepared = {}
    for key, items in groups.items():
        # unique frames, last write wins
        byf = {}
        for fr, en in items:
            byf[fr] = en
        frs = np.array(list(byf.keys()))
        ens = np.array(list(byf.values()), float)
        fracs = np.array([weights.get((tag, key[0], key[1], int(fr)), np.nan) for fr in frs])
        prepared[key] = (ens, fracs)
        s = std_or_nan(ens)
        if np.isfinite(s):
            stds.append(s)
    s_global = float(np.median(stds)) if stds else np.nan

    def weights_for(ens, fracs, how):
        n = len(ens)
        if how == "mean":
            return np.ones(n)
        if how == "pop":
            if not np.isfinite(fracs).all() or fracs.sum() <= 0:
                return None
            return fracs
        if how == "boltz":
            x = np.exp(-(ens - ens.min()) / KT)
            return x
        if how in ("scatter", "tilt", "gtilt"):
            s = std_or_nan(ens) if how != "gtilt" else s_global
            if not np.isfinite(s):
                base = np.ones(n)
            else:
                base = np.exp(-(ens - ens.min()) / s)
            if how == "scatter":
                return base
            if not np.isfinite(fracs).all() or fracs.sum() <= 0:
                return None
            return fracs * base
        raise ValueError(how)

    hows = ("mean", "pop", "boltz", "scatter", "tilt", "gtilt")
    mats = {}
    neffs = {h: [] for h in hows}
    boltz_vs_min = []
    for how in hows:
        rows = []
        for (system, rep), (ens, fracs) in prepared.items():
            w = weights_for(ens, fracs, how)
            if w is None or w.sum() <= 0:
                continue
            p = w / w.sum()
            rows.append((system, rep, float(np.dot(p, ens))))
            neffs[how].append(float(1.0 / np.dot(p, p)))
            if how == "boltz":
                boltz_vs_min.append(abs(float(np.dot(p, ens)) - float(ens.min())))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["System", "rep", "score"])
        mats[how] = df.pivot_table(index="System", columns="rep", values="score")
    if boltz_vs_min:
        identity.append(float(np.median(boltz_vs_min)))

    label_key = {
        "gfn2_gbsa": "GFN2_GBSA", "gfn2_alpb": "GFN2_ALPB", "gfn2_cosmo": "GFN2_COSMO",
        "gfn1_gbsa": "GFN1_GBSA", "gfn1_alpb": "GFN1_ALPB", "gfn1_cosmo": "GFN1_COSMO",
        "mmgbsa": "MMGBSA",
    }.get(method, method.upper())
    if method == "mmgbsa":
        display = "Nwat-MM/GBSA"
    elif method.startswith("gfn"):
        display = protocol_display_name(f"{label_key}_basin_{tag}_unw")
        # strip basin qualifier; scheme is separate
        display = display.split(",")[0]
    else:
        display = protocol_display_name(f"{label_key}_basin_{tag}_unw")
        display = display.split(",")[0]

    rec = {
        "scheme": tag,
        "method": method,
        "display": display,
        "s_global": None if not np.isfinite(s_global) else round(s_global, 2),
        "n_groups": len(prepared),
    }
    for how in hows:
        mat = mats.get(how)
        if mat is None or mat.empty:
            continue
        reps = [c for c in REPS if c in mat.columns]
        p5, rho, n = r2_rho(mat[reps].mean(axis=1))
        per = []
        for c in reps:
            v, _, nn = r2_rho(mat[c])
            if nn >= 15 and np.isfinite(v):
                per.append(v)
        spread = (max(per) - min(per)) if len(per) >= 2 else None
        rec[how] = {
            "r2": None if not np.isfinite(p5) else round(p5, 4),
            "rho": None if not np.isfinite(rho) else round(rho, 4),
            "spread": None if spread is None else round(spread, 4),
            "neff": round(float(np.median(neffs[how])), 2) if neffs[how] else None,
            "n": n,
        }
    records.append(rec)

out = DATA / "basin_weights.json"
out.write_text(json.dumps(records))
print("records", len(records), "median |boltz-min| kcal", float(np.median(identity)))

def grab(rec, how, key):
    return rec.get(how, {}).get(key)

# summary vs unweighted
for how, name in (("boltz", "Boltzmann 298 K"), ("scatter", "scatter softmax"),
                  ("tilt", "population x scatter"), ("gtilt", "population x global scatter"),
                  ("pop", "population")):
    dr, ds, both, worse_sp = [], [], 0, 0
    for rec in records:
        a, b = grab(rec, "mean", "r2"), grab(rec, how, "r2")
        sa, sb = grab(rec, "mean", "spread"), grab(rec, how, "spread")
        if a is None or b is None:
            continue
        dr.append(b - a)
        if sa is not None and sb is not None:
            ds.append(sb - sa)
            if b > a + 0.01 and sb < sa - 0.01:
                both += 1
            if b > a + 0.01 and sb > sa + 0.01:
                worse_sp += 1
    print(f"\n{name}: median dR2 {np.median(dr):+.3f}  median dSpread {np.median(ds):+.3f}  "
          f"arms {len(dr)}  R2 up>0.01 and spread down>0.01: {both}  "
          f"R2 up and spread up: {worse_sp}")

# selected GFN2-xTB and MM/GBSA rows
print("\nSelected protocols")
for rec in records:
    if rec["method"] in ("gfn2_gbsa", "gfn2_alpb", "gfn2_cosmo", "mmgbsa") and rec["scheme"] in ("iface_sc", "aa"):
        bits = []
        for how in ("mean", "pop", "boltz", "scatter", "tilt", "gtilt"):
            h = rec.get(how, {})
            bits.append(f"{how} {h.get('r2')} sp {h.get('spread')} n {h.get('neff')}")
        print(rec["scheme"], rec["display"], " | ".join(bits))
print("wrote", out, "bytes", out.stat().st_size)
flat = []
for rec in records:
    for how in ("mean", "pop", "boltz", "scatter", "tilt", "gtilt"):
        h = rec.get(how)
        if h:
            flat.append({"scheme": rec["scheme"], "method": rec["method"],
                         "display": rec["display"], "weight": how, **h})
pd.DataFrame(flat).to_csv(DATA / "basin_weights.csv", index=False)
print("wrote", DATA / "basin_weights.csv")
