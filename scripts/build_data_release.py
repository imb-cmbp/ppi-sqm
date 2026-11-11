#!/usr/bin/env python3
"""Assemble the public data repository: starting structures, scored interface
models with their fragment charges, simulation and scoring parameters, and the
result tables.

Usage: python scripts/build_data_release.py /path/to/output
Reads the calculation tree (PPISQM_WORK), the benchmark protocol files
(PPISQM_BENCHMARK) and the result tables (PPISQM_DATA).
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

import interaction
import run_md
from protocol_names import protocol_display_name
from paths import BENCHMARK, DATA, WORK

MD = WORK / "md" / "systems_propka"
SCR = WORK / "screening_ff_md_sqm"
REPS = ["r1", "r2", "r3", "r4", "r5"]

SAMPLING = [
    ("even10", SCR / "_qmmm_trunc_propka", None),
    ("basins_allatom_rmsd", SCR / "_qmmm_trunc_propka_clust_aa_gt5", "aa"),
    ("basins_backbone_rmsd", SCR / "_qmmm_trunc_propka_clust_bb_gt5", "bb"),
    ("basins_interface_sidechain_rmsd", SCR / "_qmmm_trunc_propka_clust_iface_sc_gt5", "iface_sc"),
    ("basins_interchain_contacts", SCR / "_qmmm_trunc_propka_clust_contacts_gt5", "contacts"),
]

PM6_PRESETS = [
    "pm6_d3h4_cosmo_nofc", "pm6_d3h4_cosmo_cuby", "pm6_d3h4_cosmo", "pm6_d3h4s_cosmo",
    "pm6_d3h4s_cosmo2", "pm6_d3h4x_cosmo2", "pm6_d3h4paren_cosmo", "pm6_d3_cosmo",
    "pm6_h4_cosmo_cuby", "pm6_dh2_cosmo", "pm6_dhplus_cosmo", "pm6_dhplus_cosmo2",
    "pm6_cosmo", "pm6_cosmo2", "pm6_d3h4x", "pm7", "pm7_cosmo2", "pm6_ml",
    "pm6_ml_cosmo", "pm6_ml_cosmo2",
]
XTB_VARIANTS = [("gfn2_gbsa", 2, "gbsa"), ("gfn2_alpb", 2, "alpb"), ("gfn2_cosmo", 2, "cosmo"),
                ("gfn1_gbsa", 1, "gbsa"), ("gfn1_alpb", 1, "alpb"), ("gfn1_cosmo", 1, "cosmo")]

RESULTS = {
    "experimental_data.csv": "experimental_dG.csv",
    "scores_by_trajectory.csv": "scores_by_trajectory.csv",
    "solvent_offsets_frames.csv": "solvent_models_by_frame.csv",
    "solvent_offsets_systems.csv": "solvent_offsets_by_complex.csv",
    "solvent_offsets_descriptor_corr.csv": "solvent_offsets_descriptor_correlations.csv",
    "cpcmx_metrics.csv": "gfn2_solvents_weighting.csv",
    "basin_weights.csv": "basin_weighting.csv",
    "ranking_bootstrap_family.csv": "ranking_bootstrap_family.csv",
    "ranking_bootstrap_top.csv": "ranking_bootstrap_top.csv",
    "ranking_bootstrap_summary.csv": "ranking_bootstrap_summary.csv",
    "method_comparison.csv": "method_comparison.csv",
    "structure_statistics.csv": "complexes_summary.csv",
    "rmsd_analysis.csv": "trajectory_rmsd.csv",
    "cpcmx_statistics.csv": "cpcmx_statistics.csv",
}


def structures(out: Path) -> None:
    prot = out / "structures" / "protonated"
    pka = out / "structures" / "propka"
    prot.mkdir(parents=True, exist_ok=True)
    pka.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(p for p in MD.iterdir() if p.is_dir()):
        sid = d.name
        shutil.copy(d / f"{sid}_propka_heavy.pdb", prot / f"{sid}.pdb")
        shutil.copy(d / f"{sid}_forpka.pka", pka / f"{sid}.pka")
        seen, counts = set(), {}
        for line in (d / f"{sid}_propka_heavy.pdb").read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                key = (line[21], line[22:27])
                if key not in seen:
                    seen.add(key)
                    rn = line[17:20].strip()
                    counts[rn] = counts.get(rn, 0) + 1
        row = {"complex": sid}
        for rn in ("HID", "HIE", "HIP", "ASP", "ASH", "GLU", "GLH", "LYS", "LYN", "CYS", "CYX", "CYM"):
            row[rn] = counts.get(rn, 0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "structures" / "protonation_states.csv", index=False)

    with tarfile.open(out / "structures" / "md_start.tar.gz", "w:gz") as tar:
        for d in sorted(p for p in MD.iterdir() if p.is_dir()):
            for ext in ("top", "crd"):
                f = d / f"{d.name}_complex_wat.{ext}"
                tar.add(f, arcname=f"md_start/{d.name}/{f.name}")
    print(f"structures: {len(rows)} complexes")


def _basin_populations(tag):
    base = WORK / f"screening_ff_md_sqm_propka_clust_{tag}_gt5"
    if not base.exists():
        base = WORK / f"screening_ff_md_sqm_propka_clust_{tag}"
    pops = {}
    for bf in base.glob("*/ff14sb/*/basins.json"):
        d = json.loads(bf.read_text())
        for b in d.get("basins", []):
            pops[(d["system"], d["rep"], int(b["snap"]))] = float(b["frac"])
    return pops


def models(out: Path) -> None:
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, root, tag in SAMPLING:
        pops = _basin_populations(tag) if tag else {}
        with tarfile.open(mdir / f"{name}.tar.gz", "w:gz") as tar:
            for rep in REPS:
                for pdb in sorted((root / rep).glob("*/c8/wet/f*/capped.pdb")):
                    frame = int(pdb.parent.name[1:])
                    sid = pdb.parents[3].name
                    meta = json.loads((pdb.parent / "capped.trunc.json").read_text())
                    arc = f"{name}/{sid}/{rep}/frame{frame}.pdb"
                    tar.add(pdb, arcname=arc)
                    index.append({
                        "sampling": name, "complex": sid, "trajectory": rep, "frame": frame,
                        "file": f"models/{name}.tar.gz:{arc}",
                        "n_atoms": meta.get("n_atoms"), "n_waters": meta.get("n_waters"),
                        "charge_A": meta.get("charge_a"), "charge_B": meta.get("charge_b"),
                        "basin_population": pops.get((sid, rep, frame)) if tag else None,
                    })
        print(f"models: {name} {sum(1 for r in index if r['sampling'] == name)}")
    pd.DataFrame(index).to_csv(mdir / "models_index.csv", index=False)


def _namelist(text):
    body = re.search(r"&cntrl(.*?)(&end|/)\s*$", text, re.S | re.I).group(1)
    items, cur, quote = [], "", None
    for ch in body:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur += ch
        elif ch in ",\n":
            if cur.strip():
                items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def parameters(out: Path) -> None:
    pdir = out / "parameters"
    pdir.mkdir(parents=True, exist_ok=True)
    proto = BENCHMARK / "protocol" / "protocol"
    rows = []
    for label, inp, _start, restrained in run_md.STAGES:
        text = (proto / inp).read_text()
        title = text.splitlines()[0].lstrip("#").strip()
        rows.append({"stage": label, "description": title, **_namelist(text)})
    cols = ["stage", "description"] + sorted({k for r in rows for k in r} - {"stage", "description"})
    pd.DataFrame(rows)[cols].to_csv(pdir / "md_protocol.csv", index=False)

    leap = (MD / "1ACB" / "build.leap").read_text()
    leap = re.sub(r"/\S*/(extra-params/)?([^/\s]+\.(off|prep|frcmod\S*|phos))", r"\2", leap)
    leap = leap.replace("1ACB", "<ID>")
    lines, placed = [], False
    for ln in leap.splitlines():
        if ln.startswith("bond "):
            if not placed:
                lines.append("# one 'bond x.i.SG x.j.SG' per disulfide, see structures/disulfides.csv")
                placed = True
            continue
        lines.append(ln)
    (pdir / "tleap_build.in").write_text("\n".join(lines) + "\n")

    ss = []
    for d in sorted(p for p in MD.iterdir() if p.is_dir()):
        for m in re.finditer(r"^bond x\.(\d+)\.SG x\.(\d+)\.SG", (d / "build.leap").read_text(), re.M):
            ss.append({"complex": d.name, "residue_i": int(m.group(1)), "residue_j": int(m.group(2))})
    pd.DataFrame(ss).to_csv(out / "structures" / "disulfides.csv", index=False)

    presets = []
    for p in PM6_PRESETS:
        cfg = interaction.SQM_PRESETS[p]
        presets.append({
            "preset": p,
            "protocol_name": protocol_display_name(f"{p.upper()}_even10", compact=False,
                                                   omit_even10=True),
            "method": cfg.get("method"),
            "mopac_corrections": cfg.get("mopac_corrections"),
            "cuby4_modifiers": cfg.get("modifiers"),
            "modifier_options": json.dumps(cfg.get("modifier_blocks")) if cfg.get("modifier_blocks") else None,
            "solvent_model": cfg.get("solvent_model"),
            "solvent_epsilon": cfg.get("solvent_epsilon"),
            "cosmo2_surface_tension": cfg.get("mopac_cosmo2_surf_tension"),
            "mopac_keywords": cfg.get("mopac_keywords"),
            "pm6_ml": bool(cfg.get("is_pm6_ml", False)),
        })
    pd.DataFrame(presets).to_csv(pdir / "mopac_presets.csv", index=False)

    xtb = [{"variant": v, "hamiltonian": f"GFN{g}-xTB",
            "command": f"xtb model.xyz --sp --gfn {g} --chrg <q> --acc 1.0 --norestart --{s} water"}
           for v, g, s in XTB_VARIANTS]
    xtb.append({"variant": "gfn2_cpcmx", "hamiltonian": "GFN2-xTB",
                "command": "xtb model.xyz --sp --gfn 2 --chrg <q> --cosmo infinity; "
                           "then CPCM-X (cpx) on the conductor surface, water, 298.15 K"})
    pd.DataFrame(xtb).to_csv(pdir / "xtb_settings.csv", index=False)

    settings = {
        "interface_selection": {"criterion": "buried SASA on complex formation",
                                "min_buried_sasa_A2": 0.5, "fallback_interchain_cutoff_A": 8.0},
        "interfacial_waters": {"n_retained": 20, "candidate_radius_A": 8.0,
                               "anchor_residues": ["ARG", "LYS", "HIS", "ASP", "GLU", "GLN",
                                                   "SER", "THR", "TYR", "TRP"],
                               "assigned_to": "partner A"},
        "capping": {"N_terminal_cut": "formyl from the preceding residue (BNC)",
                    "C_terminal_cut": "N-methyl from the following residue (BCC; BCG after glycine)",
                    "single_residue_gap": "bridging cap (BCB)",
                    "chain_termini": "left uncapped"},
        "sampling": {"even10": "10 frames, every 100th frame of the final production ns",
                     "basins": "10 clusters per trajectory, average linkage (cpptraj); "
                               "centroids of clusters holding >5% of 4000 frames"},
        "mm_gbsa": {"program": "MMPBSA.py", "gb_model": "GB-Neck2 (igb=8)", "salt_M": 0.15,
                    "frames": "100 from the final production ns", "entropy": "not computed",
                    "explicit_waters": [0, 10, 20, 30, 40]},
        "minimised_control": {"program": "sander", "igb": 7, "radii": "bondi", "maxcyc": 1000,
                              "ncyc": 300, "restraint": "backbone N, CA, C, O at 10 kcal/mol/A^2",
                              "caps": "ACE/NME", "scored_with": "pm6_d3h4_cosmo_nofc",
                              "models": "all-atom RMSD basin centroids"},
    }
    (pdir / "scoring_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print("parameters written")


def results(out: Path) -> None:
    rdir = out / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    for src, dst in RESULTS.items():
        shutil.copy(DATA / src, rdir / dst)

    scores = pd.read_csv(DATA / "protocol_scores.csv").set_index("System")
    scores = scores.join(pd.read_csv(DATA / "cpcmx_arms_scores.csv").set_index("System"))
    scores = scores.join(pd.read_csv(DATA / "cpcmx_arms_scores_wt.csv").set_index("System"))
    scores.rename_axis("complex").to_csv(rdir / "protocol_scores.csv", float_format="%.4f")

    metrics = pd.concat([pd.read_csv(DATA / "protocol_metrics.csv"),
                         pd.read_csv(DATA / "cpcmx_arms_metrics.csv")], ignore_index=True)
    metrics = metrics.rename(columns={"arm": "protocol"}).sort_values("pool5_r2", ascending=False)
    metrics.to_csv(rdir / "protocol_metrics.csv", index=False, float_format="%.4f")

    protocols = pd.DataFrame({"protocol": [c for c in scores.columns if c != "Exp_dG"]})
    protocols["description"] = [protocol_display_name(p, compact=False) for p in protocols.protocol]
    protocols.to_csv(rdir / "protocols.csv", index=False)
    print(f"results: {len(RESULTS) + 3} tables")


def main():
    out = Path(sys.argv[1]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    structures(out)
    models(out)
    parameters(out)
    results(out)


if __name__ == "__main__":
    main()
