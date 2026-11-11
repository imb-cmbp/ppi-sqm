#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# Formal charges by residue/ion name, including Amber protonation-state names.
RESIDUE_CHARGES = {
    "ARG": 1, "LYS": 1, "HIS": 0, 
    "ASP": -1, "GLU": -1,
    "HIP": 1, "HIE": 0, "HID": 0,
    "LYN": 0, "ASH": 0, "GLH": 0,
    "ZN": 2, "MG": 2, "CA": 2, "FE": 2, "MN": 2, "NA": 1, "K": 1, "CL": -1
}

def to_atom_ranges(indices):
    """Converts a list of atom indices [1,2,3,5,6] into a Cuby4 range string '1-3,5-6'"""
    if not indices: return ""
    indices = sorted(list(set(indices)))
    ranges = []
    start = end = indices[0]
    for n in indices[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    return ",".join(ranges)

def get_pdb_chains(pdb_path):
    """Parses PDB to find chains, calculate formal charges, and group ATOM INDICES."""
    chains = {}
    with open(pdb_path, 'r') as f:
        atom_idx = 1 # Cuby4 uses 1-based sequential atom indexing
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21].strip()
                if not chain_id: chain_id = "A" 
                
                res_name = line[17:20].strip().upper()
                
                if chain_id not in chains:
                    chains[chain_id] = {"residues": set(), "charge": 0, "atoms": []}
                
                res_seq = line[22:26].strip()
                res_key = f"{res_name}{res_seq}"
                
                if res_key not in chains[chain_id]["residues"]:
                    chains[chain_id]["residues"].add(res_key)
                    chains[chain_id]["charge"] += RESIDUE_CHARGES.get(res_name, 0)
                
                chains[chain_id]["atoms"].append(atom_idx)
                atom_idx += 1
                    
    return chains

SQM_PRESETS = {
    "pm6_d3h4x_cosmo2": {
        "method": "pm6",
        "mopac_corrections": "d3h4x",
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_d3h4s_cosmo2": {
        # PM6-D3H4XS: MOPAC bundled D3H4X + Cuby s_bond (JCTC 2021 S···O/N repulsion).
        "method": "pm6",
        "mopac_corrections": "d3h4x",
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": "s_bond",
    },
    "pm6_d3h4s": {
        # PM6-D3H4XS in gas phase.
        "method": "pm6",
        "mopac_corrections": "d3h4x",
        "solvent_model": None,
        "mopac_cosmo2_surf_tension": None,
        "mopac_keywords": "MOZYME",
        "modifiers": "s_bond",
    },
    "pm7_d3h4x_cosmo2": {
        # Not usable: MOPAC has no PM7-D3H4X method (see pm7_d3h4x).
        "method": "pm7",
        "mopac_corrections": "d3h4x",
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm7_cosmo2": {
        # PM7 with implicit solvent (COSMO2) but no dispersion/H-bond corrections
        "method": "pm7",
        "mopac_corrections": None,
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm7": {
        # Bare PM7 in gas phase, no corrections, no solvent
        "method": "pm7",
        "mopac_corrections": None,
        "solvent_model": None,
        "mopac_cosmo2_surf_tension": None,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_d3h4x": {
        # PM6 + D3H4X dispersion/H-bond/halogen corrections, gas phase, MOZYME.
        # No COSMO surface, so fast and numerically robust.
        "method": "pm6",
        "mopac_corrections": "d3h4x",
        "solvent_model": None,
        "mopac_cosmo2_surf_tension": None,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm7_d3h4x": {
        # Not usable. MOPAC 23.2.3 has no PM7-D3H4X method; Cuby's dispersion3
        # has no PM7 parameters (AM1, PM3, PM6, RM1, OM3 only); and PM7 already
        # describes noncovalent interactions, whereas D3H4 is fitted for PM6.
        # Use "pm7" or "pm7_cosmo2" for PM7.
        "method": "pm7",
        "mopac_corrections": "d3h4x",
        "solvent_model": None,
        "mopac_cosmo2_surf_tension": None,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_dhplus_cosmo2": {
        "method": "pm6",
        "mopac_corrections": "dh+",
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_dhplus_cosmo": {
        "method": "pm6",
        "mopac_corrections": "dh+",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_cosmo2": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_cosmo": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    # Halogen-free dispersion + H-bond corrections (Rezac-Hobza D3H4 family),
    # COSMO only.
    "pm6_d3h4_cosmo": {
        "method": "pm6",
        "mopac_corrections": "d3h4",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_d3h4paren_cosmo": {
        # the D3(H4) variant
        "method": "pm6",
        "mopac_corrections": "d3(h4)",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_d3_cosmo": {
        # dispersion only (isolate the D3 term)
        "method": "pm6",
        "mopac_corrections": "d3",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_h4_cosmo": {
        # Not usable: MOPAC 23.2.3 has no PM6-H4 method ("The method requested:
        # "PM6-H4" does not exist"), unlike d3 / d3h4 / d3h4x / dh+ / dh2.
        # Use pm6_h4_cosmo_cuby instead.
        "method": "pm6",
        "mopac_corrections": "h4",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    "pm6_h4_cosmo_cuby": {
        # H-bond correction only (isolates the H4 term), applied by Cuby because
        # MOPAC has no PM6-H4 method. Same mechanism as pm6_d3h4_cosmo_cuby
        # without dispersion3; compare with the Cuby-side presets rather than
        # with MOPAC-side pm6_d3h4_cosmo.
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": "h_bonds4",
    },
    "pm6_dh2_cosmo": {
        # D2 + H2 correction, predecessor of dh+ and d3h4
        "method": "pm6",
        "mopac_corrections": "dh2",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": None,
    },
    # Cuby-side D3H4, so the H4 charged-group scaling can be switched off.
    # MOPAC's `mopac_corrections: d3h4` always scales H-bonds to COO-/NH3+
    # groups; Yurenko et al. (JCIM 2025, 65, 5690) recommend switching this off
    # (their fcharge=1) for H-bonds in solution. Cuby exposes the switch as
    # h_bonds4_scale_charged via its dispersion3 + h_bonds4 modifiers.
    # pm6_d3h4_cosmo_cuby should reproduce pm6_d3h4_cosmo; pm6_d3h4_cosmo_nofc
    # disables the scaling.
    "pm6_d3h4_cosmo_cuby": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": "dispersion3, h_bonds4",
    },
    "pm6_d3h4_cosmo_nofc": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": "dispersion3, h_bonds4",
        # Cuby reads modifier settings only from a "modifier_<name>" block;
        # a top-level h_bonds4_scale_charged is ignored.
        "modifier_blocks": {"h_bonds4": {"h_bonds4_scale_charged": "no"}},
    },
    "pm6_d3h4s_cosmo": {
        # PM6-D3H4S as published: D3H4 (not D3H4X) plus the S...O/N chalcogen
        # term. pm6_d3h4s_cosmo2 uses D3H4X instead and is a different Hamiltonian.
        "method": "pm6",
        "mopac_corrections": "d3h4",
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MOZYME",
        "modifiers": "s_bond",
    },
    "pm6_ml": {
        # PM6-ML via the mopac-ml wrapper (ML correction + D3 handled internally).
        # Requires mopac_exe to point at the mopac-ml-cuby launcher.
        # MOZYME keeps the base PM6 SCF linear-scaling for large PPI interfaces.
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": None,
        "mopac_cosmo2_surf_tension": None,
        "mopac_keywords": "MLCORR MOZYME",
        "modifiers": None,
        "is_pm6_ml": True,
    },
    "pm6_ml_cosmo": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo",
        "solvent_epsilon": 80.0,
        "mopac_keywords": "MLCORR MOZYME",
        "modifiers": None,
        "is_pm6_ml": True,
    },
    "pm6_ml_cosmo2": {
        "method": "pm6",
        "mopac_corrections": None,
        "solvent_model": "cosmo2",
        "mopac_cosmo2_surf_tension": 0.046,
        "mopac_keywords": "MLCORR MOZYME",
        "modifiers": None,
        "is_pm6_ml": True,
    },
}

# Default launcher for the PM6-ML wrapper (drop-in MOPAC replacement)
from paths import MOPAC_ML_EXE as PM6_ML_EXE  # noqa: E402

def write_modifier_blocks(f, preset):
    """Emit per-modifier settings blocks for a preset.

    Cuby resolves a modifier's settings from a block named ``modifier_<name>``
    and ignores the same keyword written at the top level, so anything that
    configures a modifier (e.g. h_bonds4_scale_charged) has to be nested here.
    """
    for mod_name, settings in (preset.get("modifier_blocks") or {}).items():
        f.write(f"\nmodifier_{mod_name}:\n")
        for key, value in settings.items():
            f.write(f"  {key}: {value}\n")


def write_ppi_yaml(pdb_path, chains, mopac_exe, out_yaml, parallel, sqm_preset="pm6_d3h4x_cosmo2", torchmd_model=None):
    chain_ids = sorted(list(chains.keys()))
    if len(chain_ids) < 2:
        raise ValueError(f"PDB {pdb_path} only contains one chain ({chain_ids}). Cannot do PPI.")

    mol_a_chain = chain_ids[0]
    mol_b_chains = chain_ids[1:]
    
    charge_a = chains[mol_a_chain]["charge"]
    charge_b = sum(chains[c]["charge"] for c in mol_b_chains)
    charge_ab = charge_a + charge_b # Cuby4 requires molecule_ab charge = charge_a + charge_b
    
    # Cuby4 "@" atom selection strings
    sel_a = f"@{to_atom_ranges(chains[mol_a_chain]['atoms'])}"
    
    b_atoms = []
    for c in mol_b_chains:
        b_atoms.extend(chains[c]["atoms"])
    sel_b = f"@{to_atom_ranges(b_atoms)}"

    if sqm_preset not in SQM_PRESETS:
        raise ValueError(f"Unknown SQM preset '{sqm_preset}'. Choose from: {', '.join(SQM_PRESETS)}")

    preset = SQM_PRESETS[sqm_preset]
    # PM6-ML uses the self-contained mopac-ml wrapper as the MOPAC executable.
    if preset.get("is_pm6_ml"):
        mopac_exe = PM6_ML_EXE

    with open(out_yaml, "w") as f:
        f.write("job: interaction\n")
        f.write(f"geometry: {pdb_path.name}\n")
        f.write("geometry_setup_from_file: yes\n\n")

        f.write("interface: mopac\n")
        f.write(f"method: {preset['method']}\n")
        if preset.get("mopac_corrections"):
            f.write(f"mopac_corrections: {preset['mopac_corrections']}\n")
        if preset.get("solvent_model"):
            f.write(f"solvent_model: {preset['solvent_model']}\n")
            if preset.get("solvent_epsilon") is not None:
                f.write(f"solvent_epsilon: {preset['solvent_epsilon']}\n")
            if preset["solvent_model"] == "cosmo2" and preset.get("mopac_cosmo2_surf_tension") is not None:
                f.write(f"mopac_cosmo2_surf_tension: {preset['mopac_cosmo2_surf_tension']}\n")
        if preset.get("modifiers"):
            f.write(f"modifiers: {preset['modifiers']}\n")
        f.write(f'mopac_keywords: "{preset["mopac_keywords"]}"\n')
        write_modifier_blocks(f, preset)
        f.write("job_cleanup: no\n\n")

        f.write(f"mopac_exe: {mopac_exe}\n")
        f.write(f"parallel: {parallel}\n")
        
        f.write("\nmolecule_a:\n")
        f.write(f'  selection: "{sel_a}"\n')
        f.write(f"  charge: {charge_a}\n")
        
        f.write("\nmolecule_b:\n")
        f.write(f'  selection: "{sel_b}"\n')
        f.write(f"  charge: {charge_b}\n")

        f.write("\nmolecule_ab:\n")
        f.write(f"  charge: {charge_ab}\n")

    print(f"Generated YAML: A={mol_a_chain} (q={charge_a}), B={mol_b_chains} (q={charge_b})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--mopac-exe", default=os.environ.get("MOPAC_EXE", "mopac"))
    parser.add_argument("--out-yaml", default="interaction.yaml")
    parser.add_argument("--parallel", type=int, default=60)
    parser.add_argument("--sqm-preset", default="pm6_d3h4x_cosmo2", choices=list(SQM_PRESETS.keys()))
    parser.add_argument("--torchmd-model", default=None, help="Required for pm6_ml preset")
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    chains = get_pdb_chains(pdb_path)
    write_ppi_yaml(
        pdb_path, chains, args.mopac_exe, args.out_yaml, args.parallel,
        sqm_preset=args.sqm_preset, torchmd_model=args.torchmd_model,
    )

if __name__ == "__main__":
    main()