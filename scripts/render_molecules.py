#!/usr/bin/env python3
"""Render the molecular panels of Figure 1 with PyMOL; PNGs go to figures/panels/.

Usage (from the repository root): pymol -cq scripts/render_molecules.py
Panels: the intact 1ACB complex with its interface residues (A), the capped interface
model with retained waters (B), one retained residue with its two caps (C), and the
interfaces of further complexes (D). Shape and representation complement colour.
"""
import os

import numpy as np

from pymol import cmd

# Run from the repository root, or set PPISQM_ROOT.
ROOT = os.environ.get("PPISQM_ROOT", os.getcwd())
BENCHMARK = os.environ.get("PPISQM_BENCHMARK", os.path.join(ROOT, "benchmark", "data_for_PPI_Nwat_MMGBSA"))
OUT = os.path.join(os.environ.get("PPISQM_FIGURES", os.path.join(ROOT, "figures")), "panels")
FULL = os.path.join(BENCHMARK, "complexes", "complex_1ACB_leap.pdb")
TRUNC = os.path.join(ROOT, "structures", "capped_model_1ACB.pdb")
os.makedirs(OUT, exist_ok=True)

# colour-blind-safe blue / orange pair, matching the plots
cmd.set_color("pA", [0.17, 0.42, 0.69])
cmd.set_color("pB", [0.87, 0.42, 0.13])
cmd.set_color("wat", [0.20, 0.55, 0.45])

cmd.set("ray_opaque_background", 0)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 0)
cmd.set("cartoon_transparency", 0.0)
cmd.set("surface_quality", 1)
cmd.set("sphere_scale", 0.30)
cmd.set("stick_radius", 0.13)
cmd.set("depth_cue", 0)
cmd.set("ray_shadows", 0)

# Panel A: intact complex
cmd.load(FULL, "cplx")
cmd.remove("cplx and resn WAT+HOH+Na++Cl-")
cmd.hide("everything", "cplx")
cmd.show("cartoon", "cplx")
cmd.color("pA", "cplx and chain A")
cmd.color("pB", "cplx and chain B")

# mark the residues that the truncation keeps: within 5 A across the interface
cmd.select("ifaceA", "byres (cplx and chain A within 5 of (cplx and chain B))")
cmd.select("ifaceB", "byres (cplx and chain B within 5 of (cplx and chain A))")
cmd.show("sticks", "(ifaceA or ifaceB) and not hydro")
cmd.set("cartoon_transparency", 0.65, "cplx")

cmd.orient("cplx")
cmd.turn("y", 15)
cmd.zoom("cplx", 3)
cmd.ray(1400, 1150)
cmd.png(os.path.join(OUT, "panelA_complex.png"), dpi=300)
cmd.delete("all")

# Panel B: capped interface model
cmd.load(TRUNC, "trunc")
cmd.hide("everything", "trunc")
cmd.show("sticks", "trunc and not hydro and not resn HOH")
cmd.color("pA", "trunc and chain A")
cmd.color("pB", "trunc and chain B")
# caps drawn thinner and grey to mark the truncation points
cmd.set("stick_radius", 0.07, "trunc and resn BCC+BNC+BCG+BCB+ACE+NME")
cmd.color("grey60", "trunc and resn BCC+BNC+BCG+BCB+ACE+NME")
# retained interfacial waters as spheres: distinct shape as well as colour
cmd.show("spheres", "trunc and resn HOH and elem O")
cmd.color("wat", "trunc and resn HOH")

cmd.orient("trunc")
cmd.turn("y", 15)
cmd.zoom("trunc", 2)
cmd.ray(1400, 1150)
cmd.png(os.path.join(OUT, "panelB_truncated.png"), dpi=300)

# Panel C: capping chemistry
# Ile13 of chain A, whose two backbone cuts are closed by caps built from the
# deleted neighbours: a formyl group (BNC, the C=O of the preceding residue) and
# an N-methyl group (BCC, the N-H and CA of the following one). Both form amide
# bonds with the retained residue.
FRAG = "trunc and chain A and resi 12-14"
CAPS = f"{FRAG} and resn BNC+BCC"
cmd.hide("everything")
cmd.set("stick_radius", 0.16)
cmd.set("sphere_scale", 0.24)
cmd.show("sticks", FRAG)
cmd.show("spheres", FRAG)
cmd.color("grey50", f"{FRAG} and elem C")
cmd.color("blue", f"{FRAG} and elem N")
cmd.color("red", f"{FRAG} and elem O")
cmd.color("white", f"{FRAG} and elem H")
cmd.create("halo", CAPS)
cmd.hide("everything", "halo")
cmd.show("spheres", "halo")
cmd.set("sphere_scale", 0.62, "halo")
cmd.set("sphere_transparency", 0.55, "halo")
cmd.color("yelloworange", "halo")
cmd.set("orthoscopic", 1)
# Explicit view: backbone horizontal from the formyl carbon to the methyl
# carbon, side chain pointing down in the image plane so it hides neither cap.
xyz = lambda s: np.array(cmd.get_coords(s)).mean(axis=0)
right = xyz(f"{FRAG} and resn BCC and name CA") - xyz(f"{FRAG} and resn BNC and name C")
right /= np.linalg.norm(right)
up = xyz(f"{FRAG} and resn ILE and name CA") - xyz(f"{FRAG} and resn ILE and name CB")
up -= up.dot(right) * right
up /= np.linalg.norm(up)
toward = np.cross(right, up)
cmd.orient(FRAG)
view = list(cmd.get_view())
view[0:9] = [right[0], up[0], toward[0], right[1], up[1], toward[1],
             right[2], up[2], toward[2]]
cmd.set_view(view)
cmd.zoom(FRAG, 1.4)
cmd.set("label_size", 19)
cmd.set("label_color", "black")
cmd.set("label_font_id", 5)
cmd.set("float_labels", 1)
for name, sel, dx, dy, text in (
        ("lab_nterm", f"{FRAG} and resn BNC", -0.8, 2.5, "formyl cap"),
        ("lab_res", f"{FRAG} and resn ILE and name CB", 4.4, -2.6, "Ile13 (retained)"),
        ("lab_cterm", f"{FRAG} and resn BCC", 0.8, 2.5, "N-methyl cap")):
    centre = np.array(cmd.get_coords(sel)).mean(axis=0)
    cmd.pseudoatom(name, pos=list(centre + dx * right + dy * up), label=text)
    cmd.hide("everything", name)
    cmd.show("labels", name)
cmd.ray(1400, 1050)
cmd.png(os.path.join(OUT, "panelC_caps.png"), dpi=300)
cmd.delete("all")

# Panel D: interfaces of further complexes
# Three complexes spanning the range of affinity and interface size in the set.
DIV = [("1EMV", "tightest binder"), ("2OOB", "weakest binder"), ("1BVN", "largest interface")]
for pid, _ in DIV:
    p = os.path.join(BENCHMARK, "complexes", f"complex_{pid}_leap.pdb")
    if not os.path.exists(p):
        continue
    cmd.load(p, pid)
    cmd.remove(f"{pid} and resn WAT+HOH+Na++Cl-")
    cmd.hide("everything", pid)
    cmd.show("surface", f"{pid} and chain A")
    cmd.show("cartoon", f"{pid} and chain B")
    cmd.color("grey80", f"{pid} and chain A")
    cmd.color("pB", f"{pid} and chain B")
    cmd.select("ifA", f"byres ({pid} and chain A within 5 of ({pid} and chain B))")
    cmd.color("pA", "ifA")
    cmd.set("transparency", 0.12, pid)
    cmd.orient(pid)
    cmd.zoom(pid, 2)
    cmd.ray(1100, 950)
    cmd.png(os.path.join(OUT, f"panelD_{pid}.png"), dpi=300)
    cmd.delete(pid)

print("rendered truncation panels A-D")
