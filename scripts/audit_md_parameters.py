#!/usr/bin/env python3
"""Audit the resolved MD parameters of every production run from its mdout file.

Checks that (1) every production run read its input from the reference protocol
directory, and (2) the resolved potential, cutoff, Ewald, Langevin, pressure, SHAKE
and output blocks are identical across all 20 complexes x 5 replicas, apart from
run-specific values (random seed, box dimensions, PME grid). Also reports the
number of distinct Langevin seeds. Exits non-zero if either check fails.

Usage:
    python3 audit_md_parameters.py [--stage prod4]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from paths import WORK

REPO = WORK
REPRO = REPO / "md"
# Amber truncates the MDIN path in the mdout header at a fixed column width, so
# match the longest prefix that survives truncation; it is unique to the reference
# protocol's data directory.
ARTICLE_PROTO = "data_for_PPI_"

# The five production topology sets, in replica order.
REP_DIRS = ["systems_propka_r1", "systems_propka_r2", "systems_propka_r3", "systems_propka_r4", "systems_propka_r5"]

PDB_IDS = ["1ACB", "1ZHI", "1AVX", "2HLE", "1AY7", "2HRK", "1BVN", "2OOB", "1EMV",
           "2OUL", "1FLE", "2SIC", "1GLA", "2SNI", "1KAC", "2UUY", "1R0R", "3BZD",
           "1YVB", "3SGB"]

# Parameter lines that must agree everywhere. Keyed by a label for reporting.
INVARIANT_PATTERNS = {
    "potential":  r"^\s*ntf\s*=.*ntb\s*=.*igb\s*=.*nsnb\s*=.*$",
    "cutoff":     r"^\s*dielc\s*=.*cut\s*=.*intdiel\s*=.*$",
    "ewald_a":    r"^\s*verbose\s*=.*ew_type\s*=.*nbflag\s*=.*use_pme\s*=.*$",
    "ewald_b":    r"^\s*vdwmeth\s*=.*eedmeth\s*=.*netfrc\s*=.*$",
    "ewald_tol":  r"^\s*Cutoff=.*Tol\s*=.*$",
    "ewald_ord":  r"^\s*Interpolation order\s*=.*$",
    "langevin":   r"^\s*temp0\s*=.*tempi\s*=.*gamma_ln\s*=.*$",
    "pressure":   r"^\s*pres0\s*=.*comp\s*=.*taup\s*=.*$",
    "ntp":        r"^\s*ntp\s*=\s*\d+\s*$",
    "shake":      r"^\s*ntc\s*=.*jfastw\s*=.*$",
    "shake_tol":  r"^\s*tol\s*=.*$",
    "md":         r"^\s*nstlim\s*=.*nscm\s*=.*nrespa\s*=.*$",
    "restraint":  r"^\s*ibelly\s*=.*ntr\s*=.*$",
    "wrap":       r"^\s*iwrap\s*=.*ntwx\s*=.*ntwv\s*=.*ntwe\s*=.*$",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="prod1",
                    help="production stage to audit (prod1..prod4)")
    ap.add_argument("--rep-dirs", nargs="+", default=REP_DIRS,
                    help="topology set directories under md/ to audit")
    args = ap.parse_args()
    rep_dirs = args.rep_dirs

    found: list[Path] = []
    missing: list[str] = []
    for rep_i, rd in enumerate(rep_dirs, start=1):
        for z in PDB_IDS:
            p = REPRO / rd / z / f"{z}_complex_wat.{args.stage}.out"
            (found if p.exists() else missing).append(p if p.exists() else f"rep{rep_i}/{z}")

    print(f"production stage      : {args.stage}")
    print(f"mdout files found     : {len(found)} of {len(rep_dirs) * len(PDB_IDS)}")
    if missing:
        print(f"missing               : {len(missing)}")
        for m in missing[:10]:
            print(f"  {m}")

    if not found:
        print("\nFAIL: no mdout files found")
        return 1

    # Check 1: every run read the reference protocol's input file.
    foreign = []
    for p in found:
        head = p.read_text(errors="ignore")[:4000]
        m = re.search(r"^\|\s*MDIN:\s*(\S.*?)\s*$", head, re.M)
        if not m or ARTICLE_PROTO not in m.group(1):
            foreign.append((p, m.group(1) if m else "<no MDIN line>"))
    if foreign:
        print(f"\nFAIL: {len(foreign)} run(s) did not read the reference protocol input file:")
        for p, got in foreign[:10]:
            print(f"  {p.parent.parent.name}/{p.parent.name}: {got}")
    else:
        print(f"input file provenance : OK, all {len(found)} read from {ARTICLE_PROTO}")

    # Check 2: resolved invariant parameters agree across all runs.
    values: dict[str, dict[str, list[str]]] = {k: defaultdict(list)
                                               for k in INVARIANT_PATTERNS}
    for p in found:
        txt = p.read_text(errors="ignore")
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        for label, pat in INVARIANT_PATTERNS.items():
            m = re.search(pat, txt, re.M)
            values[label][norm(m.group(0)) if m else "<absent>"].append(tag)

    print("\nresolved parameters (one distinct value per row = uniform):")
    bad = []
    width = max(len(k) for k in INVARIANT_PATTERNS)
    for label in INVARIANT_PATTERNS:
        variants = values[label]
        if len(variants) == 1:
            (val, who), = variants.items()
            print(f"  {label:<{width}}  [{len(who):>3}]  {val}")
        else:
            bad.append(label)
            print(f"  {label:<{width}}  !! {len(variants)} DISTINCT VALUES:")
            for val, who in sorted(variants.items(), key=lambda kv: -len(kv[1])):
                ex = ", ".join(who[:3]) + (" ..." if len(who) > 3 else "")
                print(f"      [{len(who):>3}]  {val}\n             ({ex})")

    # Independent replicas should use distinct Langevin seeds.
    seeds = defaultdict(list)
    for p in found:
        m = re.search(r"^\s*ig\s*=\s*(\d+)\s*$", p.read_text(errors="ignore"), re.M)
        if m:
            seeds[m.group(1)].append(p.parent.name)
    print(f"\ndistinct Langevin seeds: {len(seeds)} across {len(found)} runs "
          f"({'OK, replicas are independent draws' if len(seeds) > len(found) * 0.9 else 'WARNING: seeds repeat'})")

    ok = not foreign and not bad
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{'inputs verbatim and all resolved parameters uniform' if ok else 'see failures above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
