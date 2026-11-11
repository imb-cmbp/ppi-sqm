# ppi-sqm

https://github.com/imb-cmbp/ppi-sqm

Code, result tables and structures for *What Limits Semiempirical Affinity Prediction at Protein–Protein Interfaces?*

This repository is released under the MIT licence (see `LICENSE`). It contains the analysis and scoring code, the CSV tables behind the reported statistics, and the PDB files those programs read. One command scores a single capped interface with the semiempirical model you choose.

## Score one structure

The input PDB uses the layout of the scored models: chain A is partner A plus any interfacial waters, chain B is partner B. Caps, where the backbone was cut, are the neutral residues BNC, BCC, BCG and BCB.

```bash
python scripts/score_model.py --list
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2-gbsa
```

`--model` is one of:

| name | method |
|---|---|
| `gfn2-gbsa` | GFN2-xTB/GBSA, the leading protocol in the paper |
| `gfn2-alpb` | GFN2-xTB/ALPB |
| `gfn2-ddcosmo` | GFN2-xTB/ddCOSMO |
| `gfn2-gas` | GFN2-xTB, gas phase |
| `gfn1-gbsa` | GFN1-xTB/GBSA |
| `gfn1-alpb` | GFN1-xTB/ALPB |
| `gfn1-ddcosmo` | GFN1-xTB/ddCOSMO |

The printed energy is E(AB) − E(A) − E(B) in kcal/mol. Partner charges are read from a `capped.trunc.json` sidecar when that file sits next to the PDB, otherwise from the residue names and the chain termini. Pass `--charge-a` and `--charge-b` to set them yourself. `--charges-only` prints the charges and does not run xtb. Logs are written to `score_<model>/`.

xtb 6.7.1 has to be installed. If it is not on `PATH`, set `XTB_HOME` to the prefix that contains `bin/xtb` and `share/xtb`.

```bash
conda env create -f environment.yml
conda activate ppi-sqm
```

`environment.yml` also installs the Python packages used to rebuild the figures.

GFN2-xTB/CPCM-X, the PM6 and PM7 models, and Nwat-MM/GBSA are the batch calculations in `workflow/`. They need Amber 22, MOPAC 23.2.3, cuby4 and, for CPCM-X, the standalone CPCM-X program. Paths are the environment variables in `scripts/paths.py`.

## Rebuild the tables and figures

The CSV files in `data/` are the interaction energies and statistics reported in the paper. With those files in place, this regenerates the comparison statistics and the figures. It does not repeat the quantum-chemical calculations:

```bash
SKIP_HARVEST=1 workflow/04_analysis.sh
```

## What else is in the repository

```
scripts/          preparation, scoring, statistics and figures
workflow/         01_md.sh, 02_score.sh, 03_cpcmx.sh, 04_analysis.sh
data/             result tables (CSV)
structures/capped_model_1ACB.pdb
                  one capped interface, the input of score_model.py
structures/chain_reference/
                  the twenty starting complexes; the truncation code reads these
                  for the partner boundaries
params/caps/      tleap parameters of the cap residues
```

The chain-reference files are full complexes (their chain IDs are the PDB chain IDs). `score_model.py` scores a capped model, not those files.

## Full calculation

`workflow/01_md.sh` builds the topologies and runs the trajectories. `workflow/02_score.sh` and `workflow/03_cpcmx.sh` score them. The calculation tree (about 265 GB of trajectories, plus the capped models of every frame) is not in this repository; it is available from the authors on request. The benchmark files distributed by Maffucci and Contini with their 2016 paper are not included either. Set `PPISQM_WORK` to the calculation tree and `PPISQM_BENCHMARK` to that benchmark before running `01_md.sh` or a harvest that reads the original outputs.

## Citation

Please cite the article when using this code. The experimental binding free energies in `data/experimental_data.csv` are those of the structure-based affinity benchmark (Kastritis et al., Protein Sci. 2011, 20, 482; Vreven et al., J. Mol. Biol. 2015, 427, 3031), for the twenty complexes of Maffucci and Contini (J. Chem. Inf. Model. 2016, 56, 1692).
