# ppi-sqm

https://github.com/imb-cmbp/ppi-sqm

Scoring code, result tables and structures for *What Limits Semiempirical Affinity Prediction at Protein–Protein Interfaces?*

The repository is released under the MIT licence (see `LICENSE`). It contains the code that scores protein–protein interfaces with semiempirical models, the CSV tables behind the reported statistics, and the PDB files the scoring reads. It does not run molecular dynamics. The scoring starts from structures you already have: a capped interface model in PDB format, or an Amber topology (prmtop) with its trajectories. The MD protocol of the paper is described in the Supporting Information.

## Installation

```bash
conda env create -f environment.yml
conda activate ppi-sqm
```

The environment provides Python and the analysis packages, xtb 6.7.1, MOPAC 23.2.3, CPCM-X 1.1.0, Ruby and PyMOL. The programs below are installed separately. Each one is located through an environment variable read by `scripts/paths.py`.

| program | needed for | where to get it | variable |
|---|---|---|---|
| cuby4 | every PM6, PM7 and PM6-ML model | http://cuby4.molecular.cz ([installation](http://cuby4.molecular.cz/installation.html)) | `CUBY4_DIR` |
| MOPAC-ML | PM6-ML models | https://github.com/Honza-R/mopac-ml | `MOPAC_ML_DIR`, `PM6ML_PYTHON` |
| AmberTools | trajectory scoring: cpptraj, tleap, sander, parmed, MMPBSA.py | https://ambermd.org/AmberTools.php (free and open source) | `AMBERHOME` |
| CPCM-X parameter files | GFN2-xTB/CPCM-X | https://github.com/grimme-lab/CPCM-X | `CPCMX_DB` |

**cuby4.** Download cuby4 from its website, unpack it and check it with `./cuby4 --info`. Then either set `CUBY4_DIR` to the directory that contains the `cuby4` script, or put that directory on `PATH`. The scripts write a complete cuby4 input for every job, including the MOPAC executable, so no cuby4 configuration file is needed. The paper used cuby4 commit c2b9056. Compiling the optional binary extensions (`./cuby4 --make`) makes cuby4 faster but does not change the results.

**MOPAC.** cuby4 runs the `mopac` executable from the conda environment. To use another build, set `MOPAC_EXE` to its path.

**MOPAC-ML** (only for `pm6_ml`, `pm6_ml_cosmo` and `pm6_ml_cosmo2`). Clone the repository and create its environment as described in its README:

```bash
git clone https://github.com/Honza-R/mopac-ml
conda create -n pm6-ml -c conda-forge torchmd-net simple-dftd3 dftd3-python
export MOPAC_ML_DIR=$PWD/mopac-ml
export PM6ML_PYTHON=$(conda run -n pm6-ml which python)
```

cuby4 calls `scripts/mopac-ml-cuby` in place of MOPAC for these models, and that script starts MOPAC-ML with `PM6ML_PYTHON`. The PM6-ML model files are distributed with MOPAC-ML under the Academic Software Licence.

**AmberTools.** Only the trajectory scoring needs it. Set `AMBERHOME` to the installation; the default is `/opt/amber22`, and the paper used Amber 22.

**CPCM-X.** The `cpx` program is installed by `environment.yml`. Its parameter files are in the source repository, so clone it with `git clone https://github.com/grimme-lab/CPCM-X` and set `CPCMX_DB` to `CPCM-X/DB/xtb`. The default is `~/CPCM-X/DB/xtb`.

**xtb.** `xtb` is found on `PATH`. To use another installation, set `XTB_HOME` to the prefix that contains `bin/xtb` and `share/xtb`.

## Score a trajectory

One command selects the frames, selects the interfacial waters, builds the capped models and scores them. With no flags other than the input, it runs the best protocol in the paper: GFN2-xTB/GBSA (`gfn2_gbsa`) on the centroids of interface side-chain RMSD basins that hold more than 5% of the frames, combined as an unweighted mean. Waters are selected in the two steps used for the paper. cpptraj takes the 30 waters closest to the polar interface residues, and the capped model then keeps the 20 with the best bridging score (within 8 Å of a polar or charged interface residue, closest to being between the two partners). All of them are counted with partner A.

```bash
python scripts/score_model.py --example --threads 4
```

`--example` is the shipped trajectory `examples/2OOB/`: the Amber topology of 2OOB and 40 frames taken every 25th frame from the final nanosecond of trajectory r1. It is there so the command can be run as it stands. It is a short test, not the 4000-frame calculation reported in the paper. On this sample the command keeps 8 basins, 551 atoms and 20 waters in each model, and the unweighted mean is −78.7847 kcal/mol.

The same command on your own files, for one of the twenty complexes:

```bash
python scripts/score_model.py \
    --top complex_wat.prmtop --traj prod1.nc prod2.nc prod3.nc prod4.nc \
    --system 2OOB
```

The complex id selects its dASA mask in `data/interface_masks/` and its partner boundary in `structures/chain_reference/`. For a complex that is not one of the twenty, pass `--dasa-resids` and `--input-pdb` (or `--n-chain-a`).

| flag | default | |
|---|---|---|
| `--model` | `gfn2_gbsa` | any name from `--list` |
| `--scheme basin` | basin | centroids of clusters larger than `--min-frac` (0.05) |
| `--scheme even10` | | ten evenly spaced frames of the last trajectory (stride 100) |
| `--metric` | `iface_sc` | also `aa`, `bb`, `iface_heavy`, `iface_polar`, `contacts` |
| `--weight mean` | unweighted | `--weight population` weights basins by their frame count |
| `--closest-waters` | 30 | waters taken from the box |
| `--max-waters` | 20 | waters kept after the bridging-score ranking |
| `--any-waters` | off | anchor waters on every interface residue, not only polar ones |
| `--water-assign nearer` | `receptor` | count each water with the closer partner |
| `--water-mask` | polar dASA mask | cpptraj mask for the closest-water step |

```bash
python scripts/score_model.py --example --scheme even10
python scripts/score_model.py --example --metric aa --weight population
python scripts/score_model.py --example --model pm6_d3h4x_cosmo2
python scripts/score_model.py --example --max-waters 10 --closest-waters 30
python scripts/score_model.py --example --water-assign nearer
```

Per-frame energies and the mean are printed, and written to `score_<ID>_<model>/frames.json`.

## Score one structure

```bash
python scripts/score_model.py --list
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --threads 8
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model pm6_d3h4x_cosmo2
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model pm6_d3h4_cosmo_nofc
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model pm7
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2_cpcmx --threads 4
```

`--model` defaults to `gfn2_gbsa`. The input PDB uses the layout of the scored models: chain A is partner A plus any interfacial waters, and chain B is partner B. Caps, where the backbone was cut, are the neutral residues BNC, BCC, BCG and BCB. The printed energy is E(AB) − E(A) − E(B) in kcal/mol.

Partner charges are read from a `.trunc.json` sidecar next to the PDB, which `truncate_interface.py` writes, or otherwise from the residue names and chain termini. Pass `--charge-a` and `--charge-b` to set them yourself, and `--charges-only` to print them without running a calculation. If MOPAC (MOZYME) rejects the charges, the script retries with the charges MOPAC computed, as the batch scoring did, and reports the charges it used.

**Water options** (the same flags for every model):

| flag | meaning |
|---|---|
| `--max-waters N` | keep only the first N waters of the PDB (file order; the paper's models are already ranked by bridging score) |
| `--water-assign receptor` | default: all waters stay with partner A, the Nwat-MM/GBSA convention used in the paper |
| `--water-assign nearer` | each water is counted with the closer partner (Supporting Information S7) |

```bash
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2_gbsa --max-waters 20
python scripts/score_model.py --pdb structures/capped_model_1ACB.pdb --model gfn2_gbsa --water-assign nearer
```

Inputs and logs go to `score_<pdb>_<model>/`, or to `--workdir`. Running the same command again replaces that directory. `--threads` sets the threads for xtb, cpx or MOPAC (default 1).

`--model` is one of the following. GFN-xTB models run xtb; `gfn2_cpcmx` runs xtb with the standalone `cpx` program; the others run MOPAC through cuby4.

| name | method | program |
|---|---|---|
| `gfn2_gbsa` | GFN2-xTB/GBSA | xtb |
| `gfn2_alpb` | GFN2-xTB/ALPB | xtb |
| `gfn2_cosmo` | GFN2-xTB/ddCOSMO | xtb |
| `gfn2_cpcmx` | GFN2-xTB/CPCM-X | xtb, cpx |
| `gfn1_gbsa` | GFN1-xTB/GBSA | xtb |
| `gfn1_alpb` | GFN1-xTB/ALPB | xtb |
| `gfn1_cosmo` | GFN1-xTB/ddCOSMO | xtb |
| `pm6_d3h4_cosmo_nofc` | PM6-D3H4/COSMO, scaling off | MOPAC, cuby4 |
| `pm6_d3h4_cosmo_cuby` | PM6-D3H4/COSMO, cuby4 corrections, scaling on | MOPAC, cuby4 |
| `pm6_d3h4_cosmo` | PM6-D3H4/COSMO, MOPAC built-in corrections | MOPAC, cuby4 |
| `pm6_d3h4s_cosmo` | PM6-D3H4S/COSMO | MOPAC, cuby4 |
| `pm6_d3h4s_cosmo2` | PM6-D3H4X + chalcogen/COSMO2 | MOPAC, cuby4 |
| `pm6_d3h4x_cosmo2` | PM6-D3H4X/COSMO2 | MOPAC, cuby4 |
| `pm6_d3h4paren_cosmo` | PM6-D3(H4)/COSMO | MOPAC, cuby4 |
| `pm6_d3_cosmo` | PM6-D3/COSMO | MOPAC, cuby4 |
| `pm6_h4_cosmo_cuby` | PM6-H4/COSMO, cuby4 | MOPAC, cuby4 |
| `pm6_dh2_cosmo` | PM6-DH2/COSMO | MOPAC, cuby4 |
| `pm6_dhplus_cosmo` | PM6-DH+/COSMO | MOPAC, cuby4 |
| `pm6_dhplus_cosmo2` | PM6-DH+/COSMO2 | MOPAC, cuby4 |
| `pm6_cosmo` | PM6/COSMO | MOPAC, cuby4 |
| `pm6_cosmo2` | PM6/COSMO2 | MOPAC, cuby4 |
| `pm6_d3h4x` | PM6-D3H4X, gas phase | MOPAC, cuby4 |
| `pm7` | PM7, gas phase | MOPAC, cuby4 |
| `pm7_cosmo2` | PM7/COSMO2 | MOPAC, cuby4 |
| `pm6_ml` | PM6-ML, gas phase | MOPAC-ML, cuby4 |
| `pm6_ml_cosmo` | PM6-ML/COSMO | MOPAC-ML, cuby4 |
| `pm6_ml_cosmo2` | PM6-ML/COSMO2 | MOPAC-ML, cuby4 |

`structures/capped_model_1ACB.pdb` (889 atoms) is the first even-10 frame of trajectory r1 of 1ACB, scored in the paper. A working installation gives these energies for it. The times were measured on one workstation.

| model | kcal/mol | time |
|---|---|---|
| `gfn2_gbsa` | −148.6916 | 30 s with `--threads 8` |
| `pm6_d3h4_cosmo_nofc` | −125.2465 | 1 min, one thread |
| `pm6_d3h4x_cosmo2` | −93.2844 | 1 min, one thread |
| `pm6_d3_cosmo` | −147.6929 | 1 min, one thread |
| `pm6_ml` | −202.3824 | 1 min, `--threads 4` |
| `gfn2_cpcmx` | −142.5842 | about 8 min, `--threads 4` |

With `--water-assign nearer` on the same structure, `gfn2_gbsa` gives −88.1715 kcal/mol (9 waters with A, 11 with B).

## Select frames from a trajectory

`select_frames.py` reads an Amber topology and its four production trajectories and writes the dry and wet PDBs that the scoring uses. Two schemes match the paper:

```bash
# even-10: ten frames of the final nanosecond
python scripts/select_frames.py --scheme even10 \
    --sysdir $PPISQM_WORK/md/systems_propka_r1 --rep r1 \
    --out-base $PPISQM_WORK/even10_r1 --systems 1ACB

# basins: centroids of clusters that hold >5% of all production frames
python scripts/select_frames.py --scheme basin --metric aa \
    --sysdir $PPISQM_WORK/md/systems_propka_r1 --rep r1 \
    --out-base $PPISQM_WORK/basins_aa_r1 --systems 1ACB
python scripts/select_frames.py --scheme basin --metric iface_sc ...
python scripts/select_frames.py --scheme basin --metric bb ...
python scripts/select_frames.py --scheme basin --metric contacts \
    --n-chain-a 237 ...
```

`--metric` for basins defaults to `iface_sc` (interface side-chain RMSD). The others are `aa` (all-atom RMSD), `bb` (backbone), `iface_heavy`, `iface_polar`, and `contacts` (Jaccard distance of interchain residue contacts). Each wet PDB keeps `--max-waters` waters (default 30) closest to the polar interface mask. The twenty dASA masks of the paper are in `data/interface_masks/`. Override them with `--water-mask` or `--dasa-resids`, and pass `--any-waters` to anchor waters on every interface residue instead of polar ones only. Output layout:

```
<out-base>/<ID>/ff14sb/<rep>/dry_f1.pdb
<out-base>/<ID>/ff14sb/<rep>/wet_f1.pdb
<out-base>/<ID>/ff14sb/<rep>/basins.json   # populations for weighted means
```

## Build a capped model from an MD frame

`truncate_interface.py` cuts the interface out of one frame of an Amber trajectory and writes the capped model and its `.trunc.json` sidecar. The frame is a PDB written by cpptraj from the prmtop: protein residues numbered from 1, partner A first, then partner B, then waters. For example, to take frame 1000 with the 30 waters closest to the interface:

```
parm complex_wat.prmtop
trajin prod.nc 1000 1000
center @CA,C,N mass origin
image origin center
strip :Na+,Cl-
closest 30 :37-40,55-57,186-191,274-283 noimage
trajout frame.pdb nobox
run
```

```bash
python scripts/truncate_interface.py --pdb frame.pdb --out capped.pdb \
    --n-chain-a 237 --protein-res-count 304 \
    --dasa-resids :37-40,55-57,186-191,274-283 --max-waters 20 --polar-waters
python scripts/score_model.py --pdb capped.pdb --model gfn2_gbsa
```

`--n-chain-a` is the number of residues in partner A and `--protein-res-count` is the number of protein residues. Instead of these two, `--input-pdb complex.pdb` takes the partner boundary from the first chain of a complex PDB such as those in `structures/chain_reference/`. `--dasa-resids` lists the interface residues, either as a cpptraj mask or as a JSON file written by `scripts/interface_dasa.py`.

The paper selected the residues that lose more than 0.5 Å² of solvent-accessible surface on binding (PyMOL InterfaceResidues), kept at most 20 interfacial waters, and anchored them on polar interface residues. With the paper's residue list for 1ACB and its MD frame, the options above rebuild `structures/capped_model_1ACB.pdb` byte for byte. Without `--dasa-resids`, the residues within `--cutoff` Å (default 8) of the other partner are selected, which gives larger models than in the paper.

## Score trajectories

`score_model.py --top --traj` (above) is the way to score one complex. `select_frames.py` and `truncate_interface.py` are the two steps it runs, and they can still be called on their own.

`workflow/02_score.sh` runs the full paper protocol on one or more trajectory sets. It extracts frames and cluster representatives, builds the capped models, scores them with the chosen models, and runs Nwat-MM/GBSA and the MM/GB-minimised control. It expects the Amber topology and the four 1 ns production trajectories of each complex:

```
$PPISQM_WORK/md/systems_propka_<set>/<ID>/<ID>_complex_wat.top
$PPISQM_WORK/md/systems_propka_<set>/<ID>/<ID>_complex_wat.prod1.mdcrd ... prod4.mdcrd
```

It also reads the benchmark files distributed by Maffucci and Contini (J. Chem. Inf. Model. 2016, 56, 1692) from `$PPISQM_BENCHMARK`: `complexes/complex_<ID>_leap.pdb`, `protocol/interface.pymol`, `protocol/complex_features.txt` and `protocol/extra-params`. The pipeline is set up for the twenty complexes of the paper (`PDB_IDS` in `scripts/common.py`). For other systems, build the capped models with `truncate_interface.py` and score them with `score_model.py`, as above.

```bash
export PPISQM_WORK=/path/to/work PPISQM_BENCHMARK=/path/to/data_for_PPI_Nwat_MMGBSA
MOPAC_MODELS="pm6_d3h4x_cosmo2 pm6_d3h4_cosmo_nofc pm7" XTB_MODELS="gfn2_gbsa" \
    workflow/02_score.sh r1 r2 r3 r4 r5
workflow/03_cpcmx.sh
```

`MOPAC_MODELS` and `XTB_MODELS` take names from `score_model.py --list`. Without them, all 20 MOPAC models and all 6 xTB models are run. The xTB step scores the capped models written by the MOPAC step, so `MOPAC_MODELS` needs at least one model. `XTB_MODELS=none` skips xTB. The numbers of parallel jobs are set by `PM6_WORKERS`, `XTB_WORKERS` and the other `*_WORKERS` variables at the top of the script. Re-running the script reuses finished jobs.

## Rebuild the tables and figures

The CSV files in `data/` hold the interaction energies and statistics reported in the paper. This regenerates the comparison statistics and the figures from them, without repeating any quantum-chemical calculation:

```bash
SKIP_HARVEST=1 workflow/04_analysis.sh
```

Without `SKIP_HARVEST=1`, the script first collects the energies from the calculation tree in `PPISQM_WORK`.

## Contents

```
scripts/score_model.py
                  one command: frame selection, water selection, scoring
examples/2OOB/    a 40-frame trajectory and its topology, for --example
scripts/          frame selection, model building, scoring, statistics and figures
workflow/         02_score.sh, 03_cpcmx.sh, 04_analysis.sh
data/             result tables (CSV) and dASA interface masks of the twenty complexes
structures/capped_model_1ACB.pdb
                  one capped interface model scored in the paper
structures/chain_reference/
                  the twenty starting complexes; their first chain defines partner A
params/caps/      tleap parameters of the cap residues
params/interface.pymol
                  PyMOL InterfaceResidues script for new dASA masks
```

The trajectories (about 265 GB) and the capped models of every frame are not in this repository. They are available from the authors on request. The benchmark files of Maffucci and Contini are not included either.

## Citation

Please cite the article when using this code, and the programs it runs: xtb, MOPAC, cuby4, MOPAC-ML, CPCM-X and AmberTools. The experimental binding free energies in `data/experimental_data.csv` are those of the structure-based affinity benchmark (Kastritis et al., Protein Sci. 2011, 20, 482; Vreven et al., J. Mol. Biol. 2015, 427, 3031) for the twenty complexes of Maffucci and Contini (J. Chem. Inf. Model. 2016, 56, 1692).
