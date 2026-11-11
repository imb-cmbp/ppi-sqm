"""File locations and external programs used by the scripts.

Every entry can be overridden with the environment variable of the same name,
for example ``PPISQM_WORK=/scratch/ppi python scripts/harvest_scores.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def _exe(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Result tables shipped with the repository and read by the analysis scripts.
DATA = _path("PPISQM_DATA", ROOT / "data")
# Output directory for regenerated figures.
FIGURES = _path("PPISQM_FIGURES", ROOT / "figures")
# Calculation tree: MD systems, snapshots, cuby4/xtb job directories.
WORK = _path("PPISQM_WORK", ROOT / "work")
# Benchmark input from Maffucci & Contini (2016): complexes/ and protocol/.
BENCHMARK = _path("PPISQM_BENCHMARK", ROOT / "benchmark" / "data_for_PPI_Nwat_MMGBSA")
# Parameters for the backbone cap residues (BNC, BCC, BCG, BCB).
CAP_PARAMS = _path("PPISQM_CAP_PARAMS", ROOT / "params" / "caps")
# Input complexes whose chain boundaries define partners A and B.
CHAIN_REF = _path("PPISQM_CHAIN_REF", ROOT / "structures" / "chain_reference")

AMBERHOME = _path("AMBERHOME", Path("/opt/amber22"))
MOPAC_EXE = _exe("MOPAC_EXE", "mopac")
MOPAC_ML_EXE = _exe("MOPAC_ML_EXE", "mopac-ml-cuby")
PM6ML_MODEL = _exe("PM6ML_MODEL", "PM6-ML_correction_seed8_best.ckpt")
CUBY4_DIR = _path("CUBY4_DIR", Path.home() / "cuby4")
RUBY_BIN = _exe("RUBY_BIN", "")
XTB_HOME = _path("XTB_HOME", Path("/usr/local"))
CPX_EXE = _exe("CPX_EXE", "cpx")
CPCMX_DB = _path("CPCMX_DB", Path.home() / "CPCM-X" / "DB" / "xtb")
PYMOL_EXE = _exe("PYMOL_EXE", "pymol")
PROPKA_EXE = _exe("PROPKA_EXE", "propka3")
