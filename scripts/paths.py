"""File locations and external programs used by the scripts.

Every entry can be overridden with the environment variable of the same name,
for example ``PPISQM_WORK=/scratch/ppi python scripts/harvest_scores.py``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def _exe(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _prefix_of(exe: str, default: Path) -> Path:
    """Installation prefix of an executable on PATH (``<prefix>/bin/<exe>``)."""
    found = shutil.which(exe)
    return Path(found).parent.parent if found else default


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
# PyMOL InterfaceResidues script (Jason Vertrees, 2009), used for dASA masks.
INTERFACE_PML = _path("PPISQM_INTERFACE_PML", ROOT / "params" / "interface.pymol")
# Cached dASA residue masks (JSON). The twenty of the paper are shipped here.
MASK_DIR = _path("PPISQM_MASK_DIR", ROOT / "data" / "interface_masks")
# Input complexes whose chain boundaries define partners A and B.
CHAIN_REF = _path("PPISQM_CHAIN_REF", ROOT / "structures" / "chain_reference")

AMBERHOME = _path("AMBERHOME", Path("/opt/amber22"))
MOPAC_EXE = _exe("MOPAC_EXE", "mopac")
MOPAC_ML_EXE = _exe("MOPAC_ML_EXE", str(ROOT / "scripts" / "mopac-ml-cuby"))
PM6ML_MODEL = _exe("PM6ML_MODEL", "PM6-ML_correction_seed8_best.ckpt")
CUBY4_DIR = _path("CUBY4_DIR", Path.home() / "cuby4")
RUBY_BIN = _exe("RUBY_BIN", "")
XTB_HOME = _path("XTB_HOME", _prefix_of("xtb", Path("/usr/local")))
CPX_EXE = _exe("CPX_EXE", "cpx")
CPCMX_DB = _path("CPCMX_DB", Path.home() / "CPCM-X" / "DB" / "xtb")
PYMOL_EXE = _exe("PYMOL_EXE", "pymol")
