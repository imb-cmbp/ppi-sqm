"""Environment and executable paths for the Amber22 installation at AMBER_HOME."""
from __future__ import annotations

import os
from pathlib import Path

from paths import AMBERHOME as AMBER_HOME


def amber_env(gpu_id=None, extra_path=None):
    """Return (env dict, amber_home str) for AMBER_HOME; ``gpu_id`` sets CUDA_VISIBLE_DEVICES."""
    home = str(AMBER_HOME)
    path_parts = [f"{home}/bin"]
    if extra_path:
        if isinstance(extra_path, str):
            path_parts.append(extra_path)
        else:
            path_parts.extend(extra_path)
    path_parts.append(os.environ.get("PATH", ""))
    env = {
        "AMBERHOME": home,
        "PATH": ":".join(path_parts),
        "LD_LIBRARY_PATH": f"{home}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PYTHONPATH": (
            f"{home}/lib/python3.11/site-packages:"
            f"{os.environ.get('PYTHONPATH', '')}"
        ),
    }
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env, home


def amber_bin(name):
    return str(AMBER_HOME / "bin" / name)
