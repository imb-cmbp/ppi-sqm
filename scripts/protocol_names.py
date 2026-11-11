"""Display names for scoring-protocol arm strings.

Data files and dictionary keys keep the raw arm tokens (e.g. PM6_DHPLUS_COSMO_even10);
these names are used only for figure and table labels. The chemistry of each preset
is defined in interaction.SQM_PRESETS.
"""
from __future__ import annotations

import re

# Longest prefix first, so that e.g. D3H4S_COSMO2 matches before D3H4S_COSMO.
# Each entry is (arm chemistry prefix, full display stem, compact stem).
# Stems already include solvent where that is part of the published method name.
_CHEM: list[tuple[str, str, str]] = [
    # This arm token denotes the D3H4X + chalcogen preset, not the published D3H4S.
    ("PM6_D3H4S_COSMO2",
     "PM6-D3H4X + chalcogen/COSMO2",
     "PM6-D3H4X + chalcogen/COSMO2"),
    ("PM6_D3H4S_COSMO", "PM6-D3H4S/COSMO", "PM6-D3H4S/COSMO"),
    ("PM6_D3H4PAREN_COSMO", "PM6-D3(H4)/COSMO", "PM6-D3(H4)/COSMO"),
    ("PM6_D3H4_COSMO_NOFC",
     "PM6-D3H4/COSMO, scaling off",
     "PM6-D3H4/COSMO, scaling off"),
    ("PM6_D3H4_COSMO_CUBY",
     "PM6-D3H4/COSMO, cuby4, scaling on",
     "PM6-D3H4/COSMO, cuby4"),
    ("PM6_D3H4_COSMO",
     "PM6-D3H4/COSMO, MOPAC built-in",
     "PM6-D3H4/COSMO, MOPAC"),
    ("PM6_D3H4X_COSMO2", "PM6-D3H4X/COSMO2", "PM6-D3H4X/COSMO2"),
    ("PM6_D3H4X", "PM6-D3H4X", "PM6-D3H4X"),
    ("PM6_DHPLUS_COSMO2", "PM6-DH+/COSMO2", "PM6-DH+/COSMO2"),
    ("PM6_DHPLUS_COSMO", "PM6-DH+/COSMO", "PM6-DH+/COSMO"),
    ("PM6_DH2_COSMO", "PM6-DH2/COSMO", "PM6-DH2/COSMO"),
    ("PM6_H4_COSMO_CUBY",
     "PM6-H4/COSMO, cuby4",
     "PM6-H4/COSMO, cuby4"),
    ("PM6_D3_COSMO", "PM6-D3/COSMO", "PM6-D3/COSMO"),
    ("PM6_ML_COSMO2", "PM6-ML/COSMO2", "PM6-ML/COSMO2"),
    ("PM6_ML_COSMO", "PM6-ML/COSMO", "PM6-ML/COSMO"),
    ("PM6_ML", "PM6-ML", "PM6-ML"),
    ("PM6_COSMO2", "PM6/COSMO2", "PM6/COSMO2"),
    ("PM6_COSMO", "PM6/COSMO", "PM6/COSMO"),
    # Geometry-treatment control: same chemistry as scaling-off D3H4/COSMO.
    ("PM6_NOFC",
     "PM6-D3H4/COSMO, scaling off",
     "PM6-D3H4/COSMO, scaling off"),
    ("PM7_COSMO2", "PM7/COSMO2", "PM7/COSMO2"),
    ("PM7", "PM7", "PM7"),
    ("GFN2_GBSA", "GFN2-xTB/GBSA", "GFN2-xTB/GBSA"),
    ("GFN2_ALPB", "GFN2-xTB/ALPB", "GFN2-xTB/ALPB"),
    ("GFN2_COSMO", "GFN2-xTB/ddCOSMO", "GFN2-xTB/ddCOSMO"),
    ("GFN2_CPCMX", "GFN2-xTB/CPCM-X", "GFN2-xTB/CPCM-X"),
    ("GFN1_GBSA", "GFN1-xTB/GBSA", "GFN1-xTB/GBSA"),
    ("GFN1_ALPB", "GFN1-xTB/ALPB", "GFN1-xTB/ALPB"),
    ("GFN1_COSMO", "GFN1-xTB/ddCOSMO", "GFN1-xTB/ddCOSMO"),
]

_BASIN_FULL = {
    "aa": "all-atom basins",
    "bb": "backbone basins",
    "iface_sc": "interface side-chain basins",
    "contacts": "interchain-contact basins",
}
# Short basin tags for crowded axes, without the "basins" suffix.
_BASIN_COMPACT = {
    "aa": "aa",
    "bb": "bb",
    "iface_sc": "iface-sc",
    "contacts": "contact",
}


def _split_arm(arm: str) -> tuple[str, str, str]:
    """Return (chemistry_key, sampling_token, weight_token).

    weight_token is 'wt', 'unw', or '' when the arm has no weight suffix.
    sampling_token is the remainder after chemistry (e.g. 'even10',
    'basin_aa', 'Nwat30', 'basin_aa_mmsp').
    """
    if arm.startswith("MMGBSA_"):
        rest = arm[len("MMGBSA_"):]
        m = re.fullmatch(r"Nwat(\d+)", rest)
        if m:
            return f"MMGBSA_Nwat{m.group(1)}", f"Nwat{m.group(1)}", ""
        m = re.fullmatch(r"basin_(aa|bb|iface_sc|contacts)_(unw|wt)", rest)
        if m:
            return "MMGBSA", f"basin_{m.group(1)}", m.group(2)
        return "MMGBSA", rest, ""

    for chem, _, _ in _CHEM:
        if arm == chem or arm.startswith(chem + "_"):
            rest = arm[len(chem):].lstrip("_")
            w = ""
            if rest.endswith("_unw"):
                rest, w = rest[: -len("_unw")], "unw"
            elif rest.endswith("_wt"):
                rest, w = rest[: -len("_wt")], "wt"
            return chem, rest, w

    return arm, "", ""


def protocol_display_name(
    arm: str,
    *,
    compact: bool = False,
    omit_even10: bool = False,
) -> str:
    """Human-readable label for a protocol arm string.

    Parameters
    ----------
    arm
        Raw CSV arm token (never modified).
    compact
        Shorter qualifier phrasing for crowded figure axes.
    omit_even10
        Drop the even-10 sampling tag (use when every row is even-10).
    """
    chem_key, sampling, weight = _split_arm(arm)

    chem_full = chem_compact = None
    for key, full, short in _CHEM:
        if key == chem_key:
            chem_full, chem_compact = full, short
            break

    if chem_key.startswith("MMGBSA_Nwat"):
        n = chem_key[len("MMGBSA_Nwat"):]
        chem_full = chem_compact = f"Nwat-MM/GBSA, {n} waters"
    elif chem_key == "MMGBSA":
        chem_full = chem_compact = "Nwat-MM/GBSA"

    if chem_full is None:
        # Unknown arm: show the raw token with underscores as spaces.
        chem_full = chem_compact = arm.replace("_", " ")

    chem = chem_compact if compact else chem_full
    parts = [chem]

    basin_map = _BASIN_COMPACT if compact else _BASIN_FULL
    if sampling == "even10":
        if not omit_even10:
            parts.append("even-10")
    elif sampling.startswith("basin_"):
        tag = sampling[len("basin_"):]
        mmsp = False
        if tag.endswith("_mmsp"):
            tag = tag[: -len("_mmsp")]
            mmsp = True
        parts.append(basin_map.get(tag, tag.replace("_", " ") + " basins"))
        if mmsp:
            parts.append("MM/GB→SP" if compact else "MM/GB minimised then SP")
    elif sampling.startswith("Nwat"):
        pass  # already in chemistry for MMGBSA_Nwat*
    elif sampling:
        parts.append(sampling.replace("_", " "))

    if weight == "wt":
        parts.append("pop.-weighted" if compact else "population-weighted")
    elif weight == "unw" and not compact:
        # Compact labels mark only population weighting; unweighted is the default.
        parts.append("unweighted")
    elif weight == "unw":
        pass

    return ", ".join(parts)
