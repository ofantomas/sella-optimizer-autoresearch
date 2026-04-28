"""Shared constants, element tables, and I/O helpers.

Single source of truth for unit conversions, element symbol/number maps,
isolated-atom reference energies, and XYZ read/write. Import from here
instead of redefining in each module.
"""

from __future__ import annotations

import numpy as np


# ── unit conversions ──────────────────────────────────────────────────────────

HARTREE_TO_KJ = 2625.4996394799              # kJ/mol per Hartree
BOHR_PER_NM = 1.0 / 0.0529177210544          # Bohr per nm (~18.8973)
ANGSTROM_TO_NM = 0.1                         # nm per Å
EV_TO_HARTREE = 1.0 / 27.211386245988
EV_TO_KJ = HARTREE_TO_KJ * EV_TO_HARTREE
FORCE_CONV = HARTREE_TO_KJ * BOHR_PER_NM     # Eh/a₀ → kJ/(mol·nm)


# ── element maps ──────────────────────────────────────────────────────────────

SYMBOL_TO_Z: dict[str, int] = {
    "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}
Z_TO_SYMBOL: dict[int, str] = {z: sym for sym, z in SYMBOL_TO_Z.items()}


# ── isolated-atom reference energies (GFN2-xTB) ───────────────────────────────
# Source: tblite / xtb atom energies at the GFN2-xTB level, in eV.
# Subtracted from raw molecular energies to yield formation energies.

_ATOM_EV: dict[int, float] = {
    1:  -10.707211250271714,
    6:  -48.84744536025193,
    7:  -71.00681811557804,
    8: -102.57117257606656,
    9: -125.69864273614553,
    15: -64.7034266257264,
    16: -85.66881798270434,
    17: -121.9757218233276,
    35: -110.1609254003934,
    53: -102.84897813915985,
}

ATOM_ENERGIES_HARTREE: dict[int, float] = {z: e * EV_TO_HARTREE for z, e in _ATOM_EV.items()}
ATOM_ENERGIES_KJ: dict[int, float] = {z: e * EV_TO_KJ for z, e in _ATOM_EV.items()}


# ── XYZ I/O ───────────────────────────────────────────────────────────────────

def parse_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an XYZ file.

    Returns
    -------
    atomic_numbers: (N,) int array
    positions_nm:   (N, 3) float array, converted from Ångström to nm

    Raises
    ------
    ValueError
        If the header declares N atoms but the file has fewer than N
        coordinate lines.
    """
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0].strip())
    coord_lines = lines[2 : 2 + n]
    if len(coord_lines) != n:
        raise ValueError(
            f"XYZ at {path!r}: header declares {n} atoms but file has "
            f"{len(coord_lines)} coordinate line(s)"
        )
    z: list[int] = []
    coords: list[list[float]] = []
    for ln in coord_lines:
        parts = ln.split()
        z.append(SYMBOL_TO_Z[parts[0]])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return (
        np.asarray(z, dtype=int),
        np.asarray(coords, dtype=float) * ANGSTROM_TO_NM,
    )


def write_xyz(atomic_numbers, positions_nm, path: str) -> None:
    """Write an XYZ file. Positions are converted from nm to Ångström."""
    pos_ang = np.asarray(positions_nm, dtype=float) / ANGSTROM_TO_NM
    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    with open(path, "w") as fh:
        fh.write(f"{len(atomic_numbers)}\n\n")
        for z, (x, y, z_) in zip(atomic_numbers, pos_ang):
            fh.write(f"{Z_TO_SYMBOL[int(z)]:<2} {x:15.8f} {y:15.8f} {z_:15.8f}\n")
