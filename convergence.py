from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import BOHR_PER_NM, FORCE_CONV, HARTREE_TO_KJ

ORCA_TOL_E = 5.0e-6
ORCA_TOL_MAX_G = 3.0e-4
ORCA_TOL_RMS_G = 1.0e-4
ORCA_TOL_MAX_D = 4.0e-3
ORCA_TOL_RMS_D = 2.0e-3

OPENMM_TOL_RMSF = 10.0  # kJ/mol/nm


@dataclass
class ConvergenceState:
    mode: str
    norm_ff: float
    prev_pos: np.ndarray | None = None
    prev_energy: float | None = None
    converged: bool = False


def init_convergence_state(mode: str, initial_pos: np.ndarray) -> ConvergenceState:
    pos = np.asarray(initial_pos, dtype=float)
    norm_ff = float(max(1.0, np.sqrt(3 * np.square(pos).mean())))
    return ConvergenceState(mode=mode, norm_ff=norm_ff)


def update_xtb_convergence(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    pos_arr = np.asarray(pos, dtype=float).copy()
    g_eh = np.abs(np.asarray(forces, dtype=float).ravel()) / FORCE_CONV
    g_max_eh = float(np.max(g_eh))
    g_rms_eh = float(np.sqrt(3 * np.mean(g_eh**2)))

    if state.prev_pos is not None:
        disp_bohr = (pos_arr - state.prev_pos) * BOHR_PER_NM
        d_max_bohr = float(np.sqrt(np.max((disp_bohr**2).sum(axis=1))))
        d_rms_bohr = float(np.sqrt(3 * np.mean(disp_bohr**2)))
    else:
        d_max_bohr = float("inf")
        d_rms_bohr = float("inf")

    state.converged = False
    if state.prev_energy is not None:
        state.converged = (
            abs(energy - state.prev_energy) / HARTREE_TO_KJ < ORCA_TOL_E
            and g_max_eh < ORCA_TOL_MAX_G
            and g_rms_eh < ORCA_TOL_RMS_G
            and d_max_bohr < ORCA_TOL_MAX_D
            and d_rms_bohr < ORCA_TOL_RMS_D
        )

    state.prev_pos = pos_arr
    state.prev_energy = float(energy)
    return state.converged


def update_ff_convergence(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    del energy
    pos_arr = np.asarray(pos, dtype=float)
    forces_arr = np.asarray(forces, dtype=float)
    state.converged = (
        np.linalg.norm(forces_arr) / max(1.0, np.linalg.norm(pos_arr))
        < OPENMM_TOL_RMSF / state.norm_ff
    )
    return state.converged


def update_convergence_state(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    if state.mode == "xtb":
        return update_xtb_convergence(state, pos, energy, forces)
    if state.mode == "ff":
        return update_ff_convergence(state, pos, energy, forces)
    raise ValueError(f"Invalid mode: {state.mode}")


def is_converged(state: ConvergenceState) -> bool:
    return state.converged
