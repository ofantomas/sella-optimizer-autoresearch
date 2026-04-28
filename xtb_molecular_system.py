"""
XTB-based molecular system for energy minimization.

Energies and forces are computed with GFN2-xTB (or GFN1-xTB) via tblite.
Atomic numbers and the starting geometry are loaded from an XYZ file.
Units: energy in kJ/mol, forces in kJ/(mol·nm).
"""

import os
import glob
import ctypes
import numpy as np

from tblite.interface import Calculator

from utils import BOHR_PER_NM, FORCE_CONV, HARTREE_TO_KJ, parse_xyz


def _set_thread_count(num_threads: int) -> None:
    """Limit in-process threading libraries to the requested thread count.

    os.environ["OMP_NUM_THREADS"] only works when set BEFORE libgomp
    initialises (library load time).  We therefore call the runtime APIs
    directly via ctypes on the already-loaded shared objects.

    Reads /proc/self/maps to find the exact libraries loaded in this process
    and uses RTLD_NOLOAD to get handles to them (not fresh copies).
    """
    num_threads = max(1, int(num_threads))
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    _RTLD_NOLOAD = 0x00004  # dlopen flag: return existing handle or fail

    loaded: dict[str, list[str]] = {"gomp": [], "openblas": [], "mkl": []}
    try:
        with open("/proc/self/maps") as _f:
            for _line in _f:
                _parts = _line.strip().split()
                if not _parts or not _parts[-1].startswith("/"):
                    continue
                _p = _parts[-1]
                if "libgomp" in _p or "libomp" in _p:
                    if _p not in loaded["gomp"]:
                        loaded["gomp"].append(_p)
                elif "libopenblas" in _p:
                    if _p not in loaded["openblas"]:
                        loaded["openblas"].append(_p)
                elif "libmkl" in _p:
                    if _p not in loaded["mkl"]:
                        loaded["mkl"].append(_p)
    except OSError:
        pass

    if not loaded["gomp"]:
        import tblite as _t
        _libs_dir = os.path.join(os.path.dirname(os.path.dirname(_t.__file__)), "tblite.libs")
        loaded["gomp"] = glob.glob(os.path.join(_libs_dir, "libgomp*.so*")) + ["libgomp.so.1"]

    for _path in loaded["gomp"]:
        try:
            _lib = ctypes.CDLL(_path, mode=_RTLD_NOLOAD)
            _lib.omp_set_num_threads(num_threads)
        except (OSError, AttributeError):
            pass

    for _path in loaded["openblas"]:
        try:
            _lib = ctypes.CDLL(_path, mode=_RTLD_NOLOAD)
            _lib.openblas_set_num_threads(num_threads)
        except (OSError, AttributeError):
            pass

    for _path in loaded["mkl"]:
        try:
            _lib = ctypes.CDLL(_path, mode=_RTLD_NOLOAD)
            _lib.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))
        except (OSError, AttributeError):
            pass


def _force_single_thread() -> None:
    _set_thread_count(1)


# ── main class ────────────────────────────────────────────────────────────────

class XTBMolecularSystem:
    """
    xTB-based molecular system for energy and force calculations.

    Atomic numbers and starting positions are parsed from an XYZ file.

    Args:
        xyz_path:  Path to an XYZ file. Element symbols + Ångström coords;
            positions are converted to nm on load.
        method:    xTB Hamiltonian — "GFN2-xTB" (default) or "GFN1-xTB".
        accuracy:  SCF convergence threshold (default 0.2). Increase (e.g. 3.0)
            to trade accuracy for speed in optimizer inner loops.
        verbose:   tblite verbosity flag.
        num_threads: threads to use for xTB calculations.
    """

    def __init__(
        self,
        xyz_path: str,
        method: str = "GFN2-xTB",
        accuracy: float = 0.2,
        verbose: bool = False,
        num_threads: int = 1,
    ):
        self._atomic_numbers, self._initial_positions = parse_xyz(xyz_path)
        self.method = method
        self.accuracy = accuracy
        self.verbose = verbose
        self.num_threads = max(1, int(num_threads))

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers

    @property
    def initial_positions(self) -> np.ndarray:
        return self._initial_positions

    def compute(self, positions, **kwargs) -> tuple[float, np.ndarray]:
        """
        Compute energy and forces with xTB.

        Args:
            positions: array-like (N, 3) in nm.

        Returns:
            (energy, forces): energy in kJ/mol, forces in kJ/(mol·nm).
        """
        pos_bohr = np.asarray(positions, dtype=float) * BOHR_PER_NM

        try:
            calc = Calculator(
                self.method,
                self._atomic_numbers,
                pos_bohr,
                charge=0,
            )
            calc.set("accuracy", self.accuracy)
            calc.set("verbosity", 1 if self.verbose else 0)
            _set_thread_count(self.num_threads)

            res = calc.singlepoint()

            energy = float(res.get("energy")) * HARTREE_TO_KJ
            gradient = np.array(res.get("gradient")).reshape(-1, 3)
            forces = -gradient * FORCE_CONV
            return energy, np.array(forces, dtype=float)
        except Exception:
            return float("inf"), np.zeros_like(positions, dtype=float)


# Alias so callers can do: from xtb_molecular_system import MolecularSystem
MolecularSystem = XTBMolecularSystem
