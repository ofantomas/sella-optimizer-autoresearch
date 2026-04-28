from __future__ import division
import os
import warnings
from typing import Tuple, Callable, Iterator, Union, TypeVar, Optional, List, Dict, Type
from itertools import product, combinations, combinations_with_replacement as cwr
from functools import partialmethod
import numpy as np
from scipy.linalg import eigh, lstsq
from scipy.sparse.linalg import LinearOperator
from scipy.integrate import LSODA
from ase import Atom, Atoms, units
from ase.cell import Cell
from ase.geometry import complete_cell, minkowski_reduce
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.optimize import Optimizer
import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, custom_jvp, vmap, jvp, device_get

_NM_TO_ANG = 10.0
_KJ_MOL_TO_EV = 1.0 / 96.48530749925793
_KJ_MOL_NM_TO_EV_ANG = _KJ_MOL_TO_EV / _NM_TO_ANG
_cache_dir = os.path.expanduser("~/.cache/sella/jax_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)
jax.config.update("jax_enable_x64", True)
try:
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except (AttributeError, ValueError):
    pass


def symmetrize_Y2(S, Y):
    _, nvecs = S.shape
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        RHS = np.linalg.lstsq(
            STS[:i, :i], YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i], rcond=None
        )[0]
        dY[:, i] = -S[:, :i] @ RHS
        dYTS[i, :] = -STS[:, :i] @ RHS
    return dY


def symmetrize_Y(S, Y, symm):
    if symm is None or S.shape[1] == 1:
        return Y
    elif symm == 0:
        return Y + S @ lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 1:
        return Y + Y @ lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 2:
        return Y + symmetrize_Y2(S, Y)
    else:
        raise ValueError("Unknown symmetrization method {}".format(symm))


def update_H(B, S, Y, method="TS-BFGS", symm=2, lams=None, vecs=None):
    if len(S.shape) == 1:
        if np.linalg.norm(S) < 1e-08:
            return B
        S = S[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    Ytilde = symmetrize_Y(S, Y, symm)
    if B is None:
        thetas, _ = eigh(S.T @ Ytilde)
        thetas_abs = np.abs(thetas)
        thetas_abs = np.maximum(thetas_abs, 1e-12)
        lam0 = np.exp(np.average(np.log(thetas_abs)))
        d, _ = S.shape
        B = lam0 * np.eye(d)
    if lams is None or vecs is None:
        lams, vecs = eigh(B)
    if method != "TS-BFGS":
        raise ValueError("sella_minimal only supports TS-BFGS Hessian updates.")
    Bplus = _MS_TS_BFGS(B, S, Ytilde, lams, vecs)
    Bplus += B
    Bplus -= np.tril(Bplus.T - Bplus, -1).T
    return Bplus


def _MS_TS_BFGS(B, S, Y, lams, vecs):
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    U = lstsq((X1 + X2) @ S, X1 + X2)[0].T
    UJT = U @ J.T
    return UJT + UJT.T - U @ (J.T @ S) @ U.T


class ApproximateHessian(LinearOperator):
    def __init__(
        self,
        dim: int,
        ncart: int,
        B0: np.ndarray = None,
        update_method: str = "TS-BFGS",
        symm: int = 2,
        initialized: bool = False,
    ) -> None:
        self.dim = dim
        self.ncart = ncart
        self.shape = (self.dim, self.dim)
        self.dtype = np.float64
        self.update_method = update_method
        self.symm = symm
        self.initialized = initialized
        self._evals = None
        self._evecs = None
        self._eigen_computed = False
        self.set_B(B0)

    def _ensure_eigen_computed(self):
        if not self._eigen_computed and self.B is not None:
            self._evals, self._evecs = eigh(self.B)
            self._eigen_computed = True

    @property
    def evals(self):
        self._ensure_eigen_computed()
        return self._evals

    @evals.setter
    def evals(self, value):
        self._evals = value
        if value is None:
            self._eigen_computed = False

    @property
    def evecs(self):
        self._ensure_eigen_computed()
        return self._evecs

    @evecs.setter
    def evecs(self, value):
        self._evecs = value
        if value is None:
            self._eigen_computed = False

    def set_B(self, target):
        if target is None:
            self.B = None
            self._evals = None
            self._evecs = None
            self._eigen_computed = False
            self.initialized = False
            return
        elif np.isscalar(target):
            target = target * np.eye(self.dim)
        else:
            self.initialized = True
        assert target.shape == self.shape
        self.B = target
        self._eigen_computed = False

    def update(self, dx, dg):
        if self.B is None:
            B = np.zeros(self.shape, dtype=self.dtype)
        else:
            B = self.B.copy()
        if not self.initialized:
            self.initialized = True
            dx_cart = dx[: self.ncart]
            dg_cart = dg[: self.ncart]
            B[: self.ncart, : self.ncart] = update_H(
                None,
                dx_cart,
                dg_cart,
                method=self.update_method,
                symm=self.symm,
                lams=None,
                vecs=None,
            )
            self.set_B(B)
            return
        self.set_B(
            update_H(
                B,
                dx,
                dg,
                method=self.update_method,
                symm=self.symm,
                lams=self.evals,
                vecs=self.evecs,
            )
        )

    def project(self, U):
        m, n = U.shape
        assert m == self.dim
        if self.B is None:
            Bproj = None
        else:
            Bproj = U.T @ self.B @ U
        return ApproximateHessian(n, 0, Bproj, self.update_method, self.symm)

    def asarray(self):
        if self.B is not None:
            return self.B
        return np.eye(self.dim)

    def _matvec(self, v):
        if self.B is None:
            return v
        return self.B @ v

    def _rmatvec(self, v):
        return self.matvec(v)

    def _matmat(self, X):
        if self.B is None:
            return X
        return self.B @ X

    def _rmatmat(self, X):
        return self.matmat(X)

    def __add__(self, other):
        initialized = self.initialized
        if isinstance(other, ApproximateHessian):
            initialized = initialized and other.initialized
            other = other.B
        if not self.initialized or other is None:
            tot = None
            initialized = False
        else:
            tot = self.B + other
        return ApproximateHessian(
            self.dim,
            self.ncart,
            tot,
            self.update_method,
            self.symm,
            initialized=initialized,
        )


class SparseInternalHessian(LinearOperator):
    dtype = np.float64

    def __init__(self, natoms: int, indices: List[int], vals: np.ndarray) -> None:
        self.natoms = natoms
        self.shape = (3 * self.natoms, 3 * self.natoms)
        self.indices = np.asarray(indices)
        self.vals = np.asarray(vals)

    def asarray(self) -> np.ndarray:
        H = np.zeros((self.natoms, self.natoms, 3, 3))
        idx = self.indices
        n = len(idx)
        if n == 0:
            return H.transpose(0, 2, 1, 3).reshape(self.shape)
        idx_a, idx_b = np.meshgrid(idx, idx, indexing="ij")
        linear_idx = idx_a * self.natoms + idx_b
        H_flat = H.reshape(self.natoms * self.natoms, 3, 3)
        vals_flat = self.vals.transpose(0, 2, 1, 3).reshape(n * n, 3, 3)
        np.add.at(H_flat, linear_idx.ravel(), vals_flat)
        return H.transpose(0, 2, 1, 3).reshape(self.shape)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        vi = v.reshape((self.natoms, 3))
        w = np.zeros_like(vi)
        idx = self.indices
        vi_sub = vi[idx]
        result = np.einsum("aibj,bj->ai", self.vals, vi_sub)
        np.add.at(w, idx, result)
        return w.ravel()

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        return self._matvec(v)


class SparseInternalHessians:
    def __init__(self, hessians: List[SparseInternalHessian], ndof: int):
        self.hessians = hessians
        self.natoms = ndof // 3
        self.shape = (len(self.hessians), ndof, ndof)
        self._prepare_batched_data()

    def _prepare_batched_data(self):
        by_size = {}
        for i, h in enumerate(self.hessians):
            n = len(h.indices)
            if n not in by_size:
                by_size[n] = {"orig_idx": [], "indices": [], "vals": []}
            by_size[n]["orig_idx"].append(i)
            by_size[n]["indices"].append(h.indices)
            by_size[n]["vals"].append(h.vals)
        i_idx, j_idx = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        i_flat = i_idx.ravel()
        j_flat = j_idx.ravel()
        self._batched_rdot = {}
        self._batched_ldot = {}
        for size, data in by_size.items():
            orig_idx = np.array(data["orig_idx"])
            indices = np.array(data["indices"])
            vals = np.array(data["vals"])
            batch = len(orig_idx)
            self._batched_rdot[size] = {
                "orig_idx": orig_idx,
                "indices": indices,
                "vals": vals,
            }
            n_pairs = size * size
            a_local, b_local = np.meshgrid(
                np.arange(size), np.arange(size), indexing="ij"
            )
            a_local = a_local.ravel()
            b_local = b_local.ravel()
            row_atoms = indices[:, a_local]
            col_atoms = indices[:, b_local]
            row_atoms = np.repeat(row_atoms, 9, axis=1)
            col_atoms = np.repeat(col_atoms, 9, axis=1)
            i_full = np.tile(i_flat, (batch, n_pairs))
            j_full = np.tile(j_flat, (batch, n_pairs))
            vals_reordered = vals.transpose(0, 1, 3, 2, 4)
            vals_flat_arr = vals_reordered.reshape(batch, -1)
            self._batched_ldot[size] = {
                "orig_idx": orig_idx,
                "vals_flat": vals_flat_arr,
                "row_atoms": row_atoms,
                "col_atoms": col_atoms,
                "i_full": i_full,
                "j_full": j_full,
            }

    def asarray(self) -> np.ndarray:
        return np.array([hess.asarray() for hess in self.hessians])

    def __array__(self, dtype=None):
        arr = self.asarray()
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def ldot(self, v: np.ndarray) -> np.ndarray:
        M = np.zeros((self.natoms, 3, self.natoms, 3))
        for size, data in self._batched_ldot.items():
            orig_idx = data["orig_idx"]
            vals_flat = data["vals_flat"]
            row_atoms = data["row_atoms"]
            col_atoms = data["col_atoms"]
            i_full = data["i_full"]
            j_full = data["j_full"]
            weights = v[orig_idx]
            weighted = vals_flat * weights[:, None]
            np.add.at(
                M,
                (row_atoms.ravel(), i_full.ravel(), col_atoms.ravel(), j_full.ravel()),
                weighted.ravel(),
            )
        return M.reshape(self.shape[1:])

    def rdot(self, v: np.ndarray) -> np.ndarray:
        vi = v.reshape((self.natoms, 3))
        M = np.zeros((self.shape[0], self.natoms, 3))
        for size, data in self._batched_rdot.items():
            orig_idx = data["orig_idx"]
            idx = data["indices"]
            vals = data["vals"]
            vi_sub = vi[idx]
            result = np.einsum("naibj,nbj->nai", vals, vi_sub)
            row_idx = np.repeat(orig_idx, size)
            col_idx = idx.ravel()
            result_flat = result.reshape(-1, 3)
            np.add.at(M, (row_idx, col_idx), result_flat)
        return M.reshape(self.shape[0], -1)

    def ddot(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        w = np.zeros(self.shape[0])
        for i, hessian in enumerate(self.hessians):
            w[i] = u @ hessian @ v
        return w


class LightAtoms:
    __slots__ = ("positions", "cell")

    def __init__(self, positions: np.ndarray, cell: np.ndarray) -> None:
        self.positions = positions
        self.cell = cell


def _bond_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])


def _angle_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


def _dihedral_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


def _make_batched_ops(func):
    return (
        jit(vmap(func, in_axes=(0, 0))),
        jit(vmap(grad(func, argnums=0), in_axes=(0, 0))),
        jit(vmap(jacfwd(grad(func, argnums=0), argnums=0), in_axes=(0, 0))),
    )


def _make_hvp_single(func):
    def hvp_single(
        pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray
    ) -> jnp.ndarray:
        primals = (pos, tvec)
        tangents = (tangent, jnp.zeros_like(tvec))
        _, hvp_result = jvp(grad(func, argnums=0), primals, tangents)
        return hvp_result

    return hvp_single


_bond_value_batched, _bond_grad_batched, _bond_hess_batched = _make_batched_ops(
    _bond_value
)
_angle_value_batched, _angle_grad_batched, _angle_hess_batched = _make_batched_ops(
    _angle_value
)
_dihedral_value_batched, _dihedral_grad_batched, _dihedral_hess_batched = (
    _make_batched_ops(_dihedral_value)
)
_bond_hvp_batched = jit(vmap(_make_hvp_single(_bond_value), in_axes=(0, 0, 0)))
_angle_hvp_batched = jit(vmap(_make_hvp_single(_angle_value), in_axes=(0, 0, 0)))
_dihedral_hvp_batched = jit(vmap(_make_hvp_single(_dihedral_value), in_axes=(0, 0, 0)))
BLOCK_SIZE = 64
IVec = Tuple[int, int, int]


class NoValidInternalError(ValueError):
    pass


class DuplicateInternalError(ValueError):
    pass


class DuplicateConstraintError(DuplicateInternalError):
    pass


def _gradient(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(grad(func, argnums=0))


def _hessian(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float],
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(jacfwd(jacrev(func, argnums=0), argnums=0))


class Coordinate:
    nindices = None
    kwargs = None

    def __init__(self, indices: Tuple[int, ...]) -> None:
        if self.nindices is not None:
            assert len(indices) == self.nindices
        self.indices = np.array(indices, dtype=np.int32)
        self.kwargs = dict()

    def reverse(self) -> "Coordinate":
        raise NotImplementedError

    def __eq__(self, other: "Coordinate") -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.indices) != len(other.indices):
            return False
        if np.all(self.indices == other.indices):
            return True
        return False

    def __add__(self, other: "Coordinate") -> "Coordinate":
        raise NotImplementedError

    def split(self) -> Tuple["Coordinate", "Coordinate"]:
        raise NotImplementedError

    def __repr__(self) -> str:
        out = [f"indices={self.indices}"]
        out += [f"{key}={val}" for key, val in self.kwargs.items()]
        str_out = ", ".join(out)
        return f"{self.__class__.__name__}({str_out})"

    @staticmethod
    def _eval0(pos: jnp.ndarray, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(atoms.positions[self.indices], **self.kwargs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(atoms.positions[self.indices], **self.kwargs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        return np.array(self._eval2(atoms.positions[self.indices], **self.kwargs))

    def _check_derivative(
        self, atoms: Atoms, delta: float, atol: float, order: int
    ) -> bool:
        if order == 1:
            derivative = "Gradient"
            f0 = self.calc
            f1 = self.calc_gradient
        elif order == 2:
            derivative = "Hessian"
            f0 = self.calc_gradient
            f1 = self.calc_hessian
        else:
            raise ValueError(f"Order {order} gradients are not implemented")
        atoms0 = atoms.copy()
        g_ref = f1(atoms0)
        g_numer = np.zeros_like(g_ref)
        atoms = atoms0.copy()
        for i, idx in enumerate(self.indices):
            for j in range(3):
                atoms.positions[idx, j] = atoms0.positions[idx, j] + delta
                fplus = f0(atoms)
                atoms.positions[idx, j] = atoms0.positions[idx, j] - delta
                fminus = f0(atoms)
                g_numer[i, j] = (fplus - fminus) / (2 * delta)
                atoms.positions[idx, j] = atoms0.positions[idx, j]
        if np.max(np.abs(g_numer - g_ref)) > atol:
            warnings.warn(f"{derivative}s for {self} failed numerical test!")
            return False
        return True

    def check_gradient(
        self, atoms: Atoms, delta: float = 0.0001, atol: float = 1e-06
    ) -> bool:
        return self._check_derivative(atoms, delta, atol, order=1)

    def check_hessian(
        self, atoms: Atoms, delta: float = 0.0001, atol: float = 1e-06
    ) -> bool:
        return self._check_derivative(atoms, delta, atol, order=2)


class Internal(Coordinate):
    union = None
    diff = None

    def __init__(
        self, indices: Tuple[int, ...], ncvecs: Tuple[IVec, ...] = None
    ) -> None:
        Coordinate.__init__(self, indices)
        if self.nindices is not None:
            if ncvecs is None:
                ncvecs = np.zeros((self.nindices - 1, 3), dtype=np.int32)
            else:
                ncvecs = np.asarray(ncvecs).reshape((self.nindices - 1, 3))
        else:
            if ncvecs is not None:
                raise ValueError(
                    "{} does not support ncvecs".format(self.__class__.__name__)
                )
            ncvecs = np.empty((0, 3), dtype=np.int32)
        self.kwargs["ncvecs"] = ncvecs

    def reverse(self) -> "Internal":
        return self.__class__(self.indices[::-1], -self.kwargs["ncvecs"][::-1])

    def __eq__(self, other: object) -> bool:
        if not Coordinate.__eq__(self, other):
            return False
        srev = self.reverse()
        if not Coordinate.__eq__(srev, other):
            return False
        if np.all(self.kwargs["ncvecs"] == other.kwargs["ncvecs"]):
            return True
        if np.all(srev.kwargs["ncvecs"] == other.kwargs["ncvecs"]):
            return True
        return False

    def __add__(self, other: object) -> "Internal":
        if self.union is None:
            return NotImplemented
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            raise NoValidInternalError(
                "Cannot add {} object to itself.".format(self.__class__.__name__)
            )
        for s, o in product([self, self.reverse()], [other, other.reverse()]):
            if np.all(s.indices[1:] == o.indices[:-1]) and np.all(
                s.kwargs["ncvecs"][1:] == o.kwargs["ncvecs"][:-1]
            ):
                new_indices = [*s.indices, o.indices[-1]]
                new_ncvecs = [*s.kwargs["ncvecs"], o.kwargs["ncvecs"][-1]]
                return self.union(new_indices, new_ncvecs)
        raise NoValidInternalError(
            "{} indices do not overlap!".format(self.__class__.__name__)
        )

    def split(self) -> Tuple["Internal", "Internal"]:
        if self.diff is None:
            raise RuntimeError(
                "Don't know how to split a {}!".format(self.__class__.__name__)
            )
        return (
            self.diff(self.indices[:-1], self.kwargs["ncvecs"][:-1]),
            self.diff(self.indices[1:], self.kwargs["ncvecs"][1:]),
        )

    @staticmethod
    def _eval0(pos: jnp.ndarray, tvecs: jnp.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(pos: jnp.ndarray, tvecs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(pos: jnp.ndarray, tvecs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        tvecs = jnp.asarray(self.kwargs["ncvecs"] @ atoms.cell, dtype=np.float64)
        return float(self._eval0(atoms.positions[self.indices], tvecs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(self.kwargs["ncvecs"] @ atoms.cell, dtype=np.float64)
        return np.array(self._eval1(atoms.positions[self.indices], tvecs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(self.kwargs["ncvecs"] @ atoms.cell, dtype=np.float64)
        return np.array(self._eval2(atoms.positions[self.indices], tvecs))


def _translation(pos: jnp.ndarray, dim: int) -> float:
    return pos[:, dim].mean()


class Translation(Coordinate):
    def __init__(self, indices: Tuple[int, ...], dim: int) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs["dim"] = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs["dim"] != other.kwargs["dim"]:
            return False
        if set(self.indices) != set(other.indices):
            return False
        return True

    _eval0 = staticmethod(jit(_translation))
    _eval1 = staticmethod(_gradient(_translation))
    _eval2 = staticmethod(_hessian(_translation))


@custom_jvp
def eigh_rightmost(X):
    return jnp.linalg.eigh((X + X.T) / 2.0)[1][:, -1]


@eigh_rightmost.defjvp
def eigh_rightmost_jvp(primals, tangents):
    (X,) = primals
    X = (X + X.T) / 2.0
    (dX,) = tangents
    dX = (dX + dX.T) / 2.0
    dim = X.shape[0]
    ws, vecs = jnp.linalg.eigh(X)
    v = vecs[:, -1]
    ldot = v.T @ dX @ v
    return (v, jnp.linalg.pinv(ws[-1] * jnp.eye(dim) - X) @ (dX @ v - ldot * v))


def _rotation_q(pos: jnp.ndarray, refpos: jnp.ndarray) -> float:
    dx = pos - pos.mean(0)
    R = dx.T @ refpos
    Rtr = jnp.trace(R)
    Ftop = jnp.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    F = jnp.block([[Rtr, Ftop[None, :]], [Ftop[:, None], -Rtr * jnp.eye(3) + R + R.T]])
    q = eigh_rightmost(F)
    return q * jnp.where(q[0] >= 0, 1.0, -1.0)


def _asinc_naive(x):
    return jnp.arccos(x) / jnp.sqrt(1 - x**2)


def _asinc_taylor(x):
    y = x - 1
    return (
        1
        - y / 3
        + 2 * y**2 / 15
        - 2 * y**3 / 35
        + 8 * y**4 / 315
        - 8 * y**5 / 693
        + 16 * y**6 / 3003
        - 16 * y**7 / 6435
        + 128 * y**8 / 109395
        - 128 * y**9 / 230945
    )


def asinc(x):
    return jnp.where(
        x < 0.97, _asinc_naive(jnp.where(x < 0.97, x, 0.97)), _asinc_taylor(x)
    )


def _rotation(pos: jnp.ndarray, axis: int, refpos: jnp.ndarray) -> float:
    q = _rotation_q(pos, refpos)
    return 2 * q[axis + 1] * asinc(q[0])


def _rotation_hvp(
    pos: jnp.ndarray, axis: int, refpos: jnp.ndarray, tangent: jnp.ndarray
) -> jnp.ndarray:
    primals = (pos,)
    tangents = (tangent,)
    _, hvp_result = jvp(
        lambda p: grad(_rotation, argnums=0)(p, axis, refpos), primals, tangents
    )
    return hvp_result


_rotation_hvp_jit = jit(_rotation_hvp, static_argnums=(1,))


class Rotation(Coordinate):
    def __init__(self, indices: Tuple[int, ...], axis: int, refpos: np.ndarray) -> None:
        assert len(indices) >= 2
        Coordinate.__init__(self, indices)
        self.kwargs["axis"] = axis
        self.kwargs["refpos"] = refpos.copy() - refpos.mean(0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs["axis"] != other.kwargs["axis"]:
            return False
        if len(self.indices) != len(other.indices):
            return False
        if set(self.indices) != set(other.indices):
            return False
        if np.any(self.kwargs["refpos"] != other.kwargs["refpos"]):
            return False
        return True

    _eval0 = staticmethod(jit(_rotation))
    _eval1 = staticmethod(jit(jacfwd(_rotation, argnums=0)))
    _eval2 = staticmethod(jit(jacfwd(jacfwd(_rotation, argnums=0), argnums=0)))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        result = np.array(self._eval2(atoms.positions[self.indices], **self.kwargs))
        if np.any(np.isnan(result)):
            np.nan_to_num(result, copy=False)
        return result


def _displacement(pos: jnp.ndarray, refpos: jnp.ndarray, W: jnp.ndarray) -> float:
    dx = (pos - refpos).ravel()
    return dx @ W @ dx


class Displacement(Coordinate):
    def __init__(self, indices: np.ndarray, refpos: np.ndarray, W: np.ndarray) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs["refpos"] = refpos.copy()
        self.kwargs["W"] = W.copy()

    def __eq__(self, other: Coordinate) -> bool:
        if not Coordinate.__eq__(self, other):
            return False
        return np.all(self.kwargs["refpos"] == other.kwargs["refpos"])

    _eval0 = staticmethod(jit(_displacement))
    _eval1 = staticmethod(jit(_gradient(_displacement)))
    _eval2 = staticmethod(jit(_hessian(_displacement)))


def _bond(pos: jnp.ndarray, tvecs: jnp.ndarray) -> float:
    return jnp.linalg.norm(pos[1] - pos[0] + tvecs[0])


class Bond(Internal):
    nindices = 2
    _eval0 = staticmethod(jit(_bond))
    _eval1 = staticmethod(_gradient(_bond))
    _eval2 = staticmethod(_hessian(_bond))

    def calc_vec(self, atoms: Atoms) -> np.ndarray:
        tvecs = np.asarray(self.kwargs["ncvecs"] @ atoms.cell, dtype=np.float64)
        i, j = self.indices
        return atoms.positions[j] - atoms.positions[i] + tvecs[0]


def _angle(pos: jnp.ndarray, tvecs: jnp.ndarray) -> float:
    dx1 = -(pos[1] - pos[0] + tvecs[0])
    dx2 = pos[2] - pos[1] + tvecs[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)


class Angle(Internal):
    nindices = 3
    _eval0 = staticmethod(jit(_angle))
    _eval1 = staticmethod(_gradient(_angle))
    _eval2 = staticmethod(_hessian(_angle))


def _dihedral(pos: jnp.ndarray, tvecs: jnp.ndarray) -> float:
    dx1 = pos[1] - pos[0] + tvecs[0]
    dx2 = pos[2] - pos[1] + tvecs[1]
    dx3 = pos[3] - pos[2] + tvecs[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)


class Dihedral(Internal):
    nindices = 4
    _eval0 = staticmethod(jit(_dihedral))
    _eval1 = staticmethod(_gradient(_dihedral))
    _eval2 = staticmethod(_hessian(_dihedral))


Bond.union = Angle
Angle.union = Dihedral
Angle.diff = Bond
Dihedral.diff = Angle


def make_internal(
    name: str,
    fun: Callable[..., float],
    nindices: int,
    use_jit: bool = True,
    jac: Callable[..., jnp.ndarray] = None,
    hess: Callable[..., jnp.ndarray] = None,
    **kwargs,
) -> Type[Coordinate]:
    if jac is None:
        jac = _gradient(fun)
    if hess is None:
        hess = _hessian(fun)
    if use_jit:
        fun = jit(fun)
        jac = jit(jac)
        hess = jit(hess)
    return type(
        name,
        (Coordinate,),
        dict(
            nindices=nindices,
            kwargs=kwargs,
            _eval0=staticmethod(fun),
            _eval1=staticmethod(jac),
            _eval2=staticmethod(hess),
        ),
    )


class BaseInternals:
    _names = ("translations", "bonds", "angles", "dihedrals", "other", "rotations")

    def __init__(
        self, atoms: Atoms, dummies: Atoms = None, dinds: np.ndarray = None
    ) -> None:
        self.atoms = atoms
        self._lastpos = None
        self._cache = dict()
        self._cache_version = 0
        if dummies is None:
            if dinds is not None:
                raise ValueError('"dinds" provided, but no "dummies"!')
            dummies = Atoms()
            dinds = -np.ones(len(self.atoms), dtype=np.int32)
        else:
            if dinds is None:
                raise ValueError('"dummies" provided, but no "dinds"!')
            ndum = len(dummies)
            ndind = np.sum(dinds >= 0)
            if ndum != ndind:
                raise ValueError(
                    "{} dummy atoms were provided, but only {} dummy indices!".format(
                        ndum, ndind
                    )
                )
        self.dummies = dummies
        self.dinds = dinds
        self._natoms = len(atoms)
        self.internals = {key: [] for key in self._names}
        self._active = {key: [] for key in self._names}
        self.cell = None
        self.rcell = None
        self.op = None
        self._batched_arrays_valid = False

    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    @property
    def ntrans(self) -> int:
        return sum(self._active["translations"])

    @property
    def nbonds(self) -> int:
        return sum(self._active["bonds"])

    @property
    def nangles(self) -> int:
        return sum(self._active["angles"])

    @property
    def ndihedrals(self) -> int:
        return sum(self._active["dihedrals"])

    @property
    def nother(self) -> int:
        return sum(self._active["other"])

    @property
    def nrotations(self) -> int:
        return sum(self._active["rotations"])

    @property
    def _active_mask(self) -> List[bool]:
        active = []
        for name in self._names:
            active += self._active[name]
        return active

    @property
    def _active_indices(self) -> List[int]:
        return [idx for idx, active in enumerate(self._active_mask) if active]

    @property
    def nint(self) -> int:
        return len(self._active_indices)

    @property
    def ndof(self) -> int:
        return 3 * (self._natoms + len(self.dummies))

    @property
    def all_positions(self) -> np.ndarray:
        if self.ndummies > 0:
            return np.vstack([self.atoms.positions, self.dummies.positions])
        return self.atoms.positions

    @property
    def all_atoms(self) -> Atoms:
        return self.atoms + self.dummies

    @property
    def light_atoms(self) -> LightAtoms:
        cell = (
            self.atoms.cell.array
            if hasattr(self.atoms.cell, "array")
            else np.asarray(self.atoms.cell)
        )
        return LightAtoms(self.all_positions, cell)

    def _cache_check(self) -> None:
        current_pos = self.all_positions
        if self._lastpos is None or np.any(current_pos != self._lastpos):
            self._cache = dict()
            self._lastpos = current_pos.copy()
            self._cache_version += 1

    def _build_batched_arrays(self) -> None:
        if self._batched_arrays_valid:
            return

        def pad_to_block(n: int) -> int:
            return (n + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

        bonds = self.internals["bonds"]
        n_bonds = len(bonds)
        if n_bonds > 0:
            n_bonds_padded = pad_to_block(n_bonds)
            self._bond_indices = np.array([b.indices for b in bonds], dtype=np.int32)
            self._bond_ncvecs = np.array(
                [b.kwargs["ncvecs"] for b in bonds], dtype=np.int32
            )
            self._bond_indices_padded = np.zeros((n_bonds_padded, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.zeros((n_bonds_padded, 1, 3), dtype=np.int32)
            self._bond_indices_padded[:n_bonds] = self._bond_indices
            self._bond_ncvecs_padded[:n_bonds] = self._bond_ncvecs
            self._bond_mask = np.zeros(n_bonds_padded, dtype=np.float64)
            self._bond_mask[:n_bonds] = 1.0
            self._n_bonds_actual = n_bonds
        else:
            self._bond_indices = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_indices_padded = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_mask = np.empty(0, dtype=np.float64)
            self._n_bonds_actual = 0
        angles = self.internals["angles"]
        n_angles = len(angles)
        if n_angles > 0:
            n_angles_padded = pad_to_block(n_angles)
            self._angle_indices = np.array([a.indices for a in angles], dtype=np.int32)
            self._angle_ncvecs = np.array(
                [a.kwargs["ncvecs"] for a in angles], dtype=np.int32
            )
            self._angle_indices_padded = np.zeros((n_angles_padded, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.zeros(
                (n_angles_padded, 2, 3), dtype=np.int32
            )
            self._angle_indices_padded[:n_angles] = self._angle_indices
            self._angle_ncvecs_padded[:n_angles] = self._angle_ncvecs
            self._angle_mask = np.zeros(n_angles_padded, dtype=np.float64)
            self._angle_mask[:n_angles] = 1.0
            self._n_angles_actual = n_angles
        else:
            self._angle_indices = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_indices_padded = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_mask = np.empty(0, dtype=np.float64)
            self._n_angles_actual = 0
        dihedrals = self.internals["dihedrals"]
        n_dihedrals = len(dihedrals)
        if n_dihedrals > 0:
            n_dihedrals_padded = pad_to_block(n_dihedrals)
            self._dihedral_indices = np.array(
                [d.indices for d in dihedrals], dtype=np.int32
            )
            self._dihedral_ncvecs = np.array(
                [d.kwargs["ncvecs"] for d in dihedrals], dtype=np.int32
            )
            self._dihedral_indices_padded = np.zeros(
                (n_dihedrals_padded, 4), dtype=np.int32
            )
            self._dihedral_ncvecs_padded = np.zeros(
                (n_dihedrals_padded, 3, 3), dtype=np.int32
            )
            self._dihedral_indices_padded[:n_dihedrals] = self._dihedral_indices
            self._dihedral_ncvecs_padded[:n_dihedrals] = self._dihedral_ncvecs
            self._dihedral_mask = np.zeros(n_dihedrals_padded, dtype=np.float64)
            self._dihedral_mask[:n_dihedrals] = 1.0
            self._n_dihedrals_actual = n_dihedrals
        else:
            self._dihedral_indices = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_indices_padded = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs_padded = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_mask = np.empty(0, dtype=np.float64)
            self._n_dihedrals_actual = 0
        self._batched_arrays_valid = True

    def _get_cached_tvecs(self, cell: np.ndarray) -> Dict[str, np.ndarray]:
        cell_hash = cell.tobytes()
        if (
            hasattr(self, "_tvecs_cache")
            and self._tvecs_cache.get("cell_hash") == cell_hash
        ):
            return self._tvecs_cache["tvecs"]
        self._build_batched_arrays()
        tvecs = {}
        if len(self._bond_indices) > 0:
            tvecs["bonds"] = self._bond_ncvecs @ cell
        else:
            tvecs["bonds"] = np.empty((0, 1, 3), dtype=np.float64)
        if len(self._angle_indices) > 0:
            tvecs["angles"] = self._angle_ncvecs @ cell
        else:
            tvecs["angles"] = np.empty((0, 2, 3), dtype=np.float64)
        if len(self._dihedral_indices) > 0:
            tvecs["dihedrals"] = self._dihedral_ncvecs @ cell
        else:
            tvecs["dihedrals"] = np.empty((0, 3, 3), dtype=np.float64)
        if len(self._bond_indices_padded) > 0:
            tvecs["bonds_padded"] = self._bond_ncvecs_padded @ cell
        else:
            tvecs["bonds_padded"] = np.empty((0, 1, 3), dtype=np.float64)
        if len(self._angle_indices_padded) > 0:
            tvecs["angles_padded"] = self._angle_ncvecs_padded @ cell
        else:
            tvecs["angles_padded"] = np.empty((0, 2, 3), dtype=np.float64)
        if len(self._dihedral_indices_padded) > 0:
            tvecs["dihedrals_padded"] = self._dihedral_ncvecs_padded @ cell
        else:
            tvecs["dihedrals_padded"] = np.empty((0, 3, 3), dtype=np.float64)
        self._tvecs_cache = {"cell_hash": cell_hash, "tvecs": tvecs}
        return tvecs

    def _invalidate_batched_arrays(self) -> None:
        self._batched_arrays_valid = False

    def _compute_batched_for_type(
        self,
        positions: np.ndarray,
        indices_padded: np.ndarray,
        n_actual: int,
        tvecs_padded: np.ndarray,
        jax_func,
        empty_shape: Tuple[int, ...],
    ):
        if n_actual == 0:
            return np.empty(empty_shape)
        pos_batch = positions[indices_padded]
        result_padded = np.asarray(device_get(jax_func(pos_batch, tvecs_padded)))
        return result_padded[:n_actual]

    def _compute_batched_values(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Dict[str, np.ndarray]:
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        return {
            "bonds": self._compute_batched_for_type(
                positions,
                self._bond_indices_padded,
                self._n_bonds_actual,
                tvecs["bonds_padded"],
                _bond_value_batched,
                (0,),
            ),
            "angles": self._compute_batched_for_type(
                positions,
                self._angle_indices_padded,
                self._n_angles_actual,
                tvecs["angles_padded"],
                _angle_value_batched,
                (0,),
            ),
            "dihedrals": self._compute_batched_for_type(
                positions,
                self._dihedral_indices_padded,
                self._n_dihedrals_actual,
                tvecs["dihedrals_padded"],
                _dihedral_value_batched,
                (0,),
            ),
        }

    def _compute_batched_gradients(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        return {
            "bonds": (
                self._bond_indices,
                self._compute_batched_for_type(
                    positions,
                    self._bond_indices_padded,
                    self._n_bonds_actual,
                    tvecs["bonds_padded"],
                    _bond_grad_batched,
                    (0, 2, 3),
                ),
            ),
            "angles": (
                self._angle_indices,
                self._compute_batched_for_type(
                    positions,
                    self._angle_indices_padded,
                    self._n_angles_actual,
                    tvecs["angles_padded"],
                    _angle_grad_batched,
                    (0, 3, 3),
                ),
            ),
            "dihedrals": (
                self._dihedral_indices,
                self._compute_batched_for_type(
                    positions,
                    self._dihedral_indices_padded,
                    self._n_dihedrals_actual,
                    tvecs["dihedrals_padded"],
                    _dihedral_grad_batched,
                    (0, 4, 3),
                ),
            ),
        }

    def _compute_batched_hessians(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        return {
            "bonds": (
                self._bond_indices,
                self._compute_batched_for_type(
                    positions,
                    self._bond_indices_padded,
                    self._n_bonds_actual,
                    tvecs["bonds_padded"],
                    _bond_hess_batched,
                    (0, 2, 3, 2, 3),
                ),
            ),
            "angles": (
                self._angle_indices,
                self._compute_batched_for_type(
                    positions,
                    self._angle_indices_padded,
                    self._n_angles_actual,
                    tvecs["angles_padded"],
                    _angle_hess_batched,
                    (0, 3, 3, 3, 3),
                ),
            ),
            "dihedrals": (
                self._dihedral_indices,
                self._compute_batched_for_type(
                    positions,
                    self._dihedral_indices_padded,
                    self._n_dihedrals_actual,
                    tvecs["dihedrals_padded"],
                    _dihedral_hess_batched,
                    (0, 4, 3, 4, 3),
                ),
            ),
        }

    @staticmethod
    def _scatter_jacobian_batch(
        B: np.ndarray,
        row: int,
        indices: np.ndarray,
        gradients: np.ndarray,
        active_mask: List[bool],
    ) -> int:
        active_arr = np.array(active_mask, dtype=bool)
        n_active = active_arr.sum()
        if n_active > 0:
            rows_idx = np.arange(row, row + n_active)[:, None]
            B[rows_idx, indices[active_arr]] = gradients[active_arr]
        return row + n_active

    @staticmethod
    def _append_hessians_batch(
        hessians: List,
        n_atoms: int,
        indices: np.ndarray,
        hess_values: np.ndarray,
        active_mask: List[bool],
    ) -> None:
        active_arr = np.array(active_mask, dtype=bool)
        if active_arr.any():
            active_idx = indices[active_arr]
            active_hess = hess_values[active_arr]
            for i in range(len(active_idx)):
                hessians.append(
                    SparseInternalHessian(n_atoms, active_idx[i], active_hess[i].copy())
                )

    def copy(self) -> "BaseInternals":
        raise NotImplementedError

    def calc(self) -> np.ndarray:
        self._cache_check()
        if "coords" not in self._cache:
            positions = self.all_positions
            cell = (
                self.atoms.cell.array
                if hasattr(self.atoms.cell, "array")
                else np.asarray(self.atoms.cell)
            )
            batched_vals = self._compute_batched_values(positions, cell)
            all_coords = []
            atoms = self.light_atoms
            for coord in self.internals["translations"]:
                all_coords.append(coord.calc(atoms))
            all_coords.extend(batched_vals["bonds"].tolist())
            all_coords.extend(batched_vals["angles"].tolist())
            all_coords.extend(batched_vals["dihedrals"].tolist())
            for coord in self.internals["other"]:
                all_coords.append(coord.calc(atoms))
            for coord in self.internals["rotations"]:
                all_coords.append(coord.calc(atoms))
            self._cache["coords"] = np.array(all_coords)
        return np.array(
            [x for x, a in zip(self._cache["coords"], self._active_mask) if a]
        )

    def jacobian(self) -> np.ndarray:
        self._cache_check()
        if "jacobian" not in self._cache:
            positions = self.all_positions
            cell = (
                self.atoms.cell.array
                if hasattr(self.atoms.cell, "array")
                else np.asarray(self.atoms.cell)
            )
            batched_grads = self._compute_batched_gradients(positions, cell)
            atoms = self.light_atoms
            trans_data = [
                (coord.indices, np.array(coord.calc_gradient(atoms)))
                for coord in self.internals["translations"]
            ]
            other_data = [
                (coord.indices, np.array(coord.calc_gradient(atoms)))
                for coord in self.internals["other"]
            ]
            rot_data = [
                (coord.indices, np.array(coord.calc_gradient(atoms)))
                for coord in self.internals["rotations"]
            ]
            self._cache["jacobian_batched"] = batched_grads
            self._cache["jacobian_nonbatched"] = (trans_data, other_data, rot_data)
            self._cache["jacobian"] = object()
        batched = self._cache["jacobian_batched"]
        trans_data, other_data, rot_data = self._cache["jacobian_nonbatched"]
        n_trans = len(trans_data)
        n_bonds = len(self.internals["bonds"])
        n_angles = len(self.internals["angles"])
        n_dihedrals = len(self.internals["dihedrals"])
        n_other = len(other_data)
        n_rot = len(rot_data)
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start : start + n_trans]
        start += n_trans
        bonds_active = active_mask[start : start + n_bonds]
        start += n_bonds
        angles_active = active_mask[start : start + n_angles]
        start += n_angles
        dihedrals_active = active_mask[start : start + n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start : start + n_other]
        start += n_other
        rot_active = active_mask[start : start + n_rot]
        n_active = sum(active_mask)
        n_atoms = self.natoms + self.ndummies
        B = np.zeros((n_active, n_atoms, 3))
        row = 0
        for i, (idx, jac) in enumerate(trans_data):
            if trans_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1
        bond_indices, bond_grads = batched["bonds"]
        row = self._scatter_jacobian_batch(
            B, row, bond_indices, bond_grads, bonds_active
        )
        angle_indices, angle_grads = batched["angles"]
        row = self._scatter_jacobian_batch(
            B, row, angle_indices, angle_grads, angles_active
        )
        dihedral_indices, dihedral_grads = batched["dihedrals"]
        row = self._scatter_jacobian_batch(
            B, row, dihedral_indices, dihedral_grads, dihedrals_active
        )
        for i, (idx, jac) in enumerate(other_data):
            if other_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1
        for i, (idx, jac) in enumerate(rot_data):
            if rot_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1
        return B.reshape((n_active, 3 * n_atoms))

    def hessian(self) -> np.ndarray:
        self._cache_check()
        if "hessian_result" in self._cache:
            return self._cache["hessian_result"]
        if "hessian" not in self._cache:
            positions = self.all_positions
            cell = (
                self.atoms.cell.array
                if hasattr(self.atoms.cell, "array")
                else np.asarray(self.atoms.cell)
            )
            batched_hess = self._compute_batched_hessians(positions, cell)
            atoms = self.light_atoms
            trans_data = [
                (coord.indices, np.array(coord.calc_hessian(atoms)))
                for coord in self.internals["translations"]
            ]
            other_data = [
                (coord.indices, np.array(coord.calc_hessian(atoms)))
                for coord in self.internals["other"]
            ]
            rot_data = [
                (coord.indices, np.array(coord.calc_hessian(atoms)))
                for coord in self.internals["rotations"]
            ]
            self._cache["hessian_batched"] = batched_hess
            self._cache["hessian_nonbatched"] = (trans_data, other_data, rot_data)
            self._cache["hessian"] = object()
        batched = self._cache["hessian_batched"]
        trans_data, other_data, rot_data = self._cache["hessian_nonbatched"]
        n_trans = len(trans_data)
        n_bonds = len(self.internals["bonds"])
        n_angles = len(self.internals["angles"])
        n_dihedrals = len(self.internals["dihedrals"])
        n_other = len(other_data)
        n_rot = len(rot_data)
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start : start + n_trans]
        start += n_trans
        bonds_active = active_mask[start : start + n_bonds]
        start += n_bonds
        angles_active = active_mask[start : start + n_angles]
        start += n_angles
        dihedrals_active = active_mask[start : start + n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start : start + n_other]
        start += n_other
        rot_active = active_mask[start : start + n_rot]
        n_atoms = self.natoms + self.ndummies
        hessians = []
        for i, (idx, hess) in enumerate(trans_data):
            if trans_active[i]:
                hessians.append(
                    SparseInternalHessian(n_atoms, np.array(idx), hess.copy())
                )
        bond_indices, bond_hess = batched["bonds"]
        self._append_hessians_batch(
            hessians, n_atoms, bond_indices, bond_hess, bonds_active
        )
        angle_indices, angle_hess = batched["angles"]
        self._append_hessians_batch(
            hessians, n_atoms, angle_indices, angle_hess, angles_active
        )
        dihedral_indices, dihedral_hess = batched["dihedrals"]
        self._append_hessians_batch(
            hessians, n_atoms, dihedral_indices, dihedral_hess, dihedrals_active
        )
        for i, (idx, hess) in enumerate(other_data):
            if other_active[i]:
                hessians.append(
                    SparseInternalHessian(n_atoms, np.array(idx), hess.copy())
                )
        for i, (idx, hess) in enumerate(rot_data):
            if rot_active[i]:
                hessians.append(
                    SparseInternalHessian(n_atoms, np.array(idx), hess.copy())
                )
        result = SparseInternalHessians(hessians, self.ndof)
        self._cache["hessian_result"] = result
        return result

    def hessian_rdot(self, v: np.ndarray) -> np.ndarray:
        self._cache_check()
        positions = self.all_positions
        cell = (
            self.atoms.cell.array
            if hasattr(self.atoms.cell, "array")
            else np.asarray(self.atoms.cell)
        )
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        v_atoms = v.reshape((-1, 3))
        n_atoms = self.natoms + self.ndummies
        ndof = self.ndof
        active_mask = self._active_mask
        n_trans = len(self.internals["translations"])
        n_bonds = len(self.internals["bonds"])
        n_angles = len(self.internals["angles"])
        n_dihedrals = len(self.internals["dihedrals"])
        n_other = len(self.internals["other"])
        n_rot = len(self.internals["rotations"])
        start = 0
        trans_active = active_mask[start : start + n_trans]
        start += n_trans
        bonds_active = np.array(active_mask[start : start + n_bonds], dtype=bool)
        start += n_bonds
        angles_active = np.array(active_mask[start : start + n_angles], dtype=bool)
        start += n_angles
        dihedrals_active = np.array(
            active_mask[start : start + n_dihedrals], dtype=bool
        )
        start += n_dihedrals
        other_active = active_mask[start : start + n_other]
        start += n_other
        rot_active = active_mask[start : start + n_rot]
        results = []
        n_active_trans = sum(trans_active)
        if n_active_trans > 0:
            results.append(np.zeros((n_active_trans, ndof)))
        if bonds_active.any() and self._n_bonds_actual > 0:
            if bonds_active.all():
                bond_pos = positions[self._bond_indices_padded]
                bond_tvecs = tvecs["bonds_padded"]
                v_sub = v_atoms[self._bond_indices_padded]
                hvp_padded = np.asarray(
                    device_get(_bond_hvp_batched(bond_pos, bond_tvecs, v_sub))
                )
                hvp = hvp_padded[: self._n_bonds_actual]
                active_idx = self._bond_indices
            else:
                active_idx = self._bond_indices[bonds_active]
                bond_pos = positions[active_idx]
                bond_tvecs = tvecs["bonds"][bonds_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(_bond_hvp_batched(bond_pos, bond_tvecs, v_sub))
                )
            n_coords = len(active_idx)
            result = np.zeros((n_coords, n_atoms, 3))
            row_idx = np.arange(n_coords)[:, None]
            result[row_idx, active_idx] = hvp
            results.append(result.reshape((n_coords, ndof)))
        if angles_active.any() and self._n_angles_actual > 0:
            if angles_active.all():
                angle_pos = positions[self._angle_indices_padded]
                angle_tvecs = tvecs["angles_padded"]
                v_sub = v_atoms[self._angle_indices_padded]
                hvp_padded = np.asarray(
                    device_get(_angle_hvp_batched(angle_pos, angle_tvecs, v_sub))
                )
                hvp = hvp_padded[: self._n_angles_actual]
                active_idx = self._angle_indices
            else:
                active_idx = self._angle_indices[angles_active]
                angle_pos = positions[active_idx]
                angle_tvecs = tvecs["angles"][angles_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(_angle_hvp_batched(angle_pos, angle_tvecs, v_sub))
                )
            n_coords = len(active_idx)
            result = np.zeros((n_coords, n_atoms, 3))
            row_idx = np.arange(n_coords)[:, None]
            result[row_idx, active_idx] = hvp
            results.append(result.reshape((n_coords, ndof)))
        if dihedrals_active.any() and self._n_dihedrals_actual > 0:
            if dihedrals_active.all():
                dih_pos = positions[self._dihedral_indices_padded]
                dih_tvecs = tvecs["dihedrals_padded"]
                v_sub = v_atoms[self._dihedral_indices_padded]
                hvp_padded = np.asarray(
                    device_get(_dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub))
                )
                hvp = hvp_padded[: self._n_dihedrals_actual]
                active_idx = self._dihedral_indices
            else:
                active_idx = self._dihedral_indices[dihedrals_active]
                dih_pos = positions[active_idx]
                dih_tvecs = tvecs["dihedrals"][dihedrals_active]
                v_sub = v_atoms[active_idx]
                hvp = np.asarray(
                    device_get(_dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub))
                )
            n_coords = len(active_idx)
            result = np.zeros((n_coords, n_atoms, 3))
            row_idx = np.arange(n_coords)[:, None]
            result[row_idx, active_idx] = hvp
            results.append(result.reshape((n_coords, ndof)))
        atoms = self.light_atoms
        for i, coord in enumerate(self.internals["other"]):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum("aibj,bj->ai", hess, v_sub)
                row = np.zeros(ndof)
                row.reshape((-1, 3))[idx] = hvp
                results.append(row[None, :])
        for i, coord in enumerate(self.internals["rotations"]):
            if rot_active[i]:
                idx = np.array(coord.indices)
                pos = positions[idx]
                v_sub = v_atoms[idx]
                axis = coord.kwargs["axis"]
                refpos = coord.kwargs["refpos"]
                hvp = np.asarray(
                    device_get(_rotation_hvp_jit(pos, axis, refpos, v_sub))
                )
                row = np.zeros(ndof)
                row.reshape((-1, 3))[idx] = hvp
                results.append(row[None, :])
        if results:
            return np.vstack(results)
        return np.empty((0, ndof))

    def wrap(self, vec: np.ndarray) -> np.ndarray:
        start = 0
        for name in self._names:
            if name == "dihedrals":
                end = start + len(self.internals[name])
                break
            start += len(self.internals[name])
        vec[start:end] = (vec[start:end] + np.pi) % (2 * np.pi) - np.pi
        return vec

    def __iter__(self) -> Iterator[Coordinate]:
        for name in self._names:
            for coord in self.internals[name]:
                yield coord

    def _get_neighbors(self, dx: np.ndarray) -> Iterator[np.ndarray]:
        pbc = self.atoms.pbc
        if self.cell is None or not np.all(self.cell == self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(complete_cell(self.cell), pbc=pbc)
            self.rcell = Cell(rcell)
        dx_sc = dx @ self.rcell.reciprocal().T
        offset = np.zeros(3, dtype=np.int32)
        for _ in range(2):
            offset += pbc * ((dx_sc - offset) // 1.0).astype(np.int32)
        for ts in product(*[np.arange(-1 * p, p + 1) for p in pbc]):
            yield ((np.array(ts) - offset) @ self.op)

    def _find_mic(self, indices: Tuple[int, ...]) -> np.ndarray:
        ncvecs = np.zeros((len(indices) - 1, 3), dtype=np.int32)
        if not np.any(self.atoms.pbc):
            return ncvecs
        pos = self.all_positions
        dxs = np.array([pos[i] - pos[j] for i, j in zip(indices[1:], indices[:-1])])
        for dx, ncvec in zip(dxs, ncvecs):
            vlen = np.inf
            for neighbor in self._get_neighbors(dx):
                trial = np.linalg.norm(dx + neighbor @ self.atoms.cell)
                if trial < vlen:
                    vlen = trial
                    ncvec[:] = neighbor
        return ncvecs

    def _get_ncvecs(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> np.ndarray:
        if ncvecs is None:
            if mic is None or not mic:
                return np.zeros((len(indices) - 1, 3), dtype=np.int32)
            else:
                return self._find_mic(indices)
        else:
            if mic:
                raise ValueError(
                    "Minimum image convention (mic) requested, but explicit periodic vectors (ncvecs) were also provided! These keyword arguments are mutually exclusive."
                )
            return np.asarray(ncvecs, dtype=np.int32).reshape((len(indices) - 1, 3))

    def get_principal_rotation_axes(self, indices: Tuple[int, ...]) -> jnp.ndarray:
        indices = np.asarray(indices, dtype=np.int32)
        pos = self.all_positions
        dx = pos[indices] - pos[indices].mean(0)
        Inertia = (dx * dx).sum() * jnp.eye(3) - (dx[:, None, :] * dx[:, :, None]).sum(
            0
        )
        _, rvecs = jnp.linalg.eigh(Inertia)
        return rvecs

    def add_dummy_to_internals(self, idx: int) -> None:
        didx = self.dinds[idx]
        assert didx >= 0
        for i, trans in enumerate(self.internals["translations"]):
            if idx in trans.indices:
                new_indices = (*trans.indices[:-1], didx)
                new_trans = Translation(new_indices, trans.dim)
                self.internals["translations"][i] = new_trans
        for i, rot in enumerate(self.internals["rotations"]):
            if idx in rot.indices[:-1]:
                new_indices = (*rot.indices[:-1], didx)
                new_rot = Rotation(
                    new_indices, rot.axis, self.all_positions[new_indices]
                )
                self.internals["rotations"][i] = new_rot

    def check_all_gradients(self, delta: float = 0.0001, atol: float = 1e-06) -> bool:
        success = True
        for coord in self:
            success &= coord.check_gradient(self.all_atoms, delta, atol)
        return success

    def check_all_hessians(self, delta: float = 0.0001, atol: float = 1e-06) -> bool:
        success = True
        for coord in self:
            success &= coord.check_hessian(self.all_atoms, delta, atol)
        return success


class Constraints(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
        ignore_rotation: bool = True,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self._targets = {key: [] for key in self._names}
        self._kind = {key: [] for key in self._names}
        self.ignore_rotation = ignore_rotation
        if atoms.constraints:
            raise ValueError(
                "sella_minimal does not support pre-existing ASE constraints."
            )

    def copy(self) -> "Constraints":
        new = self.__class__(self.atoms, self.dummies, self.dinds, self.ignore_rotation)
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new._targets[name] = self._targets[name].copy()
            new._active[name] = self._active[name].copy()
            new._kind[name] = self._kind[name].copy()
        return new

    @property
    def targets(self) -> np.ndarray:
        vec = []
        for key in self._names:
            vec += self._targets[key]
        return np.array(vec, dtype=np.float64)[self._active_indices]

    def residual(self) -> np.ndarray:
        res = self.wrap(self.calc() - self.targets)
        if self.ignore_rotation and self.nrotations:
            res[-self.nrotations :] = 0.0
        return res

    def disable_satisfied_inequalities(self) -> None:
        for name in self._names:
            for i, (coord, kind, target) in enumerate(
                zip(self.internals[name], self._kind[name], self._targets[name])
            ):
                if kind == "lt" and coord.calc(self.all_atoms) <= target:
                    active = False
                elif kind == "gt" and coord.calc(self.all_atoms) >= target:
                    active = False
                else:
                    active = True
                self._active[name][i] = active

    def validate_inequalities(self) -> bool:
        all_valid = True
        for name in self._names:
            for i, (coord, kind, target) in enumerate(
                zip(self.internals[name], self._kind[name], self._targets[name])
            ):
                if self._active[name][i]:
                    continue
                if kind == "lt" and coord.calc(self.all_atoms) > target:
                    self._active[name][i] = True
                    all_valid = False
                elif kind == "gt" and coord.calc(self.all_atoms) < target:
                    self._active[name][i] = True
                    all_valid = False
        return all_valid

    def fix_rotation(
        self, indices: Union[Tuple[int, ...], Rotation] = None, axis: int = None
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError("'axis' keyword cannot be used with explicit Rotation")
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.fix_rotation(indices, axis)
                return
            new = Rotation(indices, axis, self.all_positions[indices])
        try:
            _ = self.internals["rotations"].index(new)
        except ValueError:
            self.internals["rotations"].append(new)
            self._targets["rotations"].append(0.0)
            self._active["rotations"].append(True)
            self._kind["rotations"].append("eq")
        else:
            raise DuplicateConstraintError(
                "This rotation has already been constrained!"
            )

    def fix_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None,
        target: float = None,
        replace_ok: bool = True,
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            if np.isscalar(index):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                if target is not None:
                    raise ValueError('"target" keyword requires explicit "dim"!')
                for dim in range(3):
                    self.fix_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if target is None:
            target = new.calc(self.all_atoms)
        try:
            idx = self.internals["translations"].index(new)
        except ValueError:
            self.internals["translations"].append(new)
            self._targets["translations"].append(target)
            self._active["translations"].append(True)
            self._kind["translations"].append("eq")
        else:
            if replace_ok:
                self._targets["translations"][idx] = target
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}".format(
                    new, self._targets["translations"][idx]
                )
            )

    def _fix_internal(
        self,
        kind: TypeVar("Coordinate", bound=Coordinate),
        name: str,
        conv: float,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
        target: float = None,
        comparator: str = "eq",
        replace_ok: bool = True,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used with explicit {}'.format(
                        kind.__name__
                    )
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if target is None:
            target = new.calc(self.all_atoms)
        else:
            target *= conv
        try:
            idx = self.internals[name].index(new)
        except ValueError:
            self.internals[name].append(new)
            self._targets[name].append(target)
            self._active[name].append(True)
            self._kind[name].append(comparator)
        else:
            if replace_ok:
                self._targets[name][idx] = target
                self._kind[name][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}".format(
                    new, self._targets[name][idx] / conv
                )
            )

    fix_bond = partialmethod(_fix_internal, Bond, "bonds", 1.0)
    fix_angle = partialmethod(_fix_internal, Angle, "angles", np.pi / 180.0)
    fix_dihedral = partialmethod(_fix_internal, Dihedral, "dihedrals", np.pi / 180.0)

    def fix_other(
        self,
        coord: Coordinate,
        target: float = None,
        comparator: str = "eq",
        replace_ok: bool = True,
    ) -> None:
        if target is None:
            target = coord.calc(self.all_atoms)
        try:
            idx = self.internals["other"].index(coord)
        except ValueError:
            self.internals["other"].append(coord)
            self._targets["other"].append(target)
            self._active["other"].append(True)
            self._kind["other"].append(comparator)
        else:
            if replace_ok:
                self._targets["other"][idx] = target
                self._kind["other"][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}".format(
                    coord, self._targets["other"][idx]
                )
            )


class Internals(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        atol: float = 5.0,
        dinds: np.ndarray = None,
        cons: Constraints = None,
        allow_fragments: bool = False,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self.atol = atol * np.pi / 180.0
        self.forbidden = {key: [] for key in self._names}
        if cons is None:
            cons = Constraints(self.atoms, self.dummies, self.dinds)
        else:
            if (
                dummies is not None
                and dummies is not cons.dummies
                or (dinds is not None and dinds is not cons.dinds)
            ):
                raise RuntimeError(
                    "Constraints has inconsistent dummy atom definitions!"
                )
            self.dummies = cons.dummies
            self.dinds = cons.dinds
        self.cons = cons
        for kind, adder in zip(
            self._names,
            (
                self.add_translation,
                self.add_bond,
                self.add_angle,
                self.add_dihedral,
                self.add_other,
                self.add_rotation,
            ),
        ):
            for coord in self.cons.internals[kind]:
                adder(coord)
        self.allow_fragments = allow_fragments

    def copy(self) -> "Internals":
        new = self.__class__(
            self.atoms,
            self.dummies,
            self.atol * 180.0 / np.pi,
            self.dinds,
            self.cons.copy(),
            self.allow_fragments,
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new.forbidden[name] = self.forbidden[name].copy()
            new._active[name] = self._active[name].copy()
        return new

    def add_rotation(
        self, indices: Union[Tuple[int, ...], Rotation] = None, axis: int = None
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError("'axis' keyword cannot be used with explicit Rotation")
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.array(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.add_rotation(indices, axis)
                return
            new = Rotation(indices, axis, self.all_positions[indices])
        if new in self.internals["rotations"] or new in self.forbidden["rotations"]:
            raise DuplicateInternalError
        self.internals["rotations"].append(new)
        self._active["rotations"].append(True)

    def add_translation(
        self, index: Union[int, Tuple[int, ...], Translation] = None, dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError('"dim" keyword cannot be used with explicit Cart')
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.add_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if (
            new in self.internals["translations"]
            or new in self.forbidden["translations"]
        ):
            raise DuplicateInternalError
        self.internals["translations"].append(new)
        self._active["translations"].append(True)

    def _add_internal(
        self,
        kind: TypeVar("Coordinate", bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used with explicit {}'.format(
                        kind.__name__
                    )
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if new in self.internals[name] or new in self.forbidden[name]:
            raise DuplicateInternalError
        self.internals[name].append(new)
        self._active[name].append(True)

    add_bond = partialmethod(_add_internal, Bond, "bonds")
    add_angle = partialmethod(_add_internal, Angle, "angles")
    add_dihedral = partialmethod(_add_internal, Dihedral, "dihedrals")

    def add_other(self, coord: Coordinate) -> None:
        try:
            self.internals["other"].index(coord)
        except ValueError:
            self.internals["other"].append(coord)
            self._active["other"].append(True)
        else:
            raise DuplicateInternalError()

    def forbid_translation(
        self, index: Union[int, Tuple[int, ...], Translation] = None, dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError('"dim" keyword cannot be used with explicit Cart')
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.forbid_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        try:
            self.internals["translations"].remove(new)
        except ValueError:
            pass
        if new not in self.forbidden["translations"]:
            self.forbidden["translations"].append(new)

    def _forbid_internal(
        self,
        kind: TypeVar("Coordinate", bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used with explicit {}'.format(
                        kind.__name__
                    )
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        try:
            self.forbidden[name].remove(new)
        except ValueError:
            pass
        if new not in self.forbidden[name]:
            self.forbidden[name].append(new)

    forbid_bond = partialmethod(_forbid_internal, Bond, "bonds")
    forbid_angle = partialmethod(_forbid_internal, Angle, "angles")
    forbid_dihedral = partialmethod(_forbid_internal, Dihedral, "dihedrals")

    @staticmethod
    def flood_fill(
        index: int, nbonds: np.ndarray, c10y: np.ndarray, labels: np.ndarray, label: int
    ) -> None:
        for j in c10y[index, : nbonds[index]]:
            if labels[j] != label:
                labels[j] = label
                Internals.flood_fill(j, nbonds, c10y, labels, label)

    def find_all_bonds(
        self, nbond_cart_thr: int = 6, max_bonds: int = 20, scale: float = 1.225
    ) -> None:
        rcov = covalent_radii[self.atoms.numbers]
        nbonds = np.zeros(self.natoms, dtype=np.int32)
        labels = -np.ones(self.natoms, dtype=np.int32)
        c10y = -np.ones((self.natoms, max_bonds), dtype=np.int32)
        for bond in self.internals["bonds"]:
            i, j = bond.indices
            c10y[i, nbonds[i]] = j
            nbonds[i] += 1
            c10y[j, nbonds[j]] = i
            nbonds[j] += 1
        first_run = True
        while True:
            nlabels = 0
            labels[:] = -1
            for i in range(self.natoms):
                if labels[i] == -1:
                    labels[i] = nlabels
                    self.flood_fill(i, nbonds, c10y, labels, nlabels)
                    nlabels += 1
            if nlabels == 1:
                break
            labels[nbonds == 0] = -1
            if self.allow_fragments and (not first_run):
                break
            for i, j in cwr(range(self.natoms), 2):
                if labels[i] == labels[j] and labels[i] != -1:
                    continue
                dx = self.atoms.positions[j] - self.atoms.positions[i]
                for ts in self._get_neighbors(dx):
                    if i == j and np.all(ts == np.array((0, 0, 0))):
                        continue
                    dist = np.linalg.norm(dx + ts @ self.atoms.cell)
                    if dist <= scale * (rcov[i] + rcov[j]):
                        try:
                            self.add_bond((i, j), ts)
                        except DuplicateInternalError:
                            continue
                        if nbonds[i] < max_bonds and nbonds[j] < max_bonds:
                            c10y[i, nbonds[i]] = j
                            nbonds[i] += 1
                            c10y[j, nbonds[j]] = i
                            nbonds[j] += 1
                        else:
                            pass
            first_run = False
            scale *= 1.05
        if self.allow_fragments and nlabels != 1:
            assert nlabels > 1
            groups = [[] for _ in range(nlabels)]
            for i, label in enumerate(labels):
                if label == -1:
                    self.add_translation(i)
                else:
                    groups[label].append(i)
            for group in groups:
                if not group:
                    continue
                self.add_translation(group)
                self.add_rotation(group)

    def find_all_angles(self) -> None:
        bonds = [[] for _ in range(self.natoms)]
        for bond in self.internals["bonds"]:
            i, j = bond.indices
            if i < self.natoms:
                bonds[i].append(bond)
            if j < self.natoms:
                bonds[j].append(bond.reverse())
        for j, jbonds in enumerate(bonds):
            linear = []
            for b1, b2 in combinations(jbonds, 2):
                new = b1 + b2
                assert new.indices[1] == j, new.indices
                if self.atol < new.calc(self.atoms) < np.pi - self.atol:
                    try:
                        self.add_angle(new)
                    except DuplicateInternalError:
                        pass
                else:
                    self.forbid_angle(new)
                    linear.append((b1, b2))
            if linear:
                if len(jbonds) == 2:
                    b1, b2 = sorted(jbonds, key=lambda x: x.calc(self.atoms))
                    if self.dinds[j] < 0:
                        self.dinds[j] = self.natoms + self.ndummies
                        dx1 = -b1.calc_vec(self.atoms)
                        dx1 /= np.linalg.norm(dx1)
                        dx2 = b2.calc_vec(self.atoms)
                        dx2 /= np.linalg.norm(dx2)
                        dpos = np.cross(dx1, dx2)
                        dpos_norm = np.linalg.norm(dpos)
                        if dpos_norm < 0.0001:
                            dim = np.argmin(np.abs(dx1))
                            dpos[:] = 0.0
                            dpos[dim] = 1.0
                            dpos -= dx1 * (dpos @ dx1)
                            dpos /= np.linalg.norm(dpos)
                        else:
                            dpos /= dpos_norm
                        dpos += self.atoms.positions[j]
                        self.dummies += Atom("X", dpos)
                    dbond = Bond((j, self.dinds[j]))
                    self.cons.fix_bond(dbond, replace_ok=False)
                    self.add_bond(dbond)
                    dangle1 = b1 + dbond
                    self.cons.fix_angle(dangle1, replace_ok=False)
                    dangle2 = b2 + dbond
                    self.cons.fix_angle(dangle2, replace_ok=False)
                    if b2.indices[1] == j:
                        b2 = b2.reverse()
                    dbond2 = Bond((self.dinds[j], b2.indices[1]), b2.kwargs["ncvecs"])
                    dangle3 = dbond + dbond2
                    ddihedral = dangle1 + dangle3
                    self.add_dihedral(ddihedral)
                    self.add_dummy_to_internals(j)
                    self.cons.add_dummy_to_internals(j)
                    for b1 in jbonds:
                        new = b1 + dbond
                        assert new.indices[1] == j
                        angle = new.calc(self.all_atoms)
                        if self.atol < angle < np.pi - self.atol:
                            try:
                                self.add_angle(new)
                            except DuplicateInternalError:
                                pass
                        else:
                            self.forbid_angle(new)
                else:
                    for b1, b2 in linear:
                        for b3 in jbonds:
                            if b3 in (b1, b2):
                                continue
                            indices = (b1.indices[1], j, b3.indices[1], b2.indices[1])
                            ncvecs = (
                                -b1.kwargs["ncvecs"][0],
                                b3.kwargs["ncvecs"][0],
                                b2.kwargs["ncvecs"][0] - b3.kwargs["ncvecs"][0],
                            )
                            try:
                                self.add_dihedral(indices, ncvecs)
                            except DuplicateInternalError:
                                pass
                            break
                        else:
                            raise RuntimeError(
                                "Unable to find improper dihedral to replace linear angle!"
                            )

    def find_all_dihedrals(self) -> None:
        for a1, a2 in combinations(self.internals["angles"], 2):
            try:
                new = a1 + a2
            except NoValidInternalError:
                continue
            if new.indices[0] == new.indices[3] and np.all(
                np.sum(new.kwargs["ncvecs"], axis=0) == np.array((0, 0, 0))
            ):
                continue
            try:
                self.add_dihedral(new)
            except DuplicateInternalError:
                continue
        dihedral_centers = set()
        for d, a in zip(self.internals["dihedrals"], self._active["dihedrals"]):
            if a:
                dihedral_centers.add(int(d.indices[1]))
                dihedral_centers.add(int(d.indices[2]))
        neighbors = [[] for _ in range(self.natoms)]
        for bond in self.internals["bonds"]:
            i, j = bond.indices
            if i < self.natoms:
                neighbors[i].append((int(j), bond.kwargs["ncvecs"][0]))
            if j < self.natoms:
                neighbors[j].append((int(i), -bond.kwargs["ncvecs"][0]))
        for center in range(self.natoms):
            if len(neighbors[center]) != 3:
                continue
            if center in dihedral_centers:
                continue
            cell = np.asarray(self.atoms.cell)
            ordered = sorted(
                neighbors[center],
                key=lambda item: np.linalg.norm(
                    self.atoms.positions[item[0]]
                    + item[1] @ cell
                    - self.atoms.positions[center]
                ),
            )
            n1, ncvec1 = ordered[2]
            n0, ncvec0 = ordered[0]
            n2, ncvec2 = ordered[1]
            imp_ncvecs = (-ncvec0, ncvec1, ncvec2 - ncvec1)
            try:
                improper = Dihedral((n0, center, n1, n2), imp_ncvecs)
                improper.fallback_improper = True
                self.add_dihedral(improper)
            except DuplicateInternalError:
                pass

    def validate_basis(self) -> None:
        jac = self.jacobian()
        U, S, VT = np.linalg.svd(jac)
        ndeloc = np.sum(S > 1e-08)
        has_trics = (
            len(self.internals["translations"]) > 0
            or len(self.internals["rotations"]) > 0
        )
        if has_trics:
            ndof = 3 * (self.natoms + self.ndummies)
        else:
            ndof = 3 * (self.natoms + self.ndummies) - 6
        if ndeloc != ndof:
            warnings.warn(f"{ndeloc} coords found! Expected {ndof}.")

    def check_for_bad_internals(self) -> Optional[Dict[str, List[Coordinate]]]:
        bad = {"bonds": [], "angles": []}
        angles = self.internals["angles"]
        if not angles:
            return None
        self._build_batched_arrays()
        if self._n_angles_actual > 0:
            positions = self.all_positions
            cell = (
                self.atoms.cell.array
                if hasattr(self.atoms.cell, "array")
                else np.asarray(self.atoms.cell)
            )
            tvecs = self._get_cached_tvecs(cell)
            angle_pos = positions[self._angle_indices_padded]
            angle_vals_padded = np.asarray(
                _angle_value_batched(angle_pos, tvecs["angles_padded"])
            )
            angle_vals = angle_vals_padded[: self._n_angles_actual]
            bad_mask = ~((self.atol < angle_vals) & (angle_vals < np.pi - self.atol))
            if np.any(bad_mask):
                bad_indices = np.where(bad_mask)[0]
                for idx in bad_indices:
                    bad["angles"].append(angles[idx])
        for ints in bad.values():
            if ints:
                return bad
        return None

    def _h0_bond(self, bond: Bond, Ab: float = 0.3601, Bb: float = 1.944) -> float:
        idx = np.asarray(bond.indices, dtype=np.int32)
        rcov = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rij = bond.calc(self.all_atoms)
        h0 = Ab * np.exp(-Bb * (rij - rcov) / units.Bohr)
        return h0 * units.Hartree / units.Bohr**2

    def _h0_angle(
        self,
        angle: Angle,
        Aa: float = 0.089,
        Ba: float = 0.11,
        Ca: float = 0.44,
        Da: float = -0.42,
    ) -> float:
        bab, bbc = angle.split()
        idxab = np.asarray(bab.indices, dtype=np.int32)
        idxbc = np.asarray(bbc.indices, dtype=np.int32)
        rcovab = covalent_radii[self.all_atoms.numbers[idxab]].sum()
        rcovbc = covalent_radii[self.all_atoms.numbers[idxbc]].sum()
        rab = bab.calc(self.all_atoms)
        rbc = bbc.calc(self.all_atoms)
        h0 = (
            Aa
            + Ba
            * np.exp(-Ca * (rab + rbc - rcovab - rcovbc) / units.Bohr)
            / (rcovab * rcovbc / units.Bohr**2) ** Da
        )
        return h0 * units.Hartree

    def _h0_dihedral(
        self,
        dihedral: Dihedral,
        nbonds: np.ndarray,
        At: float = 0.0015,
        Bt: float = 14.0,
        Ct: float = 2.85,
        Dt: float = 0.57,
        Et: float = 4.0,
    ) -> float:
        _, bbc = dihedral.split()[0].split()
        idx = np.asarray(bbc.indices, dtype=np.int32)
        rcovbc = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rbc = bbc.calc(self.all_atoms)
        L = nbonds[idx].sum() - 2
        h0 = (
            At
            + Bt
            * L**Dt
            * np.exp(-Ct * (rbc - rcovbc) / units.Bohr)
            / (rbc * rcovbc / units.Bohr**2) ** Et
        )
        if getattr(dihedral, "fallback_improper", False):
            h0 *= 0.5
        return h0 * units.Hartree

    def guess_hessian(self, h0cart=70.0) -> np.ndarray:
        nbonds = np.zeros(len(self.all_atoms), dtype=np.int32)
        h0 = np.zeros(self.nint, dtype=np.float64)
        idx = 0
        for trans in self.internals["translations"]:
            h0[idx] = h0cart
            idx += 1
        for bond in self.internals["bonds"]:
            h0[idx] = self._h0_bond(bond)
            idx += 1
            i, j = bond.indices
            nbonds[i] += 1
            nbonds[j] += 1
        for angle in self.internals["angles"]:
            h0[idx] = self._h0_angle(angle)
            idx += 1
        for dihedral in self.internals["dihedrals"]:
            h0[idx] = self._h0_dihedral(dihedral, nbonds)
            idx += 1
        h0[idx:] = 1.0
        return np.diag(np.abs(h0))


class PES:
    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eta: float = 0.0001,
        proj_trans: bool = None,
        proj_rot: bool = None,
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals["translations"]:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass
        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
        self.eta = eta
        self.neval = 0
        self.curr = dict(x=None, f=None, g=None)
        self.last = self.curr.copy()
        self.int = None
        self.dummies = None
        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)
        self.savepoint = dict(apos=None, dpos=None)
        self._basis_cache = dict(pos_hash=None, result=None)

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint["apos"]
        dpos = self.savepoint["dpos"]
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return (diff, diff, self.curr.get("g", np.zeros_like(diff)))

    def get_x(self):
        return self.apos.ravel().copy()

    def get_H(self):
        return self.H

    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(self.dim, self.ncart, target, *args, **kwargs)

    def get_Hc(self):
        return self.cons.hessian().ldot(self.curr["L"])

    def get_HL(self):
        return self.get_H() - self.get_Hc()

    def get_res(self):
        return self.cons.residual()

    def get_drdx(self):
        return self.cons.jacobian()

    def _calc_basis(self):
        pos_hash = self.atoms.positions.tobytes()
        if self._basis_cache["pos_hash"] == pos_hash:
            return self._basis_cache["result"]
        drdx = self.get_drdx()
        U, S, VT = np.linalg.svd(drdx)
        ncons = np.sum(S > 1e-06)
        Ucons = VT[:ncons].T
        Ufree = VT[ncons:].T
        Unred = np.eye(self.dim)
        result = (drdx, Ucons, Unred, Ufree)
        self._basis_cache["pos_hash"] = pos_hash
        self._basis_cache["result"] = result
        return result

    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        return (f, g)

    def _calc_eg(self, x):
        self.save()
        self.set_x(x)
        f, g = self.eval()
        self.restore()
        return (f, g)

    def get_scons(self):
        Ucons = self.get_Ucons()
        scons = (
            -Ucons
            @ np.linalg.lstsq(self.get_drdx() @ Ucons, self.get_res(), rcond=None)[0]
        )
        return scons

    def _update(self, feval=True):
        x = self.get_x()
        new_point = True
        if self.curr["x"] is not None and np.all(x == self.curr["x"]):
            if feval and self.curr["f"] is None:
                new_point = False
            else:
                return False
        basis = self._calc_basis()
        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None
        if new_point:
            self.last = self.curr.copy()
        self.curr["x"] = x
        self.curr["f"] = f
        self.curr["g"] = g
        self._update_basis(basis)
        return True

    def _update_basis(self, basis=None):
        if basis is None:
            basis = self._calc_basis()
        drdx, Ucons, Unred, Ufree = basis
        self.curr["drdx"] = drdx
        self.curr["Ucons"] = Ucons
        self.curr["Unred"] = Unred
        self.curr["Ufree"] = Ufree
        if self.curr["g"] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr["g"], rcond=None)[0]
        self.curr["L"] = L

    def _update_H(self, dx, dg):
        if self.last["x"] is None or self.last["g"] is None:
            return
        self.H.update(dx, dg)

    def get_f(self):
        self._update()
        return self.curr["f"]

    def get_g(self):
        self._update()
        return self.curr["g"].copy()

    def get_Unred(self):
        self._update(False)
        return self.curr["Unred"]

    def get_Ufree(self):
        self._update(False)
        return self.curr["Ufree"]

    def get_Ucons(self):
        self._update(False)
        return self.curr["Ucons"]

    def wrap_dx(self, dx):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + dx.T @ H @ dx / 2.0

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()
        dx_initial, dx_final, g_par = self.set_x(x0 + dx)
        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        if df_pred is None or abs(df_pred) < 1e-14:
            ratio = None
        else:
            ratio = df_actual / df_pred
        self._update_H(dx_final, dg_actual)
        if diag:
            raise ValueError("sella_minimal does not support eig=True diagonalization.")
        return ratio


class InternalPES(PES):
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        auto_find_internals: bool = True,
        **kwargs,
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()
        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs,
        )
        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        if H0 is None:
            B = self.int.jacobian()
            P = B @ np.linalg.pinv(B)
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)
        self.bad_int = None
        self._pinv_cache = dict(version=None, pinv=None)

    dpos = property(lambda self: self.dummies.positions.copy())

    def _get_Binv(self):
        B = self.int.jacobian()
        version = self.int._cache_version
        if (
            self._pinv_cache.get("version") == version
            and self._pinv_cache.get("pinv") is not None
        ):
            return self._pinv_cache["pinv"]
        Binv = np.linalg.pinv(B)
        self._pinv_cache["version"] = version
        self._pinv_cache["pinv"] = Binv
        return Binv

    def _set_x_ode(self, target):
        dx = target - self.get_x()
        t0 = 0.0
        Binv = self._get_Binv()
        y0 = np.hstack(
            (
                self.apos.ravel(),
                self.dpos.ravel(),
                Binv @ dx,
                Binv @ self.curr.get("g", np.zeros_like(dx)),
            )
        )
        ode = LSODA(self._q_ode, t0, y0, t_bound=1.0, atol=1e-06)
        while ode.status == "running":
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                raise RuntimeError(
                    "Geometry update ODE is taking too long to converge!"
                )
        if ode.status == "failed":
            raise RuntimeError("Geometry update ODE failed to converge!")
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return (dx_initial, dx_final, g_final)

    def set_x(self, target):
        return self._set_x_ode(target)

    def get_x(self):
        return self.int.calc()

    def get_Hc(self):
        D_cons = self.cons.hessian().ldot(self.curr["L"])
        Binv_int = self._get_Binv()
        B_cons = self.cons.jacobian()
        L_int = self.curr["L"] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        Hc = Binv_int.T @ (D_cons - D_int) @ Binv_int
        return Hc

    def get_drdx(self):
        return PES.get_drdx(self) @ self._get_Binv()

    def _calc_basis(self, internal=None, cons=None):
        if internal is not None or cons is not None:
            if internal is None:
                internal = self.int
            if cons is None:
                cons = self.cons
            B = internal.jacobian()
            Ui, Si, VTi = np.linalg.svd(B)
            nnred = np.sum(Si > 1e-06)
            Unred = Ui[:, :nnred]
            Vnred = VTi[:nnred].T
            Siinv = np.diag(1 / Si[:nnred])
            drdxnred = cons.jacobian() @ Vnred @ Siinv
            drdx = drdxnred @ Unred.T
            Uc, Sc, VTc = np.linalg.svd(drdxnred)
            ncons = np.sum(Sc > 1e-06)
            Ucons = Unred @ VTc[:ncons].T
            Ufree = Unred @ VTc[ncons:].T
            return (drdx, Ucons, Unred, Ufree)
        pos_hash = self.atoms.positions.tobytes() + self.dummies.positions.tobytes()
        if self._basis_cache["pos_hash"] == pos_hash:
            return self._basis_cache["result"]
        internal = self.int
        cons = self.cons
        B = internal.jacobian()
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-06)
        Unred = Ui[:, :nnred]
        Vnred = VTi[:nnred].T
        Siinv = np.diag(1 / Si[:nnred])
        Binv = Vnred @ Siinv @ Unred.T
        self._pinv_cache["version"] = internal._cache_version
        self._pinv_cache["pinv"] = Binv
        drdxnred = cons.jacobian() @ Vnred @ Siinv
        drdx = drdxnred @ Unred.T
        Uc, Sc, VTc = np.linalg.svd(drdxnred)
        ncons = np.sum(Sc > 1e-06)
        Ucons = Unred @ VTc[:ncons].T
        Ufree = Unred @ VTc[ncons:].T
        result = (drdx, Ucons, Unred, Ufree)
        self._basis_cache["pos_hash"] = pos_hash
        self._basis_cache["result"] = result
        return result

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = self._get_Binv()
        return (f, g_cart @ Binv[: len(g_cart)])

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + dx_r.T @ H_r @ dx_r / 2.0

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))
        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt
        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()
        D_rdot = self.int.hessian_rdot(dxdt)
        Binv = np.linalg.pinv(self.int.jacobian())
        D_tmp = -Binv @ D_rdot
        dydt[1] = D_tmp @ dxdt
        dydt[2] = D_tmp @ g
        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)
        return ratio

    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return
        B = self.int.jacobian()
        Binv = self._get_Binv()
        self.curr.update(B=B, Binv=Binv)
        return True


class BaseStepper:
    alpha0: Optional[float] = None
    alphamin: Optional[float] = None
    alphamax: Optional[float] = None
    slope: Optional[float] = None
    synonyms: List[str] = []

    def __init__(
        self,
        g: np.ndarray,
        H: ApproximateHessian,
        order: int = 0,
        d1: Optional[np.ndarray] = None,
    ) -> None:
        self.g = g
        self.H = H
        self.order = order
        self.d1 = d1
        self._stepper_init()

    @classmethod
    def match(cls, name: str) -> bool:
        return name in cls.synonyms

    def _stepper_init(self) -> None:
        raise NotImplementedError

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class NaiveStepper(BaseStepper):
    synonyms = []
    alpha0 = 0.5
    alphamin = 0.0
    alphamax = 1.0
    slope = 1.0

    def __init__(self, dx: np.ndarray) -> None:
        self.dx = dx

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        return (alpha * self.dx, self.dx)


class QuasiNewton(BaseStepper):
    alpha0 = 0.0
    alphamin = 0.0
    alphamax = np.inf
    slope = -1
    synonyms = [
        "qn",
        "quasi-newton",
        "quasi newton",
        "quasi-newton",
        "newton",
        "mmf",
        "minimum mode following",
        "minimum-mode following",
        "dimer",
    ]

    def _stepper_init(self) -> None:
        if self.H.evals is None:
            H_array = self.H.asarray()
            self.H.evals, self.H.evecs = eigh(H_array)
        self.L = np.abs(self.H.evals)
        self.L[: self.order] *= -1
        self.V = self.H.evecs
        self.Vg = self.V.T @ self.g
        self.ones = np.ones_like(self.L)
        self.ones[: self.order] = -1

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        denom = self.L + alpha * self.ones
        sproj = self.Vg / denom
        s = -self.V @ sproj
        dsda = self.V @ (sproj / denom)
        return (s, dsda)


_all_steppers = [QuasiNewton]


def get_stepper(name: str) -> Type[BaseStepper]:
    for stepper in _all_steppers:
        if stepper.match(name):
            return stepper
    raise ValueError("Unknown stepper name: {}".format(name))


class BaseRestrictedStep:
    synonyms: List[str] = []

    def __init__(
        self,
        pes: Union[PES, InternalPES],
        order: int,
        delta: float,
        method: str = "qn",
        tol: float = 1e-15,
        maxiter: int = 1000,
        d1: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        self.pes = pes
        self.delta = delta
        self.d1 = d1
        g0 = self.pes.get_g()
        if W is None:
            W = np.eye(len(g0))
        self.scons = self.pes.get_scons()
        g = g0 + self.pes.get_H() @ self.scons
        if isinstance(method, str):
            stepper = get_stepper(method.lower())
        else:
            stepper = method
        if self.cons(self.scons) - self.delta > 1e-08:
            self.P = self.pes.get_Unred().T
            dx = self.P @ self.scons
            self.stepper = NaiveStepper(dx)
            self.scons[:] *= 0
        else:
            self.P = self.pes.get_Ufree().T @ W
            d1 = self.d1
            if d1 is not None:
                d1 = np.linalg.lstsq(self.P.T, d1, rcond=None)[0]
            self.stepper = stepper(
                self.P @ g, self.pes.get_HL().project(self.P.T), order, d1=d1
            )
        self.tol = tol
        self.maxiter = maxiter

    def cons(self, s, dsda=None):
        raise NotImplementedError

    def eval(self, alpha):
        s, dsda = self.stepper.get_s(alpha)
        stot = self.P.T @ s + self.scons
        val, dval = self.cons(stot, self.P.T @ dsda)
        return (stot, val, dval)

    def get_s(self):
        alpha = self.stepper.alpha0
        s, val, dval = self.eval(alpha)
        if val < self.delta:
            assert val > 0.0
            return (s, val)
        err = val - self.delta
        lower = self.stepper.alphamin
        upper = self.stepper.alphamax
        for niter in range(self.maxiter):
            if abs(err) <= self.tol:
                break
            if np.nextafter(lower, upper) >= upper:
                break
            if err * self.stepper.slope > 0:
                upper = alpha
            else:
                lower = alpha
            a1 = alpha - err / dval
            if np.isnan(a1) or a1 <= lower or a1 >= upper or (niter > 4):
                a2 = (lower + upper) / 2.0
                if np.isinf(a2):
                    alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
                else:
                    alpha = a2
            else:
                alpha = a1
            s, val, dval = self.eval(alpha)
            err = val - self.delta
        else:
            raise RuntimeError("Restricted step failed to converge!")
        assert val > 0
        return (s, self.delta)

    @classmethod
    def match(cls, name):
        return name in cls.synonyms


class MaxInternalStep(BaseRestrictedStep):
    synonyms = ["mis", "max internal step"]

    def __init__(
        self,
        pes,
        *args,
        wx=1.0,
        wb=1.0,
        wa=1.08,
        wd=1.0,
        wo=1.0,
        curvature_weighting=True,
        **kwargs,
    ):
        if pes.int is None:
            raise ValueError(
                f"Internal coordinates are required for the {self.__class__.__name__} trust region method"
            )
        self.wx = wx
        self.wb = wb
        self.wa = wa
        self.wd = wd
        self.wo = wo
        self.curvature_weighting = curvature_weighting
        self.curvature_weights = self._curvature_weights(pes)
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    @staticmethod
    def _curvature_weights(pes):
        counts = (
            pes.int.ntrans,
            pes.int.nbonds,
            pes.int.nangles,
            pes.int.ndihedrals,
            pes.int.nother,
            pes.int.nrotations,
        )
        nint = sum(counts)
        diag = np.abs(np.diag(pes.get_H().asarray()))[:nint]
        good = np.isfinite(diag) & (diag > 1e-12)
        if not np.any(good):
            return np.ones(nint)
        ref = np.median(diag[good])
        if not np.isfinite(ref) or ref <= 0.0:
            return np.ones(nint)
        weights = np.ones(nint)
        weights[good] = np.sqrt(diag[good] / ref)
        return np.clip(weights, 0.85, 1.2)

    def cons(self, s, dsda=None):
        w = np.array(
            [self.wx] * self.pes.int.ntrans
            + [self.wb] * self.pes.int.nbonds
            + [self.wa] * self.pes.int.nangles
            + [self.wd] * self.pes.int.ndihedrals
            + [self.wo] * self.pes.int.nother
            + [self.wx] * self.pes.int.nrotations
        )
        assert len(w) == len(s)
        if self.curvature_weighting:
            w = w * self.curvature_weights
        sw = np.abs(s * w)
        idx = np.argmax(np.abs(sw))
        val = sw[idx]
        if dsda is None:
            return val
        return (val, np.sign(s[idx]) * dsda[idx] * w[idx])


_all_restricted_step = [MaxInternalStep]


def get_restricted_step(name):
    for rs in _all_restricted_step:
        if rs.match(name):
            return rs
    raise ValueError("Unknown restricted step name: {}".format(name))


_minimum_defaults = dict(
    delta0=0.124, sigma_inc=1.41, sigma_dec=0.9, rho_inc=1.11, rho_dec=130, method="qn"
)


class Sella(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: bool = None,
        logfile: str = "-",
        master: bool = None,
        delta0: float = None,
        sigma_inc: float = None,
        sigma_dec: float = None,
        rho_dec: float = None,
        rho_inc: float = None,
        order: int = 0,
        eta: float = 0.0001,
        method: str = None,
        constraints: Constraints = None,
        constraints_tol: float = 1e-05,
        internal: Union[bool, Internals] = True,
        rs: str = None,
        **kwargs,
    ):
        if order != 0:
            raise ValueError("sella_minimal only supports minimization (order=0).")
        default = _minimum_defaults
        self.peskwargs = kwargs.copy()
        self.user_internal = internal
        self.initialize_pes(atoms, order, eta, constraints, internal, **kwargs)
        if rs is None:
            rs = "mis"
        self.rs = get_restricted_step(rs)
        Optimizer.__init__(
            self,
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=None,
            master=master,
        )
        if delta0 is None:
            delta0 = default["delta0"]
        if rs == "mis":
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]
        if sigma_inc is None:
            self.sigma_inc = default["sigma_inc"]
        else:
            self.sigma_inc = sigma_inc
        if sigma_dec is None:
            self.sigma_dec = default["sigma_dec"]
        else:
            self.sigma_dec = sigma_dec
        if rho_inc is None:
            self.rho_inc = default["rho_inc"]
        else:
            self.rho_inc = rho_inc
        if rho_dec is None:
            self.rho_dec = default["rho_dec"]
        else:
            self.rho_dec = rho_dec
        if method is None:
            self.method = default["method"]
        else:
            self.method = method
        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.rho = 1.0
        self.initialized = False
        self.xi = 1.0

    def initialize_pes(
        self,
        atoms: Atoms,
        order: int = 0,
        eta: float = 0.0001,
        constraints: Constraints = None,
        internal: Union[bool, Internals] = True,
        **kwargs,
    ):
        if not internal:
            raise ValueError("sella_minimal only supports internal coordinates.")
        if internal:
            if isinstance(internal, Internals):
                auto_find_internals = False
                if constraints is not None:
                    raise ValueError(
                        "Internals object and Constraint object cannot both be provided to Sella. Instead, you must pass the Constraints object to the constructor of the Internals object."
                    )
            else:
                auto_find_internals = True
                internal = Internals(atoms, cons=constraints)
            self.internal = internal.copy()
            self.constraints = None
            self.pes = InternalPES(
                atoms,
                internals=internal,
                eta=eta,
                auto_find_internals=auto_find_internals,
                **kwargs,
            )

    def _predict_step(self):
        if not self.initialized:
            self.pes.get_g()
            self.initialized = True
        self.pes.cons.disable_satisfied_inequalities()
        self.pes._update_basis()
        self.pes.save()
        all_valid = False
        x0 = self.pes.get_x()
        while not all_valid:
            s, smag = self.rs(
                self.pes, self.ord, self.delta, method=self.method
            ).get_s()
            self.pes.set_x(x0 + s)
            all_valid = self.pes.cons.validate_inequalities()
            self.pes._update_basis()
            self.pes.restore()
        self.pes._update_basis()
        return (s, smag)

    def step(self):
        s, smag = self._predict_step()
        rho = self.pes.kick(s, diag=False)
        if self.internal and self.pes.int.check_for_bad_internals():
            self.initialize_pes(
                atoms=self.pes.atoms,
                order=self.ord,
                eta=self.pes.eta,
                constraints=self.constraints,
                internal=self.user_internal,
            )
            self.initialized = False
            self.rho = 1
            return
        if rho is None:
            pass
        elif rho < 1.0 / self.rho_dec or rho > self.rho_dec:
            self.delta = max(smag * self.sigma_dec, self.delta_min)
        elif 1.0 / self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag, self.delta)
        self.rho = rho
        if self.rho is None:
            self.rho = 1.0


class _WrappedCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calc_func, **kwargs):
        super().__init__(**kwargs)
        self.calc_func = calc_func
        self.call_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)
        pos_nm = self.atoms.get_positions() / _NM_TO_ANG
        energy_kj, forces_kj_nm = self.calc_func(pos_nm)
        self.call_count += 1
        self.results["energy"] = energy_kj * _KJ_MOL_TO_EV
        self.results["forces"] = np.array(forces_kj_nm) * _KJ_MOL_NM_TO_EV_ANG


def converge_wrapper(converged_func):
    def wrapper(gradient):
        return converged_func()

    return wrapper


def minimize_func(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    calc: Callable,
    max_force_calls: int,
    converged: Callable,
) -> tuple[np.ndarray, int]:
    pos_ang = np.array(positions) * _NM_TO_ANG
    atoms = Atoms(numbers=atomic_numbers, positions=pos_ang)
    wrapper = _WrappedCalc(calc)
    atoms.calc = wrapper
    opt = Sella(atoms, logfile=None)
    opt.gradient_converged = converge_wrapper(converged)
    opt.run(steps=max_force_calls - 1)
    final_pos_nm = atoms.get_positions() / _NM_TO_ANG
    return (final_pos_nm, wrapper.call_count)


def entrypoint():
    return minimize_func
