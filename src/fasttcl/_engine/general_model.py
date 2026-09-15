"""Energy-basis models and memory budgets for TCL6 calculations."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import math

from typing import Any

class HRResourceError(ValueError):
    """A requested exact calculation exceeds a declared resource limit."""


@dataclass(frozen=True)
class HRResourceBudget:
    """Array/work admission limits; host bytes include a conservative workspace reserve.

    These are planned allocation limits, not an operating-system RSS sandbox.
    Process/library overhead is reported separately from the dense output floor.
    """
    host_bytes: int = 2 * 1024**3
    device_bytes: int = 2 * 1024**3
    compile_bytes: int = 512 * 1024**2
    max_contributions: int = 2_000_000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def admit(self, name: str, required: int, limit: int) -> None:
        if required > limit:
            raise HRResourceError(f"{name} requires {required:,}; declared limit is {limit:,}")


def _owned(value, dtype):
    import numpy as np
    array = np.asarray(value, dtype=dtype)
    # Immutable bytes backing prevents callers re-enabling WRITEABLE on a plan's inputs.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False)
class EnergyBasisModel:
    """One Hermitian coupling in a specified finite-dimensional energy basis."""
    energies: Any
    coupling_energy: Any
    hamiltonian_lab: Any
    coupling_lab: Any
    eigenvectors: Any

    def __init__(self, energies, coupling_operator, *, eigenvectors=None):
        import numpy as np
        e = np.asarray(energies)
        if e.ndim != 1 or not e.size or np.iscomplexobj(e) and np.any(e.imag != 0):
            raise ValueError("energies must be a nonempty real vector")
        e = np.asarray(e.real, dtype=np.float64)
        a = np.asarray(coupling_operator, dtype=np.complex128)
        n = len(e)
        if a.shape != (n, n) or not np.isfinite(e).all() or not np.isfinite(a).all():
            raise ValueError("finite coupling dimensions must match energies")
        if not np.allclose(a, a.conj().T, atol=1e-13, rtol=0):
            raise ValueError("coupling operator must be Hermitian")
        u = np.eye(n, dtype=complex) if eigenvectors is None else np.asarray(eigenvectors, dtype=complex)
        if u.shape != (n,n) or not np.isfinite(u).all() or not np.allclose(u.conj().T @ u, np.eye(n), atol=1e-12, rtol=0):
            raise ValueError("eigenvectors must be a finite unitary matrix")
        for name, value, dtype in (
            ("energies", e, np.float64), ("coupling_energy", a, np.complex128),
            ("hamiltonian_lab", (u * e) @ u.conj().T, np.complex128),
            ("coupling_lab", u @ a @ u.conj().T, np.complex128),
            ("eigenvectors", u, np.complex128),
        ):
            object.__setattr__(self, name, _owned(value, dtype))

    @property
    def dimension(self):
        return len(self.energies)

    def level_ratios(self, dt):
        import numpy as np
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        return tuple(np.exp(1j * self.energies * dt))

    @property
    def report(self):
        digest = hashlib.sha256(self.energies.tobytes() + self.coupling_energy.tobytes()).hexdigest()
        return {"dimension": self.dimension, "model_digest": digest,
                "basis": "energy", "operator_vectorization": "column-major",
                "scope": "one Hermitian coupling, stationary Hermitian scalar bath"}


def prepare_hr_model(*, hamiltonian=None, coupling_operator, energies=None):
    """Prepare a general model, preserving an explicitly supplied energy basis.

    Exactly degenerate Hamiltonians are supported in the explicit energy basis.
    No sparsity is inferred by thresholding a numerically rotated coupling.
    """
    import numpy as np
    if (hamiltonian is None) == (energies is None):
        raise ValueError("provide exactly one of hamiltonian or energies")
    if energies is not None:
        return EnergyBasisModel(energies, coupling_operator)
    h = np.asarray(hamiltonian, dtype=complex)
    a = np.asarray(coupling_operator, dtype=complex)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1] or a.shape != h.shape:
        raise ValueError("Hamiltonian and coupling must be matching nonempty square matrices")
    if not np.isfinite(h).all() or not np.allclose(h, h.conj().T, atol=1e-13, rtol=0):
        raise ValueError("Hamiltonian must be finite and Hermitian")
    if not np.isfinite(a).all() or not np.allclose(a, a.conj().T, atol=1e-13, rtol=0):
        raise ValueError("coupling must be finite and Hermitian")
    if np.count_nonzero(h - np.diag(np.diag(h))) == 0:
        return EnergyBasisModel(h.diagonal().real, a)
    e, u = np.linalg.eigh(h)
    # Preserve NumPy's eigenspace basis explicitly in the returned model.
    return EnergyBasisModel(e, u.conj().T @ a @ u, eigenvectors=u)

