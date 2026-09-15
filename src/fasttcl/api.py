"""A common physical interface for standalone TCL2, TCL4, and TCL6."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from .baths import SampledBath, _real_scalar
from .lower_orders import lower_order_generators


def _matrix(value, name):
    a = np.asarray(value, dtype=np.complex128)
    if a.ndim != 2 or not len(a) or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite, nonempty square matrix")
    if not np.allclose(a, a.conj().T, atol=1e-13, rtol=0):
        raise ValueError(f"{name} must be Hermitian")
    return a


@dataclass(frozen=True)
class Model:
    """One finite-dimensional Hamiltonian and Hermitian coupling operator."""
    hamiltonian: np.ndarray
    coupling_operator: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray
    coupling_energy: np.ndarray

    @property
    def dimension(self):
        return len(self.energies)


def prepare_model(hamiltonian, coupling_operator):
    """Diagonalize H and rotate the unscaled A; H_int=lambda*A tensor B."""
    h = _matrix(hamiltonian, "hamiltonian")
    a = _matrix(coupling_operator, "coupling_operator")
    if a.shape != h.shape:
        raise ValueError("Hamiltonian and coupling dimensions must match")
    e, u = np.linalg.eigh(h)
    u = np.asarray(u, dtype=np.complex128)
    for column in range(len(e)):
        pivot = int(np.argmax(np.abs(u[:, column])))
        u[:, column] *= np.exp(-1j*np.angle(u[pivot, column]))
        if u[pivot, column].real < 0:
            u[:, column] *= -1
    arrays = [h.copy(), a.copy(), e, u, u.conj().T @ a @ u]
    for array in arrays:
        array.setflags(write=False)
    return Model(*arrays)


@dataclass(frozen=True)
class TCL6Plan:
    """Reusable sixth-order preparation for one H,A; independent of bath/grid/lambda."""
    model: Model
    _engine_plan: Any

    @property
    def report(self):
        """Preparation settings, timings, and compiler diagnostics."""
        from copy import deepcopy
        return deepcopy(self._engine_plan.report)


def compile_plan(model, *, fusion="auto"):
    """Prepare CPU TCL6 independently of the bath and grid.

    ``auto`` uses exact fusion for sparse two-level models, the source plan
    for dense two-level models, and temporal reuse for larger models. Both
    paths evaluate the same HR expression. ``off`` and ``exact`` are explicit
    alternatives; cold dense exact fusion can take several minutes.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be returned by prepare_model")
    if fusion not in ("auto", "off", "exact"):
        raise ValueError("fusion must be 'auto', 'off', or 'exact'")
    from ._engine.runtime import compile_hr_plan
    from ._engine.spin_boson import EnergyBasisTwoLevelModel
    from ._engine.general_model import EnergyBasisModel
    e = model.energies
    if len(e) == 2 and e[1]-e[0] > 1e-12*max(1, np.max(np.abs(e))):
        internal = EnergyBasisTwoLevelModel(
            hamiltonian_lab=model.hamiltonian, coupling_lab=model.coupling_operator,
            energies=e, eigenvectors=model.eigenvectors, coupling_energy=model.coupling_energy,
        )
    else:
        internal = EnergyBasisModel(e, model.coupling_energy, eigenvectors=model.eigenvectors)
    selected = fusion
    if fusion == "auto":
        dense_two_level = isinstance(internal, EnergyBasisTwoLevelModel) and np.count_nonzero(model.coupling_energy) == 4
        selected = "off" if dense_two_level else "exact"
    prepared = compile_hr_plan(
        data_dir=Path(__file__).resolve().parent/"data", model=internal,
        fusion=selected, early_model_specialization=True, domain_completion="off",
    )
    prepared.report["public_fusion"] = {"requested": fusion, "selected": selected,
        "reason": "dense two-level source plan avoids symbolic rank preparation" if fusion == "auto" and selected == "off" else "explicit choice or sparse/general reuse",
        "sparsity_rule": "literal zero entries; no tolerance threshold"}
    return TCL6Plan(model, prepared)


def _grid(dt, n_steps, order, coupling_strength):
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    dt = _real_scalar(dt, "dt", positive=True)
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    _real_scalar(coupling_strength, "coupling_strength (lambda)")
    return float(dt)*np.arange(int(n_steps)+1)


@dataclass
class GeneratorSeries:
    """Schrodinger-picture generators in the energy basis, using column-major vec.

    ``corrections[k]`` includes lambda**k exactly once. ``free`` contains
    -i[H_S,.]. ``times`` contains the actual generator sampling grid.
    """
    times: np.ndarray
    free: np.ndarray
    corrections: dict[int, np.ndarray]
    model: Model
    coupling_strength: float
    metadata: dict

    def cumulative(self, order=None):
        if order is None:
            order = max(self.corrections)
        if isinstance(order, bool) or order not in self.corrections:
            raise ValueError("requested order was not calculated")
        result = np.broadcast_to(self.free, self.corrections[2].shape).copy()
        for k, correction in self.corrections.items():
            if k <= order:
                result += correction
        return result


def generator_series(model, bath, *, dt, n_steps, coupling_strength=1.0, order=6, plan=None):
    """Calculate generators at t=0,dt,...,n_steps*dt through the selected order.

    ``bath`` is a unit C_B callable, an object exposing correlation(times),
    or an array with n_steps+1 samples. The model's A excludes lambda.
    An optional ``plan=compile_plan(model)`` reuses sixth-order preparation.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be returned by prepare_model")
    times = _grid(dt, n_steps, order, coupling_strength)
    if hasattr(bath, "correlation"):
        values = bath.correlation(times)
    elif callable(bath):
        values = bath(times)
    else:
        values = bath
    values = np.asarray(values, dtype=np.complex128)
    if values.shape != times.shape:
        raise ValueError("bath must return exactly n_steps+1 correlation samples")
    sampled = SampledBath(values, dt)
    d = model.dimension
    omega = model.energies[:, None]-model.energies[None, :]
    omega_vec = omega.reshape(-1, order="F")
    free = np.diag(-1j*omega_vec)
    shape = (len(times), d*d, d*d)
    corrections = {k: np.zeros(shape, dtype=np.complex128) for k in (2, 4, 6) if k <= order}
    metadata = {"picture": "Schrodinger", "basis": "energy", "vectorization": "column-major",
                "bath": "unit C_B", "coupling": "lambda^k included once in correction k",
                "gamma_quadrature": "composite-trapezoid", "tcl4_quadrature": "composite-trapezoid"}
    if coupling_strength == 0 or not np.any(model.coupling_energy) or n_steps == 0:
        return GeneratorSeries(times, free, corrections, model, float(coupling_strength), metadata)
    l2, l4 = lower_order_generators(
        model.energies, model.coupling_energy, sampled.values, float(dt), max_order=min(order, 4),
    )
    corrections[2] = np.asarray(l2)*float(coupling_strength)**2
    if order >= 4:
        corrections[4] = np.asarray(l4)*float(coupling_strength)**4
    if order == 6:
        if plan is None:
            plan = compile_plan(model)
        if not isinstance(plan, TCL6Plan) or not all(np.array_equal(x, y) for x, y in (
            (plan.model.energies, model.energies), (plan.model.coupling_energy, model.coupling_energy),
            (plan.model.eigenvectors, model.eigenvectors),
        )):
            raise ValueError("TCL6 plan belongs to a different model")
        from ._engine.runtime import evaluate_hr_plan
        sixth = evaluate_hr_plan(plan._engine_plan, nt=len(times), dt=float(dt), bath=sampled, backend="numpy")
        phase = np.exp(-1j*times[:, None]*omega_vec[None, :])
        corrections[6] = phase[:, :, None]*np.asarray(sixth.generator)*phase.conj()[:, None, :]*float(coupling_strength)**6
        metadata["tcl6"] = sixth.report
        metadata["tcl6_preparation"] = {key: plan.report[key] for key in
            ("source_record_count", "compile_seconds", "public_fusion")}
    return GeneratorSeries(times, free, corrections, model, float(coupling_strength), metadata)


@dataclass
class Solution:
    """Density matrices in the input laboratory basis at the requested output times."""
    times: np.ndarray
    trajectories: dict[int, np.ndarray]
    generators: GeneratorSeries

    @property
    def rho(self):
        """Trajectory at the highest requested TCL order."""
        return self.trajectories[max(self.trajectories)]


def solve(model, bath, rho0, *, dt, n_steps, coupling_strength=1.0, order=6, plan=None):
    """Propagate TCL2 and each higher truncation up to ``order`` using midpoint RK4.

    Evaluates generators at dt/2, then returns density matrices at dt.
    Bath arrays must therefore have 2*n_steps+1 samples on that finer grid.
    No trace normalization, Hermitization, or positivity projection is applied.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be returned by prepare_model")
    _grid(dt, n_steps, order, coupling_strength)
    dt, n_steps = float(dt), int(n_steps)
    rho = _matrix(rho0, "rho0")
    if rho.shape != model.hamiltonian.shape or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0):
        raise ValueError("rho0 must match the model and have trace one")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("rho0 must be positive semidefinite")
    series = generator_series(model, bath, dt=float(dt)/2, n_steps=2*int(n_steps),
                              coupling_strength=coupling_strength, order=order, plan=plan)
    u = model.eigenvectors
    initial = (u.conj().T @ rho @ u).reshape(-1, order="F")
    curves = {}
    for truncation in series.corrections:
        total = series.cumulative(truncation)
        vectors = np.empty((n_steps+1, model.dimension**2), dtype=np.complex128)
        vectors[0] = initial
        for n in range(n_steps):
            y = vectors[n]
            l0, lh, l1 = total[2*n:2*n+3]
            k1 = l0 @ y
            k2 = lh @ (y+0.5*dt*k1)
            k3 = lh @ (y+0.5*dt*k2)
            k4 = l1 @ (y+dt*k3)
            vectors[n+1] = y+(dt/6)*(k1+2*k2+2*k3+k4)
        energy_rho = vectors.reshape(-1, model.dimension, model.dimension).transpose(0, 2, 1)
        curves[truncation] = u[None] @ energy_rho @ u.conj().T[None]
    return Solution(series.times[::2], curves, series)
