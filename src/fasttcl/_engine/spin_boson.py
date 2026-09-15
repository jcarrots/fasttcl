"""Two-level models, stationary bath kernels, and Bohr-frequency sequences."""

from __future__ import annotations

from collections import Counter

from collections.abc import Iterator, Mapping as MappingABC

from dataclasses import dataclass, field

import hashlib

from types import MappingProxyType

from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .convolution_runtime import _numpy

from .extended_evaluator import ScalarMonomialInventory

from .extended_runtime import KernelIdentity

class SpinBosonInputError(ValueError):
    """Raised when physical inputs do not define a valid two-level model."""


PHYSICAL_VW_SYMMETRY_PROFILE = "stationary-signed-gamma-vw-symmetry-v1"


PHYSICAL_EQUAL_GAP_PROFILE = "stationary-scalar-bath-equal-gap-v1"


SIGNED_GAMMA_EXPLICIT_CONSTRUCTION = "explicit-signed-correlation-trapezoid"


SIGNED_GAMMA_TACO_MIRROR_CONSTRUCTION = (
    "taco-positive-gamma-stationary-hermitian-mirror"
)


_ACTIVE_GAMMA_EXPLICIT_CONSTRUCTION = (
    "active-unique-bohr-gap-explicit-signed-trapezoid"
)


_ACTIVE_GAMMA_MIRROR_CONSTRUCTION = (
    "active-unique-bohr-gap-positive-trapezoid-hermitian-mirror"
)


PhysicalGammaKey = tuple[str, int, int]


_POSITIVE_GAMMA_SUPPORTS = frozenset({"K", "U", "V", "W", "UV", "VW"})


_NEGATIVE_GAMMA_SUPPORTS = frozenset({"-U", "-V", "-W", "-UV", "-VW"})


@dataclass(frozen=True)
class PhysicalInventoryReduction:
    """Exact monomial collection under the stationary signed-Gamma contract.

    The generic proof runtime deliberately treats every support-tagged Gamma
    sequence as independent.  Physical binding is stronger: for fixed matrix
    indices and sign, the V and W labels are views of the same stationary
    function at different integration variables.  The complete nested
    trapezoid is invariant under the global dummy-variable swap ``v <-> w``.
    This container records the resulting exact integer collection separately
    so the generic contract cannot be weakened accidentally.
    """

    monomials: tuple[tuple[tuple[Any, ...], int], ...]
    report: Mapping[str, Any]


def _swap_vw_support(support: str) -> str:
    sign = "-" if support.startswith("-") else ""
    base = support[1:] if sign else support
    swapped = {"V": "W", "W": "V"}.get(base, base)
    return sign + swapped


def _vw_swapped_monomial_key(key: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Return the integrated ``v <-> w`` image when it is representable.

    A UV factor would become a U+W lag, which is outside the compiled support
    alphabet, so those monomials remain untouched.  Signs, matrix indices,
    static factors, and output entries are never identified or approximated.
    """

    if len(key) != 6:
        raise SpinBosonInputError("malformed scalar monomial key")
    output_row, output_column, factors, phase_u, phase_v, phase_w = key
    kernels = [factor for factor in factors if factor[0] == "kernel"]
    if any(
        (factor[2][1:] if factor[2].startswith("-") else factor[2]) == "UV"
        for factor in kernels
    ):
        return None
    transformed = []
    for factor in factors:
        if factor[0] == "kernel":
            transformed.append(
                (factor[0], factor[1], _swap_vw_support(factor[2]), *factor[3:])
            )
        else:
            transformed.append(factor)
    return (
        output_row,
        output_column,
        tuple(sorted(transformed)),
        phase_u,
        phase_w,
        phase_v,
    )


def coalesce_stationary_gamma_vw_symmetry(
    monomials: ScalarMonomialInventory,
) -> PhysicalInventoryReduction:
    """Collect exact full-integral terms related by ``v <-> w``.

    This function is valid only for support labels bound to one stationary
    signed Gamma family, as performed by :func:`bind_gamma_sequences`.  It is
    intentionally not called by :func:`compile_streamed_plan` or any generic
    arbitrary-sequence evaluator.
    """

    collected: Counter[tuple[Any, ...]] = Counter()
    transformed_sources = 0
    for key, coefficient in monomials:
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise SpinBosonInputError(
                "scalar monomial coefficient must be an integer"
            )
        image = _vw_swapped_monomial_key(key)
        representative = min(key, image) if image is not None else key
        if image is not None and image != key:
            transformed_sources += representative != key
        collected[representative] += coefficient

    reduced = tuple(
        (key, coefficient)
        for key, coefficient in sorted(collected.items())
        if coefficient
    )
    canceled_classes = sum(value == 0 for value in collected.values())
    return PhysicalInventoryReduction(
        monomials=reduced,
        report={
            "profile": PHYSICAL_VW_SYMMETRY_PROFILE,
            "assumption": (
                "fixed-sign support labels share one stationary Gamma_rc curve"
            ),
            "source_scalar_monomials": len(monomials),
            "reduced_scalar_monomials": len(reduced),
            "eliminated_scalar_monomials": len(monomials) - len(reduced),
            "source_coefficient_l1": sum(abs(value) for _, value in monomials),
            "reduced_coefficient_l1": sum(abs(value) for _, value in reduced),
            "transformed_source_monomials": transformed_sources,
            "canceled_orbit_classes": canceled_classes,
            "positive_negative_signs_kept_distinct": True,
            "uv_bearing_monomials_kept_distinct": True,
        },
    )


def _equal_gap_monomial_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Canonicalize diagonal Gamma indices for the scalar stationary bath."""

    if len(key) != 6:
        raise SpinBosonInputError("malformed scalar monomial key")
    output_row, output_column, factors, phase_u, phase_v, phase_w = key
    transformed = []
    for factor in factors:
        if factor[0] == "kernel" and factor[3] == factor[4]:
            transformed.append((*factor[:3], 0, 0))
        else:
            transformed.append(factor)
    return (
        output_row,
        output_column,
        tuple(sorted(transformed)),
        phase_u,
        phase_v,
        phase_w,
    )


def coalesce_stationary_scalar_bath_equal_gaps(
    monomials: ScalarMonomialInventory,
) -> PhysicalInventoryReduction:
    """Collect exact Gamma identities with the same diagonal Bohr gap.

    In the declared single scalar-bath adapter, ``Gamma_rr`` depends on the
    system indices only through ``E_r-E_r=0``.  All diagonal entries are
    therefore the same signed curve.  This is a physical-model identity,
    never a rule for the generic arbitrary-sequence runtime.
    """

    collected: Counter[tuple[Any, ...]] = Counter()
    transformed_sources = 0
    for key, coefficient in monomials:
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise SpinBosonInputError(
                "scalar monomial coefficient must be an integer"
            )
        representative = _equal_gap_monomial_key(key)
        transformed_sources += representative != key
        collected[representative] += coefficient
    reduced = tuple(
        (key, coefficient)
        for key, coefficient in sorted(collected.items())
        if coefficient
    )
    return PhysicalInventoryReduction(
        monomials=reduced,
        report={
            "profile": PHYSICAL_EQUAL_GAP_PROFILE,
            "assumption": (
                "single stationary scalar bath: Gamma_rr depends on E_r-E_r=0"
            ),
            "source_scalar_monomials": len(monomials),
            "reduced_scalar_monomials": len(reduced),
            "eliminated_scalar_monomials": len(monomials) - len(reduced),
            "source_coefficient_l1": sum(abs(value) for _, value in monomials),
            "reduced_coefficient_l1": sum(abs(value) for _, value in reduced),
            "transformed_source_monomials": transformed_sources,
            "canceled_orbit_classes": sum(
                value == 0 for value in collected.values()
            ),
            "positive_negative_signs_kept_distinct": True,
            "only_exact_zero_bohr_gaps_identified": True,
        },
    )


def reduce_stationary_scalar_bath_inventory(
    monomials: ScalarMonomialInventory,
) -> PhysicalInventoryReduction:
    """Compose exact equal-gap and ``v <-> w`` physical reductions."""

    equal_gap = coalesce_stationary_scalar_bath_equal_gaps(monomials)
    vw = coalesce_stationary_gamma_vw_symmetry(equal_gap.monomials)
    return PhysicalInventoryReduction(
        monomials=vw.monomials,
        report={
            "profile": (
                f"{PHYSICAL_EQUAL_GAP_PROFILE}+{PHYSICAL_VW_SYMMETRY_PROFILE}"
            ),
            "assumptions": [
                equal_gap.report["assumption"],
                vw.report["assumption"],
            ],
            "source_scalar_monomials": len(monomials),
            "reduced_scalar_monomials": len(vw.monomials),
            "eliminated_scalar_monomials": len(monomials) - len(vw.monomials),
            "source_coefficient_l1": sum(abs(value) for _, value in monomials),
            "reduced_coefficient_l1": sum(
                abs(value) for _, value in vw.monomials
            ),
            "stages": {
                "equal_gap": dict(equal_gap.report),
                "vw_symmetry": dict(vw.report),
            },
            "generic_arbitrary_support_semantics_unchanged": True,
        },
    )


def stationary_scalar_bath_monomial_key(
    key: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Return the exact stationary-bath representative of one monomial.

    This is the streaming counterpart of
    :func:`reduce_stationary_scalar_bath_inventory`: diagonal zero-Bohr-gap
    kernels are identified first, followed by the admissible global
    ``v <-> w`` dummy-variable symmetry.
    """

    equal_gap = _equal_gap_monomial_key(key)
    image = _vw_swapped_monomial_key(equal_gap)
    return min(equal_gap, image) if image is not None else equal_gap


class CorrelationModel(Protocol):
    """A bath correlation model evaluated at explicit signed times."""

    def correlation(self, times: Any) -> Any:
        """Return ``C(t)`` for every signed value in ``times``."""

    @property
    def report(self) -> Mapping[str, Any]:
        """Return JSON-compatible model metadata."""


@dataclass(frozen=True)
class EnergyBasisTwoLevelModel:
    """Nondegenerate two-level Hamiltonian and coupling in its energy basis."""

    hamiltonian_lab: Any
    coupling_lab: Any
    energies: Any
    eigenvectors: Any
    coupling_energy: Any

    def __post_init__(self) -> None:
        """Own immutable, finite NumPy copies of every model array."""

        np = _numpy()
        specifications = (
            ("hamiltonian_lab", np.complex128, (2, 2)),
            ("coupling_lab", np.complex128, (2, 2)),
            ("energies", np.float64, (2,)),
            ("eigenvectors", np.complex128, (2, 2)),
            ("coupling_energy", np.complex128, (2, 2)),
        )
        for name, dtype, shape in specifications:
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.shape != shape:
                raise SpinBosonInputError(
                    f"{name} must have shape {shape}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise SpinBosonInputError(f"{name} contains a nonfinite value")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def level_ratios(self, dt: float) -> tuple[complex, complex]:
        """Return ``exp(+i E_r dt)`` for the interaction-picture runtime."""

        _validate_dt(dt)
        np = _numpy()
        values = np.exp(1j * self.energies * dt)
        return complex(values[0]), complex(values[1])

    @property
    def report(self) -> dict[str, Any]:
        return {
            "dimension": 2,
            "energies": [float(value) for value in self.energies],
            "energy_gap": float(self.energies[1] - self.energies[0]),
            "picture": "interaction",
            "operator_phase": "A_rc(t)=A_rc exp(+i(E_r-E_c)t)",
            "hbar": 1,
            "eigenvectors_lab_columns": [
                [
                    {"real": float(value.real), "imag": float(value.imag)}
                    for value in row
                ]
                for row in self.eigenvectors
            ],
        }


@dataclass(frozen=True)
class DampedModeBath:
    r"""Thermal damped-mode correlation with an explicit signed-time law.

    The convention is

    ``C(t)=strength*exp(-decay*abs(t))*((n+1)exp(-i*w*t)+n exp(+i*w*t))``

    where ``n = 1/(exp(beta*w)-1)``.  It obeys ``C(-t)=conj(C(t))`` and has a
    finite, real value at the origin.  ``strength`` is already part of the
    correlation and must not be multiplied into the final generator again.
    """

    strength: float = 0.05
    decay_rate: float = 2.0
    mode_frequency: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        for name in ("strength", "decay_rate", "mode_frequency", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpinBosonInputError(f"{name} must be a real scalar")
        if self.strength < 0:
            raise SpinBosonInputError("strength must be nonnegative")
        if self.decay_rate < 0:
            raise SpinBosonInputError("decay_rate must be nonnegative")
        if self.mode_frequency <= 0:
            raise SpinBosonInputError("mode_frequency must be positive")
        if self.beta <= 0:
            raise SpinBosonInputError("beta must be positive")
        np = _numpy()
        if not all(
            np.isfinite(getattr(self, name))
            for name in ("strength", "decay_rate", "mode_frequency", "beta")
        ):
            raise SpinBosonInputError("bath parameters must be finite")

    @property
    def occupation(self) -> float:
        np = _numpy()
        argument = self.beta * self.mode_frequency
        if argument > 700:
            return 0.0
        return float(1.0 / np.expm1(argument))

    def correlation(self, times: Any) -> Any:
        np = _numpy()
        values = np.asarray(times, dtype=np.float64)
        if not np.isfinite(values).all():
            raise SpinBosonInputError("correlation times must be finite")
        occupation = self.occupation
        envelope = np.exp(-self.decay_rate * np.abs(values))
        forward = np.exp(-1j * self.mode_frequency * values)
        backward = np.exp(1j * self.mode_frequency * values)
        return np.asarray(
            self.strength
            * envelope
            * ((occupation + 1.0) * forward + occupation * backward),
            dtype=np.complex128,
        )

    @property
    def report(self) -> dict[str, Any]:
        return {
            "model": "phenomenological-thermal-damped-mode",
            "correlation_convention": (
                "strength exp(-decay |t|) "
                "[(n+1) exp(-i omega t)+n exp(+i omega t)]"
            ),
            "strength": float(self.strength),
            "decay_rate": float(self.decay_rate),
            "mode_frequency": float(self.mode_frequency),
            "beta": float(self.beta),
            "thermal_occupation": self.occupation,
            "stationary_hermitian_symmetry": "C(-t)=conj(C(t))",
            "strength_already_in_correlation": True,
        }


@dataclass(frozen=True)
class SignedGammaGrid:
    """Positive- and negative-argument ordinary Gamma matrices."""

    positive: Any
    negative: Any
    dt: float
    correlation_positive: Any
    correlation_negative: Any
    signed_grid_construction: str

    def __post_init__(self) -> None:
        allowed = {
            SIGNED_GAMMA_EXPLICIT_CONSTRUCTION,
            SIGNED_GAMMA_TACO_MIRROR_CONSTRUCTION,
        }
        if self.signed_grid_construction not in allowed:
            raise SpinBosonInputError(
                "unknown signed Gamma construction provenance "
                f"{self.signed_grid_construction!r}"
            )

    @property
    def nt(self) -> int:
        return int(self.positive.shape[0])

    @property
    def report(self) -> dict[str, Any]:
        np = _numpy()
        if (
            self.positive.ndim != 3
            or self.positive.shape[1] != self.positive.shape[2]
            or self.negative.shape != self.positive.shape
        ):
            raise SpinBosonInputError(
                "Gamma arrays must have matching shape (Nt,d,d)"
            )
        dimension = int(self.positive.shape[1])
        symmetry_error = float(
            np.max(np.abs(self.negative + self.positive.conj()))
        )
        negative_explicit = (
            self.signed_grid_construction
            == SIGNED_GAMMA_EXPLICIT_CONSTRUCTION
        )
        return {
            "nt": self.nt,
            "dt": float(self.dt),
            "shape": [self.nt, dimension, dimension],
            "quadrature": "cumulative trapezoid",
            "definition": "Gamma_rc(s)=integral_0^s C(tau) exp(+i(E_r-E_c)tau) d tau",
            "signed_grid_construction": self.signed_grid_construction,
            "negative_lags_evaluated_explicitly": negative_explicit,
            "negative_gamma_construction": (
                "cumulative trapezoid over explicitly evaluated C(-t)"
                if negative_explicit
                else "Gamma(-t)=-conj(Gamma(+t))"
            ),
            "negative_correlation_construction": (
                "correlation callback evaluated at -t"
                if negative_explicit
                else "C(-t)=conj(C(+t))"
            ),
            "gamma_zero_exact": bool(
                np.count_nonzero(self.positive[0]) == 0
                and np.count_nonzero(self.negative[0]) == 0
            ),
            "physical_signed_symmetry_max_abs_error": symmetry_error,
            "materialization": "full-signed-matrix-grid",
            "full_signed_grid_materialized": True,
            "materialized_signed_curve_count": 2 * dimension * dimension,
            "signed_curve_integration_count": (
                2 * dimension * dimension
                if negative_explicit
                else dimension * dimension
            ),
            "persistent_gamma_bytes": int(
                self.positive.nbytes + self.negative.nbytes
            ),
            "retained_correlation_bytes": int(
                self.correlation_positive.nbytes
                + self.correlation_negative.nbytes
            ),
        }


@dataclass(frozen=True)
class BoundGammaSequences:
    """Semantic runtime keys sharing the physical signed Gamma buffers."""

    sequences: Mapping[KernelIdentity, Any]
    unique_curve_count: int
    report: Mapping[str, Any]
    semantic_to_physical: Mapping[KernelIdentity, PhysicalGammaKey] = field(
        default_factory=dict
    )
    physical_sequences: Mapping[PhysicalGammaKey, Any] = field(
        default_factory=dict
    )
    binding_profile: str = PHYSICAL_EQUAL_GAP_PROFILE
    mapping_sha256: str = ""


@dataclass(frozen=True)
class _ActiveSignedGammaGrid:
    """Report carrier for an active physical Gamma bank.

    Unlike :class:`SignedGammaGrid`, this private-runtime representation does
    not promise dense ``(Nt,2,2)`` positive and negative matrices.  It keeps
    only the signed, oriented Bohr-gap curves referenced by the compiled plan.
    """

    nt: int
    dt: float
    construction_report: Mapping[str, Any]

    @property
    def report(self) -> dict[str, Any]:
        return dict(self.construction_report)


@dataclass(frozen=True)
class _PhysicalGammaSequenceMap(MappingABC[KernelIdentity, Any]):
    """Read-only semantic view over a compact physical Gamma curve bank."""

    semantic_sequences: Mapping[KernelIdentity, Any]
    semantic_to_physical: Mapping[KernelIdentity, PhysicalGammaKey]
    physical_sequences: Mapping[PhysicalGammaKey, Any]
    packed_bank: Any | None = None
    packed_rows: Mapping[PhysicalGammaKey, int] | None = None

    def __getitem__(self, key: KernelIdentity) -> Any:
        return self.semantic_sequences[key]

    def __iter__(self) -> Iterator[KernelIdentity]:
        return iter(self.semantic_sequences)

    def __len__(self) -> int:
        return len(self.semantic_sequences)


def _validate_dt(dt: float) -> None:
    np = _numpy()
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise SpinBosonInputError("dt must be a positive real scalar")


def _matrix2(value: Sequence[Sequence[complex]], *, name: str) -> Any:
    np = _numpy()
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise SpinBosonInputError(f"{name} must be a 2-by-2 matrix")
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise SpinBosonInputError(f"{name} must contain only finite values")
    return matrix


def prepare_two_level_model(
    hamiltonian_lab: Sequence[Sequence[complex]],
    coupling_lab: Sequence[Sequence[complex]],
    *,
    degeneracy_tolerance: float = 1e-12,
) -> EnergyBasisTwoLevelModel:
    """Diagonalize a Hermitian two-level system with deterministic phases."""

    np = _numpy()
    hamiltonian = _matrix2(hamiltonian_lab, name="hamiltonian_lab")
    coupling = _matrix2(coupling_lab, name="coupling_lab")
    if not np.allclose(hamiltonian, hamiltonian.conj().T, rtol=0, atol=1e-13):
        raise SpinBosonInputError("hamiltonian_lab must be Hermitian")
    if not np.allclose(coupling, coupling.conj().T, rtol=0, atol=1e-13):
        raise SpinBosonInputError("coupling_lab must be Hermitian")
    if (
        isinstance(degeneracy_tolerance, bool)
        or not isinstance(degeneracy_tolerance, (int, float))
        or not np.isfinite(degeneracy_tolerance)
        or degeneracy_tolerance < 0
    ):
        raise SpinBosonInputError(
            "degeneracy_tolerance must be a nonnegative real scalar"
        )

    energies, vectors = np.linalg.eigh(hamiltonian)
    scale = max(1.0, float(np.max(np.abs(energies))))
    if float(energies[1] - energies[0]) <= degeneracy_tolerance * scale:
        raise SpinBosonInputError(
            "the Hamiltonian is degenerate; supply a nondegenerate model so "
            "the Hadamard energy basis is unambiguous"
        )

    vectors = np.asarray(vectors, dtype=np.complex128)
    for column in range(2):
        magnitudes = np.abs(vectors[:, column])
        pivot = int(np.flatnonzero(magnitudes == magnitudes.max())[0])
        phase = np.angle(vectors[pivot, column])
        vectors[:, column] *= np.exp(-1j * phase)
        if vectors[pivot, column].real < 0:
            vectors[:, column] *= -1

    coupling_energy = vectors.conj().T @ coupling @ vectors
    return EnergyBasisTwoLevelModel(
        hamiltonian_lab=hamiltonian,
        coupling_lab=coupling,
        energies=np.asarray(energies, dtype=np.float64),
        eigenvectors=vectors,
        coupling_energy=np.asarray(coupling_energy, dtype=np.complex128),
    )


def prepare_spin_boson_model(
    *,
    epsilon: float = 0.0,
    delta: float = 1.0,
    coupling_scale: float = 1.0,
) -> EnergyBasisTwoLevelModel:
    r"""Return ``Hs=(epsilon*sigma_z+delta*sigma_x)/2``, ``A=g*sigma_z``."""

    np = _numpy()
    for name, value in (
        ("epsilon", epsilon),
        ("delta", delta),
        ("coupling_scale", coupling_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpinBosonInputError(f"{name} must be a real scalar")
        if not np.isfinite(value):
            raise SpinBosonInputError(f"{name} must be finite")
    sigma_x = np.asarray(((0, 1), (1, 0)), dtype=np.complex128)
    sigma_z = np.asarray(((1, 0), (0, -1)), dtype=np.complex128)
    model = prepare_two_level_model(
        0.5 * (epsilon * sigma_z + delta * sigma_x),
        coupling_scale * sigma_z,
    )
    # These zeros follow analytically from the declared sigma_x/sigma_z
    # model; they are not a tolerance-based approximation.  Keeping the
    # structural zeros exact enables exact static pruning without erasing a
    # genuinely small coupling in the generic model adapter.
    coupling_energy = model.coupling_energy.copy()
    if epsilon == 0:
        coupling_energy[0, 0] = 0
        coupling_energy[1, 1] = 0
    if delta == 0:
        coupling_energy[0, 1] = 0
        coupling_energy[1, 0] = 0
    return EnergyBasisTwoLevelModel(
        hamiltonian_lab=model.hamiltonian_lab,
        coupling_lab=model.coupling_lab,
        energies=model.energies,
        eigenvectors=model.eigenvectors,
        coupling_energy=coupling_energy,
    )


def physical_gamma_key(
    identity: KernelIdentity, *, dimension: int = 2
) -> PhysicalGammaKey:
    """Return the exact physical signed-curve key for one semantic identity.

    This quotient is deliberately applied only in the stationary scalar-bath
    adapter.  Support labels such as ``U``, ``V``, and ``W`` remain distinct
    in the HR and exact-fusion plans; only their bound one-dimensional arrays
    share a physical key here.
    """

    if len(identity) != 3:
        raise SpinBosonInputError(f"malformed Gamma identity {identity!r}")
    support, row, column = identity
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise SpinBosonInputError("dimension must be a positive integer")
    if row not in range(dimension) or column not in range(dimension):
        raise SpinBosonInputError(
            f"Gamma identity lies outside d={dimension}: {identity!r}"
        )
    if support in _POSITIVE_GAMMA_SUPPORTS:
        sign = "positive"
    elif support in _NEGATIVE_GAMMA_SUPPORTS:
        sign = "negative"
    else:
        raise SpinBosonInputError(f"unknown signed Gamma support {support!r}")
    # All diagonal entries have exactly the same zero Bohr frequency for a
    # stationary scalar bath.  Off-diagonal orientations are never merged.
    physical_row, physical_column = (
        (0, 0) if row == column else (row, column)
    )
    return sign, physical_row, physical_column


def _gamma_binding_layout(
    identities: Iterable[KernelIdentity],
    *,
    dimension: int = 2,
) -> tuple[
    dict[KernelIdentity, PhysicalGammaKey],
    tuple[PhysicalGammaKey, ...],
]:
    semantic_to_physical = {
        identity: physical_gamma_key(identity, dimension=dimension)
        for identity in sorted(set(identities))
    }
    return semantic_to_physical, tuple(
        sorted(set(semantic_to_physical.values()))
    )


def _gamma_binding_digest(
    semantic_to_physical: Mapping[KernelIdentity, PhysicalGammaKey],
) -> str:
    payload = tuple(sorted(semantic_to_physical.items()))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _bound_gamma_sequences(
    semantic_to_physical: Mapping[KernelIdentity, PhysicalGammaKey],
    physical_sequences: Mapping[PhysicalGammaKey, Any],
    *,
    packed_bank: Any | None = None,
    packed_rows: Mapping[PhysicalGammaKey, int] | None = None,
) -> BoundGammaSequences:
    semantic = {
        identity: physical_sequences[physical_key]
        for identity, physical_key in semantic_to_physical.items()
    }
    alias_counts = Counter(semantic_to_physical.values())
    mapping_digest = _gamma_binding_digest(semantic_to_physical)
    semantic_proxy = MappingProxyType(semantic)
    semantic_to_physical_proxy = MappingProxyType(dict(semantic_to_physical))
    physical_proxy = MappingProxyType(dict(physical_sequences))
    packed_rows_proxy = (
        None if packed_rows is None else MappingProxyType(dict(packed_rows))
    )
    sequence_map = _PhysicalGammaSequenceMap(
        semantic_sequences=semantic_proxy,
        semantic_to_physical=semantic_to_physical_proxy,
        physical_sequences=physical_proxy,
        packed_bank=packed_bank,
        packed_rows=packed_rows_proxy,
    )
    report = MappingProxyType(
        {
            "semantic_identity_count": len(semantic),
            "unique_physical_curve_count": len(physical_sequences),
            "physical_alias_class_sizes": sorted(alias_counts.values()),
            "physical_binding_profile": PHYSICAL_EQUAL_GAP_PROFILE,
            "physical_binding_mapping_sha256": mapping_digest,
            "transpose_handling": (
                "already mapped to Gamma[c,r] by scalarization"
            ),
            "signed_supports_are_explicit": True,
            "equal_diagonal_bohr_gaps_share_buffers": True,
        }
    )
    return BoundGammaSequences(
        sequences=sequence_map,
        unique_curve_count=len(physical_sequences),
        report=report,
        semantic_to_physical=semantic_to_physical_proxy,
        physical_sequences=physical_proxy,
        binding_profile=PHYSICAL_EQUAL_GAP_PROFILE,
        mapping_sha256=mapping_digest,
    )


def _sample_signed_correlation(
    correlation: Callable[[Any], Any],
    *,
    nt: int,
    dt: float,
) -> tuple[Any, Any, Any]:
    np = _numpy()
    x = dt * np.arange(nt, dtype=np.float64)
    c_plus = np.asarray(correlation(x), dtype=np.complex128)
    c_minus = np.asarray(correlation(-x), dtype=np.complex128)
    if c_plus.shape != (nt,) or c_minus.shape != (nt,):
        raise SpinBosonInputError(
            "correlation must return one complex value per input time"
        )
    for name, values in (("C(+t)", c_plus), ("C(-t)", c_minus)):
        if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
            raise SpinBosonInputError(f"{name} contains a nonfinite value")
    return x, c_plus, c_minus


def _has_exact_stationary_hermitian_samples(
    c_plus: Any,
    c_minus: Any,
    *,
    chunk_size: int = 65_536,
) -> bool:
    """Check exact signed symmetry with bounded temporary storage."""

    np = _numpy()
    for start in range(0, int(c_plus.size), chunk_size):
        stop = min(start + chunk_size, int(c_plus.size))
        if not np.array_equal(
            c_minus[start:stop], c_plus[start:stop].conj()
        ):
            return False
    return True


def _integrate_gamma_curve(
    correlation_values: Any,
    x: Any,
    *,
    omega: float,
    dt: float,
    sign: str,
) -> Any:
    np = _numpy()
    result = np.zeros(int(x.size), dtype=np.complex128)
    if x.size > 1:
        if sign == "positive":
            integrand = correlation_values * np.exp(1j * omega * x)
            multiplier = dt
        elif sign == "negative":
            integrand = correlation_values * np.exp(-1j * omega * x)
            multiplier = -dt
        else:  # pragma: no cover - all callers use validated physical keys
            raise SpinBosonInputError(f"unknown Gamma sign {sign!r}")
        result[1:] = multiplier * np.cumsum(
            0.5 * (integrand[:-1] + integrand[1:]),
            dtype=np.complex128,
        )
    result[0] = 0
    return result


def integrate_signed_gamma(
    energies: Sequence[float],
    correlation: Callable[[Any], Any],
    *,
    nt: int,
    dt: float,
) -> SignedGammaGrid:
    """Integrate every ``Gamma_rc`` curve on both signed half-grids."""

    np = _numpy()
    if isinstance(nt, bool) or not isinstance(nt, int) or nt <= 0:
        raise SpinBosonInputError("nt must be a positive integer")
    _validate_dt(dt)
    energy = np.asarray(energies, dtype=np.float64)
    if energy.ndim != 1 or energy.size == 0 or not np.isfinite(energy).all():
        raise SpinBosonInputError(
            "energies must be a nonempty one-dimensional finite array"
        )
    dimension = int(energy.size)

    x = dt * np.arange(nt, dtype=np.float64)
    c_plus = np.asarray(correlation(x), dtype=np.complex128)
    c_minus = np.asarray(correlation(-x), dtype=np.complex128)
    if c_plus.shape != (nt,) or c_minus.shape != (nt,):
        raise SpinBosonInputError(
            "correlation must return one complex value per input time"
        )
    for name, values in (("C(+t)", c_plus), ("C(-t)", c_minus)):
        if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
            raise SpinBosonInputError(f"{name} contains a nonfinite value")

    positive = np.zeros((nt, dimension, dimension), dtype=np.complex128)
    negative = np.zeros((nt, dimension, dimension), dtype=np.complex128)
    if nt > 1:
        for row in range(dimension):
            for column in range(dimension):
                omega = energy[row] - energy[column]
                f_plus = c_plus * np.exp(1j * omega * x)
                f_minus = c_minus * np.exp(-1j * omega * x)
                positive[1:, row, column] = dt * np.cumsum(
                    0.5 * (f_plus[:-1] + f_plus[1:]),
                    dtype=np.complex128,
                )
                negative[1:, row, column] = -dt * np.cumsum(
                    0.5 * (f_minus[:-1] + f_minus[1:]),
                    dtype=np.complex128,
                )
    positive[0] = 0
    negative[0] = 0
    return SignedGammaGrid(
        positive=positive,
        negative=negative,
        dt=float(dt),
        correlation_positive=c_plus,
        correlation_negative=c_minus,
        signed_grid_construction=SIGNED_GAMMA_EXPLICIT_CONSTRUCTION,
    )


def bind_gamma_sequences(
    gamma: SignedGammaGrid,
    identities: Iterable[KernelIdentity],
) -> BoundGammaSequences:
    """Bind support-tagged identities to shared physical Gamma matrix views."""

    if (
        gamma.positive.ndim != 3
        or gamma.positive.shape != gamma.negative.shape
        or gamma.positive.shape[1] != gamma.positive.shape[2]
    ):
        raise SpinBosonInputError(
            "signed Gamma arrays must have matching shape (Nt,d,d)"
        )
    dimension = int(gamma.positive.shape[1])
    semantic_to_physical, physical_keys = _gamma_binding_layout(
        identities, dimension=dimension
    )
    physical_views = {
        key: (
            gamma.positive[:, key[1], key[2]]
            if key[0] == "positive"
            else gamma.negative[:, key[1], key[2]]
        )
        for key in physical_keys
    }
    return _bound_gamma_sequences(
        semantic_to_physical,
        physical_views,
    )


def _build_active_physical_gamma_sequences(
    model: EnergyBasisTwoLevelModel,
    bath: CorrelationModel,
    identities: Iterable[KernelIdentity],
    *,
    nt: int,
    dt: float,
    construction: str = "auto",
) -> tuple[_ActiveSignedGammaGrid, BoundGammaSequences]:
    """Build only the physical signed curves required by a compiled plan.

    ``construction='explicit'`` preserves independently evaluated positive and
    negative callbacks and trapezoids.  ``'mirror'`` requires exact sampled
    stationary-Hermitian symmetry and computes only positive trapezoids.
    ``'auto'`` selects the mirror path only when that exact sample certificate
    holds; asymmetric custom callbacks therefore retain explicit behavior.
    """

    np = _numpy()
    if isinstance(nt, bool) or not isinstance(nt, int) or nt <= 0:
        raise SpinBosonInputError("nt must be a positive integer")
    _validate_dt(dt)
    if construction not in {"auto", "explicit", "mirror"}:
        raise SpinBosonInputError(
            "active Gamma construction must be 'auto', 'explicit', or 'mirror'"
        )
    energy = np.asarray(model.energies, dtype=np.float64)
    if energy.ndim != 1 or not energy.size or not np.isfinite(energy).all():
        raise SpinBosonInputError("model energies must contain finite values")

    semantic_to_physical, physical_keys = _gamma_binding_layout(identities, dimension=len(energy))
    oriented_pairs = tuple(sorted({(key[1], key[2]) for key in physical_keys}))
    x, c_plus, c_minus = _sample_signed_correlation(
        bath.correlation,
        nt=nt,
        dt=float(dt),
    )
    exact_stationary_samples = _has_exact_stationary_hermitian_samples(
        c_plus, c_minus
    )
    if construction == "mirror" and not exact_stationary_samples:
        raise SpinBosonInputError(
            "mirror Gamma construction requires exact sampled "
            "C(-t)=conj(C(+t)) symmetry"
        )
    selected = (
        "mirror"
        if construction == "mirror"
        or (construction == "auto" and exact_stationary_samples)
        else "explicit"
    )

    packed = np.zeros((len(physical_keys), nt), dtype=np.complex128)
    packed_rows = {key: row for row, key in enumerate(physical_keys)}
    integration_count = 0
    if selected == "explicit":
        for key, packed_row in packed_rows.items():
            sign, row, column = key
            omega = float(energy[row] - energy[column])
            packed[packed_row] = _integrate_gamma_curve(
                c_plus if sign == "positive" else c_minus,
                x,
                omega=omega,
                dt=float(dt),
                sign=sign,
            )
            integration_count += 1
        construction_name = _ACTIVE_GAMMA_EXPLICIT_CONSTRUCTION
        negative_gamma_construction = (
            "cumulative trapezoid over explicitly evaluated C(-t)"
        )
    else:
        positive_by_pair: dict[tuple[int, int], Any] = {}
        for row, column in oriented_pairs:
            omega = float(energy[row] - energy[column])
            positive_by_pair[(row, column)] = _integrate_gamma_curve(
                c_plus,
                x,
                omega=omega,
                dt=float(dt),
                sign="positive",
            )
            integration_count += 1
        for key, packed_row in packed_rows.items():
            sign, row, column = key
            positive_values = positive_by_pair[(row, column)]
            if sign == "positive":
                packed[packed_row] = positive_values
            else:
                # -conj(a+ib) = -a+ib.  Assign components to avoid another
                # full-length complex temporary at large Nt.
                packed[packed_row].real = -positive_values.real
                packed[packed_row].imag = positive_values.imag
        construction_name = _ACTIVE_GAMMA_MIRROR_CONSTRUCTION
        negative_gamma_construction = "Gamma(-t)=-conj(Gamma(+t))"

    if packed.size:
        packed[:, 0] = 0
    packed.setflags(write=False)
    physical_sequences = {
        key: packed[packed_row]
        for key, packed_row in packed_rows.items()
    }
    bound = _bound_gamma_sequences(
        semantic_to_physical,
        physical_sequences,
        packed_bank=packed,
        packed_rows=packed_rows,
    )
    if physical_keys:
        signed_symmetry_errors = []
        for row, column in oriented_pairs:
            positive_key = ("positive", row, column)
            negative_key = ("negative", row, column)
            if positive_key in physical_sequences and negative_key in physical_sequences:
                signed_symmetry_errors.append(
                    float(
                        np.max(
                            np.abs(
                                physical_sequences[negative_key]
                                + physical_sequences[positive_key].conj()
                            )
                        )
                    )
                )
        symmetry_error = max(signed_symmetry_errors, default=0.0)
    else:
        symmetry_error = 0.0
    full_gamma_bytes = int(2 * len(energy)**2 * nt * np.dtype(np.complex128).itemsize)
    report = MappingProxyType(
        {
            "nt": nt,
            "dt": float(dt),
            "shape": [nt, len(energy), len(energy)],
            "storage_shape": [len(physical_keys), nt],
            "quadrature": "cumulative trapezoid",
            "definition": (
                "Gamma_rc(s)=integral_0^s C(tau) "
                "exp(+i(E_r-E_c)tau) d tau"
            ),
            "signed_grid_construction": construction_name,
            "requested_active_gamma_construction": construction,
            "selected_active_gamma_construction": selected,
            "negative_lags_evaluated_explicitly": True,
            "negative_gamma_construction": negative_gamma_construction,
            "negative_correlation_construction": (
                "correlation callback evaluated at -t"
            ),
            "exact_stationary_hermitian_sample_certificate": (
                exact_stationary_samples
            ),
            "auto_fell_back_to_explicit": bool(
                construction == "auto" and selected == "explicit"
            ),
            "gamma_zero_exact": bool(
                not packed.size or np.count_nonzero(packed[:, 0]) == 0
            ),
            "physical_signed_symmetry_max_abs_error": symmetry_error,
            "materialization": "active-unique-bohr-gap-bank",
            "full_signed_grid_materialized": False,
            "active_unique_bohr_gap_count": len(oriented_pairs),
            "signed_curve_integration_count": integration_count,
            "materialized_signed_curve_count": len(physical_keys),
            "full_signed_grid_curve_count": 2 * len(energy)**2,
            "avoided_signed_curve_integrations": 2 * len(energy)**2 - integration_count,
            "persistent_gamma_bytes": int(packed.nbytes),
            "full_signed_grid_gamma_bytes": full_gamma_bytes,
            "persistent_gamma_byte_reduction": full_gamma_bytes - int(packed.nbytes),
            "retained_correlation_bytes": 0,
            "correlation_samples_released_after_binding": True,
        }
    )
    return (
        _ActiveSignedGammaGrid(
            nt=nt,
            dt=float(dt),
            construction_report=report,
        ),
        bound,
    )


def build_physical_gamma_sequences(
    model: EnergyBasisTwoLevelModel,
    bath: CorrelationModel,
    identities: Iterable[KernelIdentity],
    *,
    nt: int,
    dt: float,
) -> tuple[SignedGammaGrid, BoundGammaSequences]:
    """Convenience composition of signed integration and semantic binding."""

    grid = integrate_signed_gamma(
        model.energies, bath.correlation, nt=nt, dt=dt
    )
    return grid, bind_gamma_sequences(grid, identities)

