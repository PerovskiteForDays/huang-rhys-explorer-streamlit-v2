"""Numerical Huang–Rhys progression model.

The UI uses a common full width at half maximum (FWHM) for both line shapes.
This makes a Gaussian/Lorentzian toggle physically interpretable: switching the
shape changes the tails, not the displayed peak width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class SpectrumResult:
    """Calculated spectrum and its individual phonon replicas."""

    energy_eV: np.ndarray
    total: np.ndarray
    replicas: np.ndarray
    centers_eV: np.ndarray
    weights: np.ndarray
    replica_fwhm_eV: np.ndarray


@dataclass(frozen=True)
class MultiSpectrumResult:
    """Summed progression from one primary and zero or more additional ZPLs."""

    energy_eV: np.ndarray
    total: np.ndarray
    replicas: np.ndarray
    centers_eV: np.ndarray
    weights: np.ndarray
    replica_fwhm_eV: np.ndarray
    branch_totals: np.ndarray
    zpl_energies_eV: np.ndarray
    relative_amplitudes: np.ndarray
    huang_rhys_factors: np.ndarray
    phonon_energies_eV: np.ndarray
    base_fwhm_eV: np.ndarray
    broadening_multipliers: np.ndarray
    replica_branch_indices: np.ndarray
    replica_orders: np.ndarray


BranchParameter = float | np.ndarray | list[float] | tuple[float, ...]


def _branch_values(
    value: BranchParameter, branch_count: int, name: str
) -> np.ndarray:
    """Return one finite value per ZPL, broadcasting a scalar when needed."""

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(branch_count, float(values), dtype=float)
    elif values.ndim != 1 or values.size != branch_count:
        raise ValueError(
            f"{name} must be a scalar or a 1D array with one value per ZPL branch"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def gaussian_profile(
    energy_eV: np.ndarray, center_eV: float, fwhm_eV: float
) -> np.ndarray:
    """Return a unit-area Gaussian described by its FWHM."""

    fwhm_eV = max(float(fwhm_eV), np.finfo(float).eps)
    sigma = fwhm_eV * _FWHM_TO_SIGMA
    offset = (energy_eV - center_eV) / sigma
    return np.exp(-0.5 * offset**2) / (sigma * np.sqrt(2.0 * np.pi))


def lorentzian_profile(
    energy_eV: np.ndarray, center_eV: float, fwhm_eV: float
) -> np.ndarray:
    """Return a unit-area Lorentzian described by its FWHM."""

    fwhm_eV = max(float(fwhm_eV), np.finfo(float).eps)
    half_width = 0.5 * fwhm_eV
    return (half_width / np.pi) / (
        (energy_eV - center_eV) ** 2 + half_width**2
    )


def _replica_count(huang_rhys_factor: float) -> int:
    """Choose enough terms to cover the Poisson tail without an early break."""

    estimate = int(np.ceil(huang_rhys_factor + 8.0 * np.sqrt(huang_rhys_factor + 1.0)))
    return int(np.clip(max(20, estimate), 20, 80))


def _poisson_weights(huang_rhys_factor: float, count: int) -> np.ndarray:
    n = np.arange(count, dtype=float)
    if huang_rhys_factor == 0.0:
        weights = np.zeros(count, dtype=float)
        weights[0] = 1.0
        return weights
    return np.exp(-huang_rhys_factor + n * np.log(huang_rhys_factor) - gammaln(n + 1.0))


def huang_rhys_spectrum(
    energy_eV: np.ndarray,
    huang_rhys_factor: float,
    zpl_eV: float,
    fwhm_eV: float,
    phonon_energy_eV: float,
    *,
    line_shape: str = "Gaussian",
    direction: str = "Emission",
    amplitude: float = 1.0,
    replica_count: int | None = None,
    successive_broadening_multiplier: float = 1.0,
) -> SpectrumResult:
    """Calculate a broadened Huang–Rhys vibronic progression.

    For emission, the nth replica is centered at ``E_ZPL - n*hbar_omega``.
    For absorption, it is centered at ``E_ZPL + n*hbar_omega``.
    """

    energy_eV = np.asarray(energy_eV, dtype=float)
    if energy_eV.ndim != 1 or energy_eV.size < 2:
        raise ValueError("energy_eV must be a one-dimensional array with at least 2 points")
    if not np.all(np.isfinite(energy_eV)):
        raise ValueError("energy_eV contains non-finite values")
    if huang_rhys_factor < 0:
        raise ValueError("huang_rhys_factor must be non-negative")
    if fwhm_eV <= 0:
        raise ValueError("fwhm_eV must be positive")
    if phonon_energy_eV <= 0:
        raise ValueError("phonon_energy_eV must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    if successive_broadening_multiplier < 1.0:
        raise ValueError("successive_broadening_multiplier must be at least 1")

    normalized_shape = line_shape.strip().lower()
    if normalized_shape == "gaussian":
        profile = gaussian_profile
    elif normalized_shape == "lorentzian":
        profile = lorentzian_profile
    else:
        raise ValueError("line_shape must be 'Gaussian' or 'Lorentzian'")

    normalized_direction = direction.strip().lower()
    if normalized_direction == "emission":
        sign = -1.0
    elif normalized_direction == "absorption":
        sign = 1.0
    else:
        raise ValueError("direction must be 'Emission' or 'Absorption'")

    count = _replica_count(float(huang_rhys_factor)) if replica_count is None else int(replica_count)
    if count < 1:
        raise ValueError("replica_count must be at least 1")

    weights = _poisson_weights(float(huang_rhys_factor), count)
    orders = np.arange(count, dtype=float)
    centers = float(zpl_eV) + sign * orders * float(phonon_energy_eV)
    replica_fwhm = float(fwhm_eV) * (
        1.0 + orders * (float(successive_broadening_multiplier) - 1.0)
    )

    replicas = np.empty((count, energy_eV.size), dtype=float)
    for index, (center, weight, width) in enumerate(
        zip(centers, weights, replica_fwhm, strict=True)
    ):
        if weight < 1e-14:
            replicas[index] = 0.0
        else:
            replicas[index] = amplitude * weight * profile(
                energy_eV, float(center), float(width)
            )

    return SpectrumResult(
        energy_eV=energy_eV,
        total=np.sum(replicas, axis=0),
        replicas=replicas,
        centers_eV=centers,
        weights=weights,
        replica_fwhm_eV=replica_fwhm,
    )


def multi_zpl_spectrum(
    energy_eV: np.ndarray,
    huang_rhys_factor: BranchParameter,
    zpl_energies_eV: np.ndarray | list[float] | tuple[float, ...],
    relative_amplitudes: np.ndarray | list[float] | tuple[float, ...],
    fwhm_eV: BranchParameter,
    phonon_energy_eV: BranchParameter,
    *,
    line_shape: str = "Gaussian",
    direction: str = "Emission",
    successive_broadening_multiplier: BranchParameter = 1.0,
) -> MultiSpectrumResult:
    """Sum Huang–Rhys progressions from multiple electronic ZPL branches.

    A scalar ``S``, phonon energy, FWHM, or linewidth multiplier is broadcast
    to every branch for backwards compatibility. Passing one value per branch
    makes each ZPL progression independent. Line shape and direction are shared.
    The first branch is conventionally the primary ZPL.
    """

    zpls = np.asarray(zpl_energies_eV, dtype=float)
    amplitudes = np.asarray(relative_amplitudes, dtype=float)
    if zpls.ndim != 1 or amplitudes.ndim != 1 or zpls.size != amplitudes.size:
        raise ValueError("ZPL energies and relative amplitudes must be equal-length 1D arrays")
    if zpls.size < 1:
        raise ValueError("at least one ZPL branch is required")
    if not np.all(np.isfinite(zpls)) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("ZPL energies and amplitudes must be finite")
    if np.any(amplitudes < 0) or not np.any(amplitudes > 0):
        raise ValueError("relative amplitudes must be non-negative with at least one positive value")

    factors = _branch_values(huang_rhys_factor, zpls.size, "huang_rhys_factor")
    widths = _branch_values(fwhm_eV, zpls.size, "fwhm_eV")
    phonon_energies = _branch_values(
        phonon_energy_eV, zpls.size, "phonon_energy_eV"
    )
    broadening_multipliers = _branch_values(
        successive_broadening_multiplier,
        zpls.size,
        "successive_broadening_multiplier",
    )
    branch_results = [
        huang_rhys_spectrum(
            energy_eV,
            float(factor),
            float(zpl),
            float(width),
            float(phonon_energy),
            line_shape=line_shape,
            direction=direction,
            amplitude=float(amplitude),
            successive_broadening_multiplier=float(multiplier),
        )
        for zpl, amplitude, factor, width, phonon_energy, multiplier in zip(
            zpls,
            amplitudes,
            factors,
            widths,
            phonon_energies,
            broadening_multipliers,
            strict=True,
        )
    ]

    branch_totals = np.asarray([branch.total for branch in branch_results])
    replicas = np.concatenate([branch.replicas for branch in branch_results], axis=0)
    centers = np.concatenate([branch.centers_eV for branch in branch_results])
    replica_widths = np.concatenate(
        [branch.replica_fwhm_eV for branch in branch_results]
    )
    weighted_replica_areas = np.concatenate(
        [branch.weights * amplitude for branch, amplitude in zip(branch_results, amplitudes, strict=True)]
    )
    branch_indices = np.concatenate(
        [
            np.full(branch.replicas.shape[0], index, dtype=int)
            for index, branch in enumerate(branch_results)
        ]
    )
    orders = np.concatenate(
        [np.arange(branch.replicas.shape[0], dtype=int) for branch in branch_results]
    )

    return MultiSpectrumResult(
        energy_eV=np.asarray(energy_eV, dtype=float),
        total=np.sum(branch_totals, axis=0),
        replicas=replicas,
        centers_eV=centers,
        weights=weighted_replica_areas,
        replica_fwhm_eV=replica_widths,
        branch_totals=branch_totals,
        zpl_energies_eV=zpls,
        relative_amplitudes=amplitudes,
        huang_rhys_factors=factors,
        phonon_energies_eV=phonon_energies,
        base_fwhm_eV=widths,
        broadening_multipliers=broadening_multipliers,
        replica_branch_indices=branch_indices,
        replica_orders=orders,
    )
