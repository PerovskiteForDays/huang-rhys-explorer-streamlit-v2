"""Spectral self-absorption and photon-recycling models.

The Beer–Lambert mode treats the supplied normalized absorption spectrum as
the energy dependence of the optical depth.  Reabsorbed photons are re-emitted
with the intrinsic Huang–Rhys spectrum according to the PLQY.  The simple mode
maps one recycling-strength control directly onto peak reabsorption probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# User-supplied, digitized CsPbBr3 absorption reference.  Keeping the values in
# code makes both wheel and PyInstaller builds independent of external data files.
CSPBBR3_ABSORPTION_ENERGY_EV = np.array(
    [
        2.1017371403501954, 2.131082380805793, 2.1475207098044393,
        2.162679127270557, 2.1778367081435093, 2.192993010888015,
        2.208151056534948, 2.2233150728676234, 2.2376230572615583,
        2.2688447751977043, 2.282671956043792, 2.2930665486565114,
        2.3000140288489703, 2.3055853907473445, 2.3104686934986765,
        2.315355404592532, 2.319551074239637, 2.3275102641046512,
        2.3338949794150827, 2.330033367427799, 2.3345236373586458,
        2.3358673433433697, 2.3453086003623826, 2.348692164236282,
        2.352616683193886, 2.3575151104651684, 2.363780112671468,
        2.374855072421852, 2.390001591674067, 2.405108605137935,
        2.4202153862148132, 2.435333089480234, 2.4504638064171105,
        2.465607304638452, 2.480758006856492, 2.49591498352327,
        2.511077769864804, 2.526242415302261, 2.541410081770591,
        2.556580072108824, 2.5717521539299697, 2.584558834973378,
        2.5952016279577594, 2.33067826970194, 2.3675750983123898,
        2.3801741827889833, 2.2552992608555407,
    ],
    dtype=float,
)

CSPBBR3_ABSORPTION_INTENSITY = np.array(
    [
        0.008502978779393544, 0.013254074095396806, 0.006934067421607404,
        0.00858289184878136, 0.009666318728613899, 0.00988594380000718,
        0.01128348042836258, 0.01671620550471009, 0.03172396297269375,
        0.05846175764625339, 0.09159058877224635, 0.13279709485232316,
        0.17226534624513512, 0.21288107295437708, 0.25407053732183393,
        0.2975634731784609, 0.33961460914027874, 0.39031536101868447,
        0.5889380963237358, 0.4400920874570652, 0.5379863630165805,
        0.6581911207882211, 0.7069235171364471, 0.7588309548150197,
        0.803864234269682, 0.8552753533703309, 0.8990892769848737,
        0.9345235290686771, 0.9281311439336575, 0.8950394301741671,
        0.8617906615404153, 0.8359234719969583, 0.8188513554121503,
        0.8104172569117299, 0.8068518595134191, 0.8075269437201711,
        0.812128399783463, 0.8179862948408476, 0.8258859032636329,
        0.8353560604290341, 0.8462397114627895, 0.8548051086127159,
        0.8642163944072472, 0.48525028195467235, 0.9213569511835235,
        0.9362382099481388, 0.04160346992979158,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PhotonRecyclingResult:
    """Escaped spectrum and photon-balance diagnostics per initial photon."""

    energy_eV: np.ndarray
    intrinsic_density: np.ndarray
    absorption_profile: np.ndarray
    reabsorption_probability: np.ndarray
    escape_probability: np.ndarray
    escaped_spectrum: np.ndarray
    escaped_generations: np.ndarray
    escape_fraction: float
    first_pass_escape_fraction: float
    nonradiative_loss_fraction: float
    residual_fraction: float
    mean_reemissions: float
    peak_shift_meV: float

    @property
    def recycling_enhancement(self) -> float:
        if self.first_pass_escape_fraction <= 0:
            return 1.0
        return self.escape_fraction / self.first_pass_escape_fraction


def _average_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return unique_x, np.bincount(inverse, weights=y) / counts


def prepare_absorption_profile(
    target_energy_eV: np.ndarray,
    source_energy_eV: np.ndarray,
    source_absorption: np.ndarray,
    *,
    shift_meV: float = 0.0,
    broadening_fwhm_meV: float = 0.0,
) -> np.ndarray:
    """Baseline-correct, shift, broaden, interpolate, and normalize absorption."""

    target = np.asarray(target_energy_eV, dtype=float)
    source_x = np.asarray(source_energy_eV, dtype=float)
    source_y = np.asarray(source_absorption, dtype=float)
    if target.ndim != 1 or target.size < 3 or np.any(np.diff(target) <= 0):
        raise ValueError("target_energy_eV must be a strictly increasing 1D array")
    if source_x.shape != source_y.shape or source_x.ndim != 1:
        raise ValueError("source energy and absorption arrays must have the same 1D shape")

    finite = np.isfinite(source_x) & np.isfinite(source_y)
    source_x, source_y = source_x[finite], source_y[finite]
    if source_x.size < 3:
        raise ValueError("at least three finite absorption points are required")
    source_x, source_y = _average_duplicate_x(source_x, source_y)

    source_y = np.clip(source_y - np.min(source_y), 0.0, None)
    peak = float(np.max(source_y))
    if peak <= np.finfo(float).eps:
        raise ValueError("absorption spectrum has no positive range after baseline correction")
    source_y /= peak

    shifted_x = source_x + float(shift_meV) / 1000.0
    profile = np.interp(
        target,
        shifted_x,
        source_y,
        left=0.0,
        right=float(source_y[-1]),
    )

    broadening_eV = max(float(broadening_fwhm_meV), 0.0) / 1000.0
    if broadening_eV > 0:
        spacing = float(np.median(np.diff(target)))
        sigma_points = broadening_eV / (2.0 * np.sqrt(2.0 * np.log(2.0))) / spacing
        if sigma_points >= 0.25:
            half_width = max(2, int(np.ceil(4.0 * sigma_points)))
            offsets = np.arange(-half_width, half_width + 1, dtype=float)
            kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
            kernel /= np.sum(kernel)
            padded = np.pad(profile, half_width, mode="edge")
            profile = np.convolve(padded, kernel, mode="valid")

    profile = np.clip(profile, 0.0, None)
    profile_peak = float(np.max(profile))
    return profile / profile_peak if profile_peak > 0 else profile


def photon_recycle(
    energy_eV: np.ndarray,
    intrinsic_emission: np.ndarray,
    absorption_profile: np.ndarray,
    *,
    mode: str = "Beer–Lambert",
    peak_optical_depth: float = 1.0,
    plqy: float = 0.9,
    max_reemissions: int = 20,
    simple_strength: float = 0.5,
) -> PhotonRecyclingResult:
    """Propagate emission through repeated absorption and re-emission events.

    ``peak_optical_depth`` is alpha*L at the absorption-profile maximum.  In
    simple mode, ``simple_strength`` is the maximum reabsorption probability;
    PLQY is treated as unity and 25 possible re-emissions are used internally.
    """

    energy = np.asarray(energy_eV, dtype=float)
    emission = np.asarray(intrinsic_emission, dtype=float)
    absorption = np.asarray(absorption_profile, dtype=float)
    if energy.ndim != 1 or energy.size < 3 or np.any(np.diff(energy) <= 0):
        raise ValueError("energy_eV must be a strictly increasing 1D array")
    if emission.shape != energy.shape or absorption.shape != energy.shape:
        raise ValueError("emission and absorption must match the energy grid")
    if not 0.0 <= plqy <= 1.0:
        raise ValueError("plqy must lie between 0 and 1")
    if peak_optical_depth < 0:
        raise ValueError("peak_optical_depth must be non-negative")
    if not 0.0 <= simple_strength < 1.0:
        raise ValueError("simple_strength must lie in [0, 1)")

    emission = np.clip(emission, 0.0, None)
    area = float(np.trapezoid(emission, energy))
    if area <= np.finfo(float).eps:
        raise ValueError("intrinsic emission must have positive integrated intensity")
    intrinsic = emission / area
    absorption = np.clip(absorption, 0.0, 1.0)

    normalized_mode = mode.strip().lower()
    if normalized_mode in {"beer–lambert", "beer-lambert"}:
        reabsorption = 1.0 - np.exp(-float(peak_optical_depth) * absorption)
        effective_plqy = float(plqy)
        cycle_limit = max(0, int(max_reemissions))
    elif normalized_mode == "simple":
        reabsorption = float(simple_strength) * absorption
        effective_plqy = 1.0
        cycle_limit = 25
    else:
        raise ValueError("mode must be 'Beer–Lambert' or 'Simple'")

    escape_probability = 1.0 - reabsorption
    current = intrinsic.copy()  # one initially emitted photon
    escaped_total = np.zeros_like(intrinsic)
    escaped_generations: list[np.ndarray] = []
    escaped_amounts: list[float] = []
    nonradiative_loss = 0.0
    residual = 0.0

    for generation in range(cycle_limit + 1):
        escaped = current * escape_probability
        escaped_amount = float(np.trapezoid(escaped, energy))
        escaped_total += escaped
        escaped_generations.append(escaped)
        escaped_amounts.append(escaped_amount)

        absorbed_amount = max(0.0, float(np.trapezoid(current * reabsorption, energy)))
        nonradiative_loss += absorbed_amount * (1.0 - effective_plqy)
        next_photons = absorbed_amount * effective_plqy
        if generation == cycle_limit:
            residual = next_photons
            break
        if next_photons < 1e-12:
            break
        current = intrinsic * next_photons

    escape_fraction = float(sum(escaped_amounts))
    first_pass = float(escaped_amounts[0])
    if escape_fraction > 0:
        mean_reemissions = float(
            sum(i * amount for i, amount in enumerate(escaped_amounts)) / escape_fraction
        )
    else:
        mean_reemissions = 0.0

    intrinsic_peak_eV = float(energy[int(np.argmax(intrinsic))])
    escaped_peak_eV = float(energy[int(np.argmax(escaped_total))])

    return PhotonRecyclingResult(
        energy_eV=energy,
        intrinsic_density=intrinsic,
        absorption_profile=absorption,
        reabsorption_probability=reabsorption,
        escape_probability=escape_probability,
        escaped_spectrum=escaped_total,
        escaped_generations=np.asarray(escaped_generations),
        escape_fraction=escape_fraction,
        first_pass_escape_fraction=first_pass,
        nonradiative_loss_fraction=float(nonradiative_loss),
        residual_fraction=float(residual),
        mean_reemissions=mean_reemissions,
        peak_shift_meV=(escaped_peak_eV - intrinsic_peak_eV) * 1000.0,
    )
