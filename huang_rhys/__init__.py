"""Core tools for the Huang–Rhys Spectrum Explorer."""

from .data_io import (
    SpectrumData,
    SpectrumImportError,
    load_spectrum,
    load_spectrum_bytes,
)
from .model import (
    MultiSpectrumResult,
    SpectrumResult,
    huang_rhys_spectrum,
    multi_zpl_spectrum,
)
from .photon_recycling import (
    PhotonRecyclingResult,
    photon_recycle,
    prepare_absorption_profile,
)

__all__ = [
    "SpectrumData",
    "SpectrumImportError",
    "SpectrumResult",
    "MultiSpectrumResult",
    "PhotonRecyclingResult",
    "huang_rhys_spectrum",
    "multi_zpl_spectrum",
    "load_spectrum",
    "load_spectrum_bytes",
    "photon_recycle",
    "prepare_absorption_profile",
]

__version__ = "2.0.0"
