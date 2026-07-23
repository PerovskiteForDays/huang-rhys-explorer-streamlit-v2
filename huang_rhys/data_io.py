"""Defensive import helpers for two-column spectral data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


HC_EV_NM = 1239.8419843320026
_NUMBER = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][+-]?\d+)?$"
)


class SpectrumImportError(ValueError):
    """Raised when a file cannot be interpreted as a spectrum."""


@dataclass(frozen=True)
class SpectrumData:
    """Sanitized spectrum in increasing energy order."""

    energy_eV: np.ndarray
    intensity: np.ndarray
    source_name: str
    interpreted_unit: str
    parsed_rows: int
    skipped_rows: int
    duplicate_points_merged: int

    @property
    def summary(self) -> str:
        details = [f"{self.energy_eV.size:,} points", f"input: {self.interpreted_unit}"]
        if self.duplicate_points_merged:
            details.append(f"{self.duplicate_points_merged} duplicates averaged")
        if self.skipped_rows:
            details.append(f"{self.skipped_rows} non-data rows skipped")
        return " • ".join(details)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SpectrumImportError("The file uses an unsupported text encoding.")


def _numeric_tokens(line: str) -> list[float]:
    # Remove common inline comments, then support CSV, TSV, semicolon, and
    # arbitrary whitespace-delimited files without requiring pandas.
    line = re.split(r"(?://|#|%)", line, maxsplit=1)[0].strip()
    if not line:
        return []

    tokens = re.split(r"[,;\t\s]+", line)
    values: list[float] = []
    for token in tokens:
        cleaned = token.strip().strip('"\'').replace("−", "-")
        if _NUMBER.fullmatch(cleaned):
            values.append(float(cleaned.replace("D", "E").replace("d", "e")))
    return values


def _merge_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if unique_x.size == x.size:
        return x, y, 0
    sums = np.bincount(inverse, weights=y)
    averaged_y = sums / counts
    return unique_x, averaged_y, int(x.size - unique_x.size)


def _interpret_x(x: np.ndarray, requested_unit: str) -> tuple[np.ndarray, str]:
    unit = requested_unit.strip().lower()
    if unit not in {"auto", "energy (ev)", "wavelength (nm)"}:
        raise SpectrumImportError(f"Unsupported x-axis unit: {requested_unit}")

    if unit == "auto":
        median = float(np.median(x))
        # Optical spectra near hundreds of units are overwhelmingly likely to
        # be wavelength in nm; energy data are normally below 20 eV.
        unit = "wavelength (nm)" if 100.0 <= median <= 100_000.0 else "energy (ev)"

    if unit == "wavelength (nm)":
        if np.any(x <= 0):
            raise SpectrumImportError("Wavelength values must be greater than zero.")
        return HC_EV_NM / x, "wavelength (nm), converted to eV"

    if np.any(x <= 0):
        raise SpectrumImportError("Energy values must be greater than zero.")
    return x, "energy (eV)"


def _parse_spectrum_text(
    text: str, source_name: str, x_unit: str
) -> SpectrumData:
    """Sanitize decoded two-column spectral text."""

    rows: list[tuple[float, float]] = []
    skipped = 0
    nonempty = 0
    for line in text.splitlines():
        if line.strip():
            nonempty += 1
        values = _numeric_tokens(line)
        if len(values) >= 2:
            rows.append((values[0], values[1]))
        elif line.strip():
            skipped += 1

    if len(rows) < 3:
        raise SpectrumImportError(
            "No usable two-column spectrum was found. Use CSV, TSV, TXT, or DAT "
            "with x in the first numeric column and intensity in the second."
        )

    data = np.asarray(rows, dtype=float)
    finite = np.all(np.isfinite(data), axis=1)
    skipped += int(np.count_nonzero(~finite))
    data = data[finite]
    if data.shape[0] < 3:
        raise SpectrumImportError("Fewer than three finite data points remain after cleanup.")

    x, y = data[:, 0], data[:, 1]
    x, interpreted_unit = _interpret_x(x, x_unit)
    x, y, duplicates = _merge_duplicate_x(x, y)

    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 3 or np.ptp(x) <= np.finfo(float).eps:
        raise SpectrumImportError("The x-axis must contain at least three distinct values.")

    peak = float(np.max(np.abs(y)))
    if not np.isfinite(peak) or peak <= np.finfo(float).eps:
        raise SpectrumImportError("Intensity values are all zero or invalid.")
    y = y / peak

    return SpectrumData(
        energy_eV=x,
        intensity=y,
        source_name=source_name,
        interpreted_unit=interpreted_unit,
        parsed_rows=len(rows),
        skipped_rows=max(skipped, nonempty - len(rows)),
        duplicate_points_merged=duplicates,
    )


def load_spectrum_bytes(
    raw: bytes, source_name: str = "uploaded_spectrum", x_unit: str = "Auto"
) -> SpectrumData:
    """Load a spectrum directly from uploaded bytes without a temporary file."""

    if not isinstance(raw, bytes) or not raw:
        raise SpectrumImportError("The uploaded spectrum is empty or invalid.")
    return _parse_spectrum_text(_decode_text(raw), str(source_name), x_unit)


def load_spectrum(path: str | Path, x_unit: str = "Auto") -> SpectrumData:
    """Load, sanitize, normalize, and sort a two-column spectrum.

    Header/comment lines and malformed rows are skipped. If a row contains more
    than two numeric columns, the first two numeric values are used. Duplicate
    x-values are averaged instead of being silently discarded.
    """

    path = Path(path)
    if not path.is_file():
        raise SpectrumImportError(f"File not found: {path}")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SpectrumImportError(f"Could not read the file: {exc}") from exc
    return load_spectrum_bytes(raw, source_name=path.name, x_unit=x_unit)
