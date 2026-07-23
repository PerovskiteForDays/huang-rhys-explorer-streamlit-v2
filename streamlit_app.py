"""Streamlit deployment entry point for the Huang–Rhys Spectrum Explorer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from huang_rhys.data_io import (
    SpectrumData,
    SpectrumImportError,
    load_spectrum_bytes,
)
from huang_rhys.model import MultiSpectrumResult, multi_zpl_spectrum
from huang_rhys.photon_recycling import (
    CSPBBR3_ABSORPTION_ENERGY_EV,
    CSPBBR3_ABSORPTION_INTENSITY,
    PhotonRecyclingResult,
    photon_recycle,
    prepare_absorption_profile,
)


COLORS = {
    "model": "#172033",
    "accent": "#2563EB",
    "data": "#E45756",
    "absorption": "#0F9D8A",
    "intrinsic": "#8A96A8",
    "grid": "#E8EDF3",
}
BRANCH_COLORS = (
    "#2563EB",
    "#7C3AED",
    "#0F9D8A",
    "#E45756",
    "#F59E0B",
    "#0891B2",
    "#DB2777",
    "#65A30D",
    "#475569",
)
DEFAULT_BRANCH = {
    "Active": True,
    "Label": "ZPL 1",
    "ZPL (eV)": 2.300,
    "Amplitude": 1.0,
    "S": 2.0,
    "Phonon (meV)": 30.0,
    "FWHM₀ (meV)": 15.0,
    "Width m": 1.0,
}


st.set_page_config(
    page_title="Huang–Rhys Spectrum Explorer",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid #dce3ec;}
    [data-testid="stMetric"] {
        background: white; border: 1px solid #dce3ec; border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }
    .app-subtitle {color: #64748b; margin-top: -0.7rem; margin-bottom: 1rem;}
    .small-note {color: #64748b; font-size: 0.86rem;}
    div[data-testid="stExpander"] {border-color: #dce3ec; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _reset_app() -> None:
    """Clear widget state so Streamlit restores all clean defaults."""

    for key in list(st.session_state):
        del st.session_state[key]


def _uploaded_spectrum(
    uploaded_file, x_unit: str
) -> tuple[SpectrumData | None, str | None]:
    if uploaded_file is None:
        return None, None
    try:
        return (
            load_spectrum_bytes(
                uploaded_file.getvalue(),
                source_name=uploaded_file.name,
                x_unit=x_unit,
            ),
            None,
        )
    except SpectrumImportError as exc:
        return None, str(exc)


def _clean_branches(edited: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate the dynamic ZPL table and provide usable row defaults."""

    warnings: list[str] = []
    table = edited.copy()
    if table.empty:
        table = pd.DataFrame([DEFAULT_BRANCH])
        warnings.append("At least one ZPL is required; the default branch is being used.")
    if len(table) > 9:
        table = table.iloc[:9].copy()
        warnings.append("Only the first nine ZPL branches are calculated.")

    numeric_defaults = {
        "ZPL (eV)": 2.300,
        "Amplitude": 0.5,
        "S": 2.0,
        "Phonon (meV)": 30.0,
        "FWHM₀ (meV)": 15.0,
        "Width m": 1.0,
    }
    bounds = {
        "ZPL (eV)": (1.5, 3.5),
        "Amplitude": (0.001, 10.0),
        "S": (0.0, 10.0),
        "Phonon (meV)": (0.1, 100.0),
        "FWHM₀ (meV)": (0.1, 100.0),
        "Width m": (1.0, 3.0),
    }
    for column, default in numeric_defaults.items():
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(default)
        low, high = bounds[column]
        table[column] = table[column].clip(low, high)

    table["Active"] = table["Active"].fillna(True).astype(bool)
    table["Label"] = table["Label"].fillna("").astype(str).str.strip()
    for index in range(len(table)):
        if not table.iloc[index]["Label"]:
            table.iat[index, table.columns.get_loc("Label")] = f"ZPL {index + 1}"
    if not bool(table["Active"].any()):
        table.iat[0, table.columns.get_loc("Active")] = True
        warnings.append("At least one ZPL must remain active; the first branch is enabled.")
    return table.reset_index(drop=True), warnings


def _energy_bounds(
    branches: pd.DataFrame,
    direction: str,
    emission_data: SpectrumData | None,
    absorption_energy: np.ndarray | None,
    absorption_shift_meV: float,
    absorption_broadening_meV: float,
    include_absorption: bool,
) -> tuple[float, float]:
    low, high = np.inf, -np.inf
    if emission_data is not None:
        low = min(low, float(np.min(emission_data.energy_eV)))
        high = max(high, float(np.max(emission_data.energy_eV)))

    for _, row in branches[branches["Active"]].iterrows():
        zpl = float(row["ZPL (eV)"])
        factor = float(row["S"])
        phonon = float(row["Phonon (meV)"]) / 1000.0
        fwhm = float(row["FWHM₀ (meV)"]) / 1000.0
        visible_order = max(4, int(np.ceil(factor + 4.0 * np.sqrt(factor + 1.0))))
        progression = visible_order * phonon
        pad = max(0.06, 5.0 * fwhm)
        if direction == "Emission":
            branch_low, branch_high = zpl - progression - pad, zpl + pad
        else:
            branch_low, branch_high = zpl - pad, zpl + progression + pad
        low, high = min(low, branch_low), max(high, branch_high)

    if include_absorption and absorption_energy is not None:
        shift = absorption_shift_meV / 1000.0
        broadening = absorption_broadening_meV / 1000.0
        low = min(low, float(np.min(absorption_energy)) + shift - 2 * broadening)
        high = max(high, float(np.max(absorption_energy)) + shift + 2 * broadening)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 1.8, 2.7
    span = high - low
    pad = max(0.015, 0.025 * span)
    return max(0.001, low - pad), high + pad


def _legend_position(name: str) -> dict[str, float | str]:
    positions = {
        "Top right": {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"},
        "Top left": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
        "Bottom right": {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"},
        "Bottom left": {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
    }
    return positions[name]


def _line_trace(
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    color: str,
    width: float,
    dash: str = "solid",
    opacity: float = 1.0,
    yaxis: str | None = None,
    showlegend: bool = True,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": width, "dash": dash},
        opacity=opacity,
        yaxis=yaxis,
        showlegend=showlegend,
        hovertemplate="E = %{x:.5f} eV<br>I = %{y:.5g}<extra>%{fullData.name}</extra>",
    )


with st.sidebar:
    st.markdown("### Controls")
    st.button("Reset app", on_click=_reset_app, width="stretch")

    st.markdown("#### Spectrum")
    line_shape = st.radio(
        "Line shape", ("Gaussian", "Lorentzian"), horizontal=True
    )
    direction = st.radio("Process", ("Emission", "Absorption"), horizontal=True)
    log_y = st.toggle("LOG Y-SCALE", value=False)
    model_scale = st.number_input(
        "Model scale", min_value=0.01, max_value=5.0, value=1.0, step=0.05
    )
    data_scale = st.number_input(
        "Data scale", min_value=0.01, max_value=5.0, value=1.0, step=0.05
    )

    with st.expander("Experimental emission", expanded=False):
        emission_unit = st.selectbox(
            "Imported x-axis",
            ("Auto", "Energy (eV)", "Wavelength (nm)"),
            key="emission_unit",
        )
        emission_upload = st.file_uploader(
            "Upload CSV, TSV, TXT, or DAT",
            type=("csv", "tsv", "txt", "dat"),
            key="emission_upload",
        )

    st.markdown("#### Photon recycling")
    recycling_enabled = st.toggle("Enable recycling", value=True)
    recycling_mode = st.radio(
        "Recycling model", ("Beer–Lambert", "Simple"), horizontal=True
    )
    absorption_upload = st.file_uploader(
        "Optional absorption spectrum",
        type=("csv", "tsv", "txt", "dat"),
        key="absorption_upload",
        help="Leave empty to use the built-in digitized CsPbBr₃ absorption reference.",
    )
    absorption_unit = st.selectbox(
        "Absorption x-axis",
        ("Auto", "Energy (eV)", "Wavelength (nm)"),
        key="absorption_unit",
    )
    absorption_shift_meV = st.slider(
        "Absorption shift ΔE(T) (meV)", -100.0, 100.0, 0.0, 1.0
    )
    absorption_broadening_meV = st.slider(
        "Extra absorption FWHM (meV)", 0.0, 100.0, 0.0, 1.0
    )
    peak_optical_depth = st.slider(
        "Peak optical depth αL",
        0.0,
        8.0,
        1.0,
        0.05,
        disabled=recycling_mode != "Beer–Lambert",
    )
    plqy = st.slider(
        "Internal PLQY",
        0.0,
        1.0,
        0.90,
        0.01,
        disabled=recycling_mode != "Beer–Lambert",
    )
    max_reemissions = st.slider(
        "Maximum re-emissions",
        0,
        30,
        12,
        1,
        disabled=recycling_mode != "Beer–Lambert",
    )
    simple_strength = st.slider(
        "Recycling strength",
        0.0,
        0.98,
        0.50,
        0.01,
        disabled=recycling_mode != "Simple",
    )

    st.markdown("#### Visible plot layers")
    show_final = st.checkbox("Final/model spectrum", value=True)
    show_intrinsic = st.checkbox("Intrinsic before recycling", value=False)
    show_data = st.checkbox("Experimental data", value=True)
    show_components = st.checkbox("ZPL component curves", value=False)
    show_replicas = st.checkbox("Individual phonon replicas", value=False)
    show_markers = st.checkbox("ZPL position markers", value=True)
    show_absorption = st.checkbox("Absorption profile", value=False)

    with st.expander("Plot text and axes", expanded=False):
        custom_title = st.text_input("Title (blank = automatic)", "")
        x_label = st.text_input("X-axis label", "Photon energy (eV)")
        y_label = st.text_input("Y-axis label", "Normalized intensity (a.u.)")
        absorption_y_label = st.text_input(
            "Right-axis label", "Normalized absorption"
        )
        final_legend = st.text_input("Final/model legend", "Final spectrum")
        intrinsic_legend = st.text_input("Intrinsic legend", "Intrinsic emission")
        data_legend = st.text_input("Data legend (blank = filename)", "")
        absorption_legend = st.text_input("Absorption legend", "Absorption overlap")
        show_legend = st.checkbox("Show legend", value=True)
        legend_location = st.selectbox(
            "Legend position",
            ("Top right", "Top left", "Bottom right", "Bottom left"),
        )
        manual_limits = st.checkbox("Use manual axis limits", value=False)
        axis_col1, axis_col2 = st.columns(2)
        with axis_col1:
            x_min = st.number_input("X min (eV)", value=1.90, step=0.01)
            y_min = st.number_input(
                "Y min", value=0.0001 if log_y else 0.0, format="%.6g"
            )
        with axis_col2:
            x_max = st.number_input("X max (eV)", value=2.70, step=0.01)
            y_max = st.number_input("Y max", value=1.20, format="%.6g")


emission_data, emission_error = _uploaded_spectrum(emission_upload, emission_unit)
absorption_data, absorption_error = _uploaded_spectrum(
    absorption_upload, absorption_unit
)
if emission_error:
    st.error(f"Emission import error: {emission_error}")
if absorption_error:
    st.error(f"Absorption import error: {absorption_error}")

st.title("Huang–Rhys Spectrum Explorer")
st.markdown(
    '<p class="app-subtitle">Interactive multi-ZPL phonon progressions, experimental data, self-absorption, and photon recycling</p>',
    unsafe_allow_html=True,
)

with st.expander("ZPL branch mixer — add, mute, or edit progressions", expanded=True):
    st.markdown(
        '<p class="small-note">Use the + control at the bottom of the table to add a ZPL. '
        "Each active row has its own S, phonon energy, base FWHM, and successive-width multiplier.</p>",
        unsafe_allow_html=True,
    )
    branch_editor = st.data_editor(
        pd.DataFrame([DEFAULT_BRANCH]),
        key="zpl_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        height=215,
        column_config={
            "Active": st.column_config.CheckboxColumn("Active", default=True),
            "Label": st.column_config.TextColumn("Label", width="medium"),
            "ZPL (eV)": st.column_config.NumberColumn(
                "ZPL (eV)", min_value=1.5, max_value=3.5, step=0.001, format="%.4f"
            ),
            "Amplitude": st.column_config.NumberColumn(
                "Amplitude", min_value=0.001, max_value=10.0, step=0.05, format="%.3f"
            ),
            "S": st.column_config.NumberColumn(
                "S", min_value=0.0, max_value=10.0, step=0.05, format="%.3f"
            ),
            "Phonon (meV)": st.column_config.NumberColumn(
                "Phonon (meV)", min_value=0.1, max_value=100.0, step=0.5, format="%.2f"
            ),
            "FWHM₀ (meV)": st.column_config.NumberColumn(
                "FWHM₀ (meV)", min_value=0.1, max_value=100.0, step=0.5, format="%.2f"
            ),
            "Width m": st.column_config.NumberColumn(
                "Width m", min_value=1.0, max_value=3.0, step=0.05, format="%.3f"
            ),
        },
    )

branches, branch_warnings = _clean_branches(branch_editor)
for warning in branch_warnings:
    st.warning(warning)

if absorption_data is not None:
    absorption_source_energy = absorption_data.energy_eV
    absorption_source_intensity = absorption_data.intensity
    absorption_source_name = absorption_data.source_name
else:
    absorption_source_energy = CSPBBR3_ABSORPTION_ENERGY_EV
    absorption_source_intensity = CSPBBR3_ABSORPTION_INTENSITY
    absorption_source_name = "Built-in CsPbBr₃ reference"

low, high = _energy_bounds(
    branches,
    direction,
    emission_data,
    absorption_source_energy,
    absorption_shift_meV,
    absorption_broadening_meV,
    show_absorption,
)
energy = np.linspace(low, high, 2600)
active = branches["Active"].to_numpy(dtype=bool)
amplitudes = branches["Amplitude"].to_numpy(dtype=float) * active

result: MultiSpectrumResult = multi_zpl_spectrum(
    energy,
    branches["S"].to_numpy(dtype=float),
    branches["ZPL (eV)"].to_numpy(dtype=float),
    amplitudes,
    branches["FWHM₀ (meV)"].to_numpy(dtype=float) / 1000.0,
    branches["Phonon (meV)"].to_numpy(dtype=float) / 1000.0,
    line_shape=line_shape,
    direction=direction,
    successive_broadening_multiplier=branches["Width m"].to_numpy(dtype=float),
)
model_peak = max(float(np.max(result.total)), np.finfo(float).eps)
intrinsic_display = result.total / model_peak * model_scale
branch_displays = result.branch_totals / model_peak * model_scale
replica_displays = result.replicas / model_peak * model_scale

absorption_profile: np.ndarray | None = None
recycling_result: PhotonRecyclingResult | None = None
final_display = intrinsic_display
if recycling_enabled or show_absorption:
    try:
        absorption_profile = prepare_absorption_profile(
            energy,
            absorption_source_energy,
            absorption_source_intensity,
            shift_meV=absorption_shift_meV,
            broadening_fwhm_meV=absorption_broadening_meV,
        )
    except ValueError as exc:
        st.error(f"Absorption processing error: {exc}")

recycling_active = (
    recycling_enabled and direction == "Emission" and absorption_profile is not None
)
if recycling_active:
    recycling_result = photon_recycle(
        energy,
        result.total,
        absorption_profile,
        mode=recycling_mode,
        peak_optical_depth=peak_optical_depth,
        plqy=plqy,
        max_reemissions=max_reemissions,
        simple_strength=simple_strength,
    )
    density_peak = max(
        float(np.max(recycling_result.intrinsic_density)), np.finfo(float).eps
    )
    final_display = recycling_result.escaped_spectrum / density_peak * model_scale
elif recycling_enabled and direction != "Emission":
    st.info("Photon recycling is applied only in Emission mode.")

metric_columns = st.columns(4)
metric_columns[0].metric("Active ZPLs", f"{int(np.sum(active))}/{len(branches)}")
metric_columns[1].metric("Model peak", f"{energy[int(np.argmax(intrinsic_display))]:.4f} eV")
if recycling_result is not None:
    metric_columns[2].metric(
        "Escaped photons", f"{100 * recycling_result.escape_fraction:.1f}%"
    )
    metric_columns[3].metric(
        "Recycling peak shift", f"{recycling_result.peak_shift_meV:+.1f} meV"
    )
else:
    metric_columns[2].metric("Line shape", line_shape)
    metric_columns[3].metric("Process", direction)

figure = go.Figure()
if show_replicas:
    for branch_index, color in enumerate(BRANCH_COLORS[: len(branches)]):
        if not active[branch_index]:
            continue
        indices = np.flatnonzero(result.replica_branch_indices == branch_index)
        branch_weights = result.weights[indices]
        threshold = max(1e-7, float(np.max(branch_weights, initial=0.0)) * 1e-3)
        for replica_index in indices[branch_weights >= threshold]:
            order = int(result.replica_orders[replica_index])
            figure.add_trace(
                _line_trace(
                    energy,
                    replica_displays[replica_index],
                    name=f"{branches.iloc[branch_index]['Label']} · n={order}",
                    color=color,
                    width=1.0,
                    opacity=max(0.30, 0.65 - 0.025 * order),
                    showlegend=False,
                )
            )

if show_components and int(np.sum(active)) > 1:
    for index, (is_active, color) in enumerate(
        zip(active, BRANCH_COLORS[: len(branches)], strict=True)
    ):
        if is_active:
            figure.add_trace(
                _line_trace(
                    energy,
                    branch_displays[index],
                    name=f"{branches.iloc[index]['Label']} progression",
                    color=color,
                    width=1.8,
                    dash="dot",
                    opacity=0.85,
                )
            )

if recycling_active and show_intrinsic:
    figure.add_trace(
        _line_trace(
            energy,
            intrinsic_display,
            name=intrinsic_legend,
            color=COLORS["intrinsic"],
            width=2.0,
            dash="dash",
        )
    )
if show_final:
    figure.add_trace(
        _line_trace(
            energy,
            final_display,
            name=final_legend,
            color=COLORS["accent"] if recycling_active else COLORS["model"],
            width=3.0,
        )
    )

plotted_data: np.ndarray | None = None
if emission_data is not None and show_data:
    plotted_data = emission_data.intensity * data_scale
    figure.add_trace(
        _line_trace(
            emission_data.energy_eV,
            plotted_data,
            name=data_legend.strip() or emission_data.source_name,
            color=COLORS["data"],
            width=2.0,
            opacity=0.9,
        )
    )

if show_absorption and absorption_profile is not None:
    figure.add_trace(
        _line_trace(
            energy,
            absorption_profile,
            name=absorption_legend,
            color=COLORS["absorption"],
            width=2.0,
            opacity=0.9,
            yaxis="y2",
        )
    )

visible_maxima = [float(np.max(final_display))]
if plotted_data is not None:
    visible_maxima.append(float(np.nanmax(plotted_data)))
marker_y = max(max(visible_maxima), 1e-6) * 1.035
if show_markers:
    for index, (is_active, color) in enumerate(
        zip(active, BRANCH_COLORS[: len(branches)], strict=True)
    ):
        if not is_active:
            continue
        figure.add_trace(
            go.Scatter(
                x=[float(branches.iloc[index]["ZPL (eV)"])],
                y=[marker_y],
                mode="markers+text",
                marker={"symbol": "triangle-down", "size": 10, "color": color},
                text=[str(index + 1)],
                textposition="bottom center",
                name=str(branches.iloc[index]["Label"]),
                showlegend=False,
                hovertemplate=(
                    f"{branches.iloc[index]['Label']}<br>ZPL = "
                    "%{x:.5f} eV<extra></extra>"
                ),
            )
        )

automatic_title = (
    f"Emission · {recycling_mode} photon recycling"
    if recycling_active
    else f"{direction} progression · {line_shape} broadening"
)
legend_settings = _legend_position(legend_location)
layout: dict = {
    "title": {"text": custom_title.strip() or automatic_title, "x": 0.01, "xanchor": "left"},
    "height": 650,
    "margin": {"l": 70, "r": 75 if show_absorption else 30, "t": 70, "b": 65},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "hovermode": "x unified",
    "uirevision": "huang-rhys-view-v1",
    "showlegend": show_legend,
    "legend": {
        **legend_settings,
        "bgcolor": "rgba(255,255,255,0.88)",
        "bordercolor": "#DCE3EC",
        "borderwidth": 1,
    },
    "xaxis": {
        "title": x_label,
        "showgrid": True,
        "gridcolor": COLORS["grid"],
        "zeroline": False,
    },
    "yaxis": {
        "title": y_label,
        "type": "log" if log_y else "linear",
        "showgrid": True,
        "gridcolor": COLORS["grid"],
        "zeroline": False,
    },
}
if show_absorption:
    layout["yaxis2"] = {
        "title": absorption_y_label,
        "overlaying": "y",
        "side": "right",
        "range": [0.0, 1.08],
        "showgrid": False,
        "color": COLORS["absorption"],
    }
if manual_limits:
    if x_max > x_min:
        layout["xaxis"]["range"] = [x_min, x_max]
    else:
        st.warning("Manual X max must be greater than X min.")
    if y_max > y_min and (not log_y or y_min > 0):
        layout["yaxis"]["range"] = (
            [float(np.log10(y_min)), float(np.log10(y_max))]
            if log_y
            else [y_min, y_max]
        )
    else:
        st.warning("Manual Y limits are invalid for the selected scale.")

figure.update_layout(**layout)
st.plotly_chart(
    figure,
    width="stretch",
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {"format": "png", "scale": 3},
    },
)

if recycling_result is not None:
    st.caption(
        f"Absorption: {absorption_source_name} · "
        f"escape {100 * recycling_result.escape_fraction:.1f}% · "
        f"nonradiative loss {100 * recycling_result.nonradiative_loss_fraction:.1f}% · "
        f"residual {100 * recycling_result.residual_fraction:.2f}% · "
        f"mean re-emissions {recycling_result.mean_reemissions:.2f}"
    )
elif absorption_profile is not None:
    st.caption(f"Absorption: {absorption_source_name}")

export_table = pd.DataFrame(
    {
        "energy_eV": energy,
        "intrinsic_emission": intrinsic_display,
        "final_displayed_spectrum": final_display,
        "absorption_normalized": (
            absorption_profile
            if absorption_profile is not None
            else np.zeros_like(energy)
        ),
        "reabsorption_probability": (
            recycling_result.reabsorption_probability
            if recycling_result is not None
            else np.zeros_like(energy)
        ),
    }
)
for index, row in branches.iterrows():
    state = "active" if bool(row["Active"]) else "muted"
    safe_label = "_".join(str(row["Label"]).split())
    export_table[f"branch_{index + 1}_{safe_label}_{state}"] = branch_displays[index]

parameters = {
    "line_shape": line_shape,
    "direction": direction,
    "log_y": log_y,
    "model_scale": model_scale,
    "data_scale": data_scale,
    "photon_recycling": {
        "enabled": recycling_enabled,
        "mode": recycling_mode,
        "absorption_source": absorption_source_name,
        "absorption_shift_meV": absorption_shift_meV,
        "absorption_broadening_meV": absorption_broadening_meV,
        "peak_optical_depth": peak_optical_depth,
        "plqy": plqy,
        "max_reemissions": max_reemissions,
        "simple_strength": simple_strength,
    },
    "zpl_branches": branches.to_dict(orient="records"),
}
download_columns = st.columns(3)
download_columns[0].download_button(
    "Download model CSV",
    export_table.to_csv(index=False).encode("utf-8"),
    file_name="huang_rhys_model.csv",
    mime="text/csv",
    width="stretch",
)
download_columns[1].download_button(
    "Download parameters JSON",
    json.dumps(parameters, indent=2).encode("utf-8"),
    file_name="huang_rhys_parameters.json",
    mime="application/json",
    width="stretch",
)
download_columns[2].download_button(
    "Download interactive HTML",
    figure.to_html(include_plotlyjs=True, full_html=True).encode("utf-8"),
    file_name="huang_rhys_spectrum.html",
    mime="text/html",
    width="stretch",
)

with st.expander("Model conventions and deployment notes", expanded=False):
    st.markdown(
        r"""
        The nth replica has Poisson weight
        $w_n=e^{-S}S^n/n!$ and center
        $E_n=E_{\mathrm{ZPL}}\mp n\hbar\omega$ for emission/absorption.
        Successive linewidths follow
        $\Gamma_n=\Gamma_0[1+n(m-1)]$.

        Plotly preserves browser zoom across ordinary Streamlit reruns. Use the
        modebar Home button to restore the automatic view, or enable manual axis
        limits in the sidebar.
        """
    )
