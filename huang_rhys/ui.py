"""Tk desktop interface for the Huang–Rhys Spectrum Explorer."""

from __future__ import annotations

import csv
from pathlib import Path
import platform
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import font_manager
import numpy as np

from .data_io import SpectrumData, SpectrumImportError, load_spectrum
from .model import MultiSpectrumResult, multi_zpl_spectrum
from .photon_recycling import (
    CSPBBR3_ABSORPTION_ENERGY_EV,
    CSPBBR3_ABSORPTION_INTENSITY,
    PhotonRecyclingResult,
    photon_recycle,
    prepare_absorption_profile,
)


COLORS = {
    "app": "#F3F6FA",
    "card": "#FFFFFF",
    "plot": "#FFFFFF",
    "text": "#172033",
    "muted": "#64748B",
    "border": "#DCE3EC",
    "grid": "#E8EDF3",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "model": "#15243A",
    "data": "#E45756",
    "absorption": "#0F9D8A",
    "intrinsic": "#8A96A8",
    "success": "#15803D",
    "danger": "#B42318",
}


def _font_family() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    system = platform.system()
    if system == "Darwin":
        candidates = ("SF Pro Text", "Helvetica Neue", "Helvetica", "Arial")
    elif system == "Windows":
        candidates = ("Segoe UI", "Arial")
    else:
        candidates = ("DejaVu Sans", "Liberation Sans", "Arial")
    return next((name for name in candidates if name in installed), "DejaVu Sans")


FONT = _font_family()


class ParameterControl(ttk.Frame):
    """Compact labeled slider with a precise numeric entry."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        label: str,
        minimum: float,
        maximum: float,
        initial: float,
        decimals: int,
        command,
        suffix: str = "",
    ) -> None:
        super().__init__(parent, style="Card.TFrame")
        self.minimum = minimum
        self.maximum = maximum
        self.decimals = decimals
        self.command = command
        self.suffix = suffix
        self._updating = False

        self.value = tk.DoubleVar(value=initial)
        self.entry_value = tk.StringVar(value=self._format(initial))

        self.label_widget = ttk.Label(self, text=label, style="Parameter.TLabel")
        self.label_widget.grid(
            row=0, column=0, sticky="w"
        )
        self.entry = ttk.Entry(
            self,
            textvariable=self.entry_value,
            width=10,
            justify="right",
            style="Value.TEntry",
        )
        self.entry.grid(row=0, column=1, sticky="e")
        self.entry.bind("<Return>", self._commit_entry)
        self.entry.bind("<FocusOut>", self._commit_entry)

        self.scale = ttk.Scale(
            self,
            from_=minimum,
            to=maximum,
            variable=self.value,
            command=self._on_slider,
            style="Parameter.Horizontal.TScale",
        )
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.columnconfigure(0, weight=1)

    def _format(self, value: float) -> str:
        number = f"{value:.{self.decimals}f}"
        return f"{number}{self.suffix}"

    def _on_slider(self, raw_value: str) -> None:
        if self._updating:
            return
        value = float(raw_value)
        if self.decimals == 0:
            value = float(round(value))
            self._updating = True
            try:
                self.value.set(value)
            finally:
                self._updating = False
        self.entry_value.set(self._format(value))
        self.command()

    def _commit_entry(self, _event=None) -> None:
        raw = self.entry_value.get().strip()
        if self.suffix and raw.lower().endswith(self.suffix.lower()):
            raw = raw[: -len(self.suffix)].strip()
        try:
            value = float(raw)
        except ValueError:
            self.entry_value.set(self._format(self.get()))
            self.entry.bell()
            return
        value = float(np.clip(value, self.minimum, self.maximum))
        self.set(value)
        self.command()

    def get(self) -> float:
        return float(self.value.get())

    def set(self, value: float) -> None:
        self._updating = True
        try:
            value = float(np.clip(value, self.minimum, self.maximum))
            self.value.set(value)
            self.entry_value.set(self._format(value))
        finally:
            self._updating = False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or gray out the complete control without changing its value."""

        if enabled:
            self.label_widget.state(["!disabled"])
            self.entry.state(["!disabled"])
            self.scale.state(["!disabled"])
        else:
            self.label_widget.state(["disabled"])
            self.entry.state(["disabled"])
            self.scale.state(["disabled"])


class HuangRhysExplorer:
    """Polished, cross-platform desktop application."""

    DEFAULTS = {
        "S": 2.0,
        "zpl": 2.300,
        "fwhm_meV": 15.0,
        "phonon_meV": 30.0,
        "broadening_multiplier": 1.0,
        "model_scale": 1.0,
        "data_scale": 1.0,
    }
    RECYCLING_DEFAULTS = {
        "abs_shift_meV": 0.0,
        "abs_broadening_meV": 0.0,
        "peak_optical_depth": 1.0,
        "plqy": 0.90,
        "max_reemissions": 12.0,
        "simple_strength": 0.50,
    }
    TEXT_DEFAULTS = {
        "title": "",
        "x_label": "Photon energy (eV)",
        "y_label": "Normalized intensity (a.u.)",
        "absorption_y_label": "Normalized absorption",
        "model_legend": "Huang–Rhys model",
        "combined_legend": "Combined model",
        "intrinsic_legend": "Intrinsic emission",
        "escaped_legend": "Escaped emission",
        "data_legend": "",
        "absorption_legend": "Absorption overlap",
        "branch_prefix": "ZPL",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.data: SpectrumData | None = None
        self.absorption_data: SpectrumData | None = None
        self.extra_peaks: list[dict[str, float | bool]] = []
        self.last_result: MultiSpectrumResult | None = None
        self.last_recycling: PhotonRecyclingResult | None = None
        self.last_model_display: np.ndarray | None = None
        self.last_intrinsic_display: np.ndarray | None = None
        self.last_branch_displays: np.ndarray | None = None
        self.last_absorption_profile: np.ndarray | None = None
        self.last_update_id: str | None = None
        self._plot_initialized = False
        self._fit_axes_once = False
        self._last_y_scale = "Linear"

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._bind_shortcuts()
        self._connect_plot_events()
        self.update_plot()

    def _configure_window(self) -> None:
        self.root.title("Huang–Rhys Spectrum Explorer")
        self.root.geometry("1320x900")
        self.root.minsize(1120, 820)
        self.root.configure(bg=COLORS["app"])

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["app"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("TNotebook", background=COLORS["card"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#EEF3F9",
            foreground=COLORS["muted"],
            padding=(13, 7),
            font=(FONT, 9, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["card"])],
            foreground=[("selected", COLORS["accent"])],
        )
        style.configure(
            "Header.TLabel",
            background=COLORS["app"],
            foreground=COLORS["text"],
            font=(FONT, 22, "bold"),
        )
        style.configure(
            "Subheader.TLabel",
            background=COLORS["app"],
            foreground=COLORS["muted"],
            font=(FONT, 10),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=(FONT, 11, "bold"),
        )
        style.configure(
            "Parameter.TLabel",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=(FONT, 9),
        )
        style.configure(
            "Muted.TLabel",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            font=(FONT, 8),
        )
        style.configure(
            "Status.TLabel",
            background=COLORS["app"],
            foreground=COLORS["muted"],
            font=(FONT, 9),
        )
        style.configure(
            "Value.TEntry",
            fieldbackground="#F8FAFC",
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["border"],
            darkcolor=COLORS["border"],
            padding=(5, 3),
            font=(FONT, 9),
        )
        style.configure(
            "Parameter.Horizontal.TScale",
            background=COLORS["card"],
            troughcolor="#DCE6F4",
        )
        style.configure(
            "Primary.TButton",
            background=COLORS["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(12, 8),
            font=(FONT, 9, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["accent_hover"]), ("pressed", "#1E40AF")],
        )
        style.configure(
            "Secondary.TButton",
            background="#EEF3F9",
            foreground=COLORS["text"],
            borderwidth=0,
            padding=(10, 8),
            font=(FONT, 9),
        )
        style.map("Secondary.TButton", background=[("active", "#E1E9F3")])
        style.configure(
            "TCombobox",
            fieldbackground="#F8FAFC",
            background="#F8FAFC",
            bordercolor=COLORS["border"],
            padding=4,
            font=(FONT, 9),
        )
        # Copy the button layout to create indicator-free segmented radios.
        style.layout("Segment.TRadiobutton", style.layout("TButton"))
        style.configure(
            "Segment.TRadiobutton",
            background="#EEF3F9",
            foreground=COLORS["muted"],
            borderwidth=0,
            padding=(9, 6),
            anchor="center",
            font=(FONT, 8, "bold"),
        )
        style.map(
            "Segment.TRadiobutton",
            background=[("selected", COLORS["accent"]), ("active", "#E1E9F3")],
            foreground=[("selected", "#FFFFFF"), ("active", COLORS["text"])],
        )
        style.configure(
            "Clean.TCheckbutton",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=(FONT, 9),
        )
        style.configure(
            "Clean.TRadiobutton",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=(FONT, 9),
        )
        style.layout("LogToggle.TCheckbutton", style.layout("TButton"))
        style.configure(
            "LogToggle.TCheckbutton",
            background="#E8EEF7",
            foreground=COLORS["text"],
            borderwidth=1,
            padding=(13, 8),
            anchor="center",
            font=(FONT, 9, "bold"),
        )
        style.map(
            "LogToggle.TCheckbutton",
            background=[
                ("selected", COLORS["accent"]),
                ("active", "#D7E3F4"),
            ],
            foreground=[("selected", "#FFFFFF")],
        )

        mpl.rcParams.update(
            {
                "font.family": FONT,
                "font.size": 10,
                "axes.labelcolor": COLORS["text"],
                "xtick.color": COLORS["muted"],
                "ytick.color": COLORS["muted"],
                "text.color": COLORS["text"],
            }
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 15, 24, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(header, text="Huang–Rhys Spectrum Explorer", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Interactive vibronic spectra, self-absorption, and photon recycling",
            style="Subheader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.sidebar = ttk.Frame(
            self.root, style="Card.TFrame", padding=(18, 16), width=370
        )
        self.sidebar.grid(row=1, column=0, sticky="nsw", padx=(24, 12), pady=(0, 12))
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(5, weight=1)

        self._build_data_section()
        ttk.Separator(self.sidebar).grid(row=4, column=0, sticky="ew", pady=12)

        self.control_notebook = ttk.Notebook(self.sidebar)
        self.control_notebook.grid(row=5, column=0, sticky="nsew")
        self.model_tab = ttk.Frame(
            self.control_notebook, style="Card.TFrame", padding=(10, 12)
        )
        self.peaks_tab = ttk.Frame(
            self.control_notebook, style="Card.TFrame", padding=(10, 12)
        )
        self.recycling_tab = ttk.Frame(
            self.control_notebook, style="Card.TFrame", padding=(10, 12)
        )
        self.plot_tab = ttk.Frame(
            self.control_notebook, style="Card.TFrame", padding=(10, 12)
        )
        self.control_notebook.add(self.model_tab, text="Spectrum")
        self.control_notebook.add(self.peaks_tab, text="ZPL branches")
        self.control_notebook.add(self.recycling_tab, text="Photon recycling")
        self.control_notebook.add(self.plot_tab, text="Plot")
        self._build_model_section(self.model_tab)
        self.peaks_tab.columnconfigure(0, weight=1)
        self._build_extra_peaks_section(self.peaks_tab, row=0)
        self._build_recycling_section(self.recycling_tab)
        self._build_plot_section(self.plot_tab)
        self._build_action_section()

        plot_card = ttk.Frame(self.root, style="Card.TFrame", padding=(16, 14))
        plot_card.grid(row=1, column=1, sticky="nsew", padx=(0, 24), pady=(0, 12))
        plot_card.rowconfigure(0, weight=1)
        plot_card.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8.5, 6.5), dpi=100, facecolor=COLORS["plot"])
        self.ax = self.figure.add_subplot(111)
        self.abs_ax = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_card)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(plot_card, style="Card.TFrame")
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        toolbar_frame.columnconfigure(2, weight=1)
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, toolbar_frame, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            toolbar_frame,
            text="LOG Y-SCALE",
            variable=self.y_scale,
            onvalue="Log",
            offvalue="Linear",
            command=self.schedule_update,
            style="LogToggle.TCheckbutton",
        ).grid(row=0, column=1, sticky="w", padx=(10, 8))
        self.cursor_status = tk.StringVar(
            value="Scroll: zoom x  •  Double-click: set primary ZPL  •  Shift-click: add ZPL"
        )
        ttk.Label(
            toolbar_frame,
            textvariable=self.cursor_status,
            style="Muted.TLabel",
            anchor="e",
        ).grid(row=0, column=2, sticky="e", padx=(10, 0))

        self.status = tk.StringVar(value="Ready")
        ttk.Label(
            self.root,
            textvariable=self.status,
            style="Status.TLabel",
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(0, 9))

    def _section_title(self, parent: tk.Misc, row: int, text: str) -> None:
        ttk.Label(parent, text=text, style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 8)
        )

    def _build_data_section(self) -> None:
        self._section_title(self.sidebar, 0, "Experimental emission data")
        data_controls = ttk.Frame(self.sidebar, style="Card.TFrame")
        data_controls.grid(row=1, column=0, sticky="ew")
        data_controls.columnconfigure(0, weight=1)
        data_controls.columnconfigure(1, weight=1)

        ttk.Button(
            data_controls,
            text="Load spectrum",
            command=self.load_data,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(
            data_controls,
            text="Clear",
            command=self.clear_data,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        unit_row = ttk.Frame(self.sidebar, style="Card.TFrame")
        unit_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        unit_row.columnconfigure(1, weight=1)
        ttk.Label(unit_row, text="Imported x-axis", style="Parameter.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.import_unit = tk.StringVar(value="Auto")
        unit_box = ttk.Combobox(
            unit_row,
            textvariable=self.import_unit,
            values=("Auto", "Energy (eV)", "Wavelength (nm)"),
            state="readonly",
            width=16,
        )
        unit_box.grid(row=0, column=1, sticky="ew")

        self.data_summary = tk.StringVar(value="No spectrum loaded")
        ttk.Label(
            self.sidebar,
            textvariable=self.data_summary,
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(7, 0))

    def _build_model_section(self, parent: tk.Misc) -> None:
        parent.columnconfigure(0, weight=1)
        self._section_title(parent, 0, "Model parameters")
        parameter_frame = ttk.Frame(parent, style="Card.TFrame")
        parameter_frame.grid(row=1, column=0, sticky="ew")
        parameter_frame.columnconfigure(0, weight=1)

        specs = (
            ("S", "Huang–Rhys factor, S", 0.0, 10.0, self.DEFAULTS["S"], 3, ""),
            ("zpl", "Zero-phonon line", 1.5, 3.5, self.DEFAULTS["zpl"], 4, " eV"),
            ("fwhm_meV", "Linewidth, FWHM", 0.1, 100.0, self.DEFAULTS["fwhm_meV"], 2, " meV"),
            ("phonon_meV", "Phonon energy", 0.1, 100.0, self.DEFAULTS["phonon_meV"], 2, " meV"),
            (
                "broadening_multiplier",
                "Successive linewidth multiplier",
                1.0,
                3.0,
                self.DEFAULTS["broadening_multiplier"],
                3,
                "×",
            ),
            ("model_scale", "Model scale", 0.01, 5.0, self.DEFAULTS["model_scale"], 3, ""),
            ("data_scale", "Data scale", 0.01, 5.0, self.DEFAULTS["data_scale"], 3, ""),
        )
        self.parameters: dict[str, ParameterControl] = {}
        for row, (key, label, minimum, maximum, initial, decimals, suffix) in enumerate(specs):
            control = ParameterControl(
                parameter_frame,
                label=label,
                minimum=minimum,
                maximum=maximum,
                initial=initial,
                decimals=decimals,
                suffix=suffix,
                command=self.schedule_update,
            )
            control.grid(row=row, column=0, sticky="ew", pady=(0, 7))
            self.parameters[key] = control

        ttk.Separator(parent).grid(row=2, column=0, sticky="ew", pady=(7, 10))
        self._build_display_section(parent, row=3)

    def _segmented_row(
        self,
        parent: tk.Misc,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, str],
        command=None,
    ) -> None:
        ttk.Label(parent, text=label, style="Parameter.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 5)
        )
        segment = ttk.Frame(parent, style="Card.TFrame")
        segment.grid(row=row + 1, column=0, sticky="ew", pady=(0, 8))
        segment.columnconfigure(0, weight=1)
        segment.columnconfigure(1, weight=1)
        for column, value in enumerate(values):
            ttk.Radiobutton(
                segment,
                text=value,
                value=value,
                variable=variable,
                command=command or self.schedule_update,
                style="Segment.TRadiobutton",
            ).grid(row=0, column=column, sticky="ew", padx=(0, 3) if column == 0 else (3, 0))

    def _build_display_section(self, parent: tk.Misc, *, row: int) -> None:
        self._section_title(parent, row, "Display")
        display = ttk.Frame(parent, style="Card.TFrame")
        display.grid(row=row + 1, column=0, sticky="ew")
        display.columnconfigure(0, weight=1)

        self.line_shape = tk.StringVar(value="Gaussian")
        self.direction = tk.StringVar(value="Emission")
        self.y_scale = tk.StringVar(value="Linear")
        self.show_replicas = tk.BooleanVar(value=False)

        self._segmented_row(
            display,
            row=0,
            label="Line shape",
            variable=self.line_shape,
            values=("Gaussian", "Lorentzian"),
        )
        self._segmented_row(
            display,
            row=2,
            label="Process",
            variable=self.direction,
            values=("Emission", "Absorption"),
        )
        scale_row = ttk.Frame(display, style="Card.TFrame")
        scale_row.grid(row=4, column=0, sticky="ew")
        scale_row.columnconfigure(0, weight=1)
        ttk.Label(
            scale_row,
            text="Y scale",
            style="Parameter.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            scale_row,
            text="Linear",
            variable=self.y_scale,
            value="Linear",
            command=self.schedule_update,
            style="Clean.TRadiobutton",
        ).grid(row=0, column=1, sticky="e", padx=(4, 0))
        ttk.Radiobutton(
            scale_row,
            text="Log",
            variable=self.y_scale,
            value="Log",
            command=self.schedule_update,
            style="Clean.TRadiobutton",
        ).grid(row=0, column=2, sticky="e", padx=(4, 0))

    def _build_plot_section(self, parent: tk.Misc) -> None:
        """Build persistent axis controls and editable graph text."""

        parent.columnconfigure(0, weight=1)
        self._section_title(parent, 0, "Axis view")
        axis_frame = ttk.Frame(parent, style="Card.TFrame")
        axis_frame.grid(row=1, column=0, sticky="ew")
        for column in range(4):
            axis_frame.columnconfigure(column, weight=1)

        self.preserve_axes = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            axis_frame,
            text="Preserve zoom and limits when the model updates",
            variable=self.preserve_axes,
            command=self._on_preserve_axes_changed,
            style="Clean.TCheckbutton",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 7))

        self.axis_limit_values = {
            "x_min": tk.StringVar(),
            "x_max": tk.StringVar(),
            "y_min": tk.StringVar(),
            "y_max": tk.StringVar(),
        }
        for column, (key, label) in enumerate(
            (("x_min", "X min"), ("x_max", "X max"), ("y_min", "Y min"), ("y_max", "Y max"))
        ):
            ttk.Label(axis_frame, text=label, style="Muted.TLabel").grid(
                row=1, column=column, sticky="w", padx=(0, 4)
            )
            entry = ttk.Entry(
                axis_frame,
                textvariable=self.axis_limit_values[key],
                width=8,
                justify="center",
                style="Value.TEntry",
            )
            entry.grid(row=2, column=column, sticky="ew", padx=(0, 4))
            entry.bind("<Return>", self.apply_axis_limits)

        axis_buttons = ttk.Frame(axis_frame, style="Card.TFrame")
        axis_buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        axis_buttons.columnconfigure(0, weight=1)
        axis_buttons.columnconfigure(1, weight=1)
        ttk.Button(
            axis_buttons,
            text="Apply limits",
            command=self.apply_axis_limits,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            axis_buttons,
            text="Fit axes now",
            command=self.fit_axes,
            style="Primary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Separator(parent).grid(row=2, column=0, sticky="ew", pady=(12, 10))
        self._section_title(parent, 3, "Visible plot layers")
        layer_frame = ttk.Frame(parent, style="Card.TFrame")
        layer_frame.grid(row=4, column=0, sticky="ew")
        layer_frame.columnconfigure(0, weight=1)
        layer_frame.columnconfigure(1, weight=1)
        self.show_final_spectrum = tk.BooleanVar(value=True)
        self.show_intrinsic_spectrum = tk.BooleanVar(value=False)
        self.show_data_layer = tk.BooleanVar(value=True)
        self.show_zpl_components = tk.BooleanVar(value=False)
        self.show_zpl_markers = tk.BooleanVar(value=True)
        layers = (
            ("Final/model spectrum", self.show_final_spectrum),
            ("Intrinsic before recycling", self.show_intrinsic_spectrum),
            ("Experimental data", self.show_data_layer),
            ("ZPL component curves", self.show_zpl_components),
            ("Individual replicas", self.show_replicas),
            ("ZPL position markers", self.show_zpl_markers),
            ("Absorption profile", self.show_absorption),
        )
        for index, (label, variable) in enumerate(layers):
            ttk.Checkbutton(
                layer_frame,
                text=label,
                variable=variable,
                command=self.schedule_update,
                style="Clean.TCheckbutton",
            ).grid(
                row=index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 5) if index % 2 == 0 else (5, 0),
                pady=(0, 3),
            )

        ttk.Separator(parent).grid(row=5, column=0, sticky="ew", pady=(10, 9))
        self._section_title(parent, 6, "Text and legend")
        text_frame = ttk.Frame(parent, style="Card.TFrame")
        text_frame.grid(row=7, column=0, sticky="ew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)

        self.plot_text = {
            key: tk.StringVar(value=value) for key, value in self.TEXT_DEFAULTS.items()
        }
        fields = (
            ("title", "Title (blank = automatic)"),
            ("x_label", "X-axis label"),
            ("y_label", "Y-axis label"),
            ("absorption_y_label", "Right-axis label"),
            ("model_legend", "Single-model legend"),
            ("combined_legend", "Combined-model legend"),
            ("intrinsic_legend", "Intrinsic legend"),
            ("escaped_legend", "Escaped legend"),
            ("data_legend", "Data legend (blank = file)"),
            ("absorption_legend", "Absorption legend"),
            ("branch_prefix", "Branch legend prefix"),
        )
        for index, (key, label) in enumerate(fields):
            block = ttk.Frame(text_frame, style="Card.TFrame")
            block.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=(0, 4) if index % 2 == 0 else (4, 0),
                pady=(0, 5),
            )
            block.columnconfigure(0, weight=1)
            ttk.Label(block, text=label, style="Muted.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            entry = ttk.Entry(
                block,
                textvariable=self.plot_text[key],
                style="Value.TEntry",
            )
            entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))
            entry.bind("<Return>", lambda _event: self.schedule_update())
            entry.bind("<FocusOut>", lambda _event: self.schedule_update())

        legend_row = ttk.Frame(parent, style="Card.TFrame")
        legend_row.grid(row=8, column=0, sticky="ew", pady=(7, 0))
        legend_row.columnconfigure(1, weight=1)
        self.show_legend = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            legend_row,
            text="Show legend",
            variable=self.show_legend,
            command=self.schedule_update,
            style="Clean.TCheckbutton",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.legend_location = tk.StringVar(value="upper right")
        location = ttk.Combobox(
            legend_row,
            textvariable=self.legend_location,
            values=(
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
            ),
            state="readonly",
            width=13,
        )
        location.grid(row=0, column=1, sticky="e")
        location.bind("<<ComboboxSelected>>", lambda _event: self.schedule_update())

    def _build_extra_peaks_section(self, parent: tk.Misc, *, row: int) -> None:
        self._section_title(parent, row, "Additional ZPL branches")
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row + 1, column=0, sticky="ew")
        for column in range(3):
            frame.columnconfigure(column, weight=1)

        self.primary_zpl_active = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Primary ZPL active (controlled by Spectrum sliders)",
            variable=self.primary_zpl_active,
            command=self._on_primary_zpl_active_changed,
            style="Clean.TCheckbutton",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 7))

        self.extra_peaks_tree = ttk.Treeview(
            frame,
            columns=(
                "state",
                "energy",
                "amplitude",
                "S",
                "phonon",
                "fwhm",
                "multiplier",
            ),
            show="headings",
            height=4,
            selectmode="browse",
        )
        headings = {
            "state": "State",
            "energy": "E (eV)",
            "amplitude": "Amp",
            "S": "S",
            "phonon": "ℏω",
            "fwhm": "Γ₀",
            "multiplier": "m",
        }
        widths = {
            "state": 46,
            "energy": 56,
            "amplitude": 43,
            "S": 36,
            "phonon": 42,
            "fwhm": 42,
            "multiplier": 36,
        }
        for name, heading in headings.items():
            self.extra_peaks_tree.heading(name, text=heading)
            self.extra_peaks_tree.column(
                name, width=widths[name], minwidth=34, anchor="center", stretch=True
            )
        self.extra_peaks_tree.grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )
        self.extra_peaks_tree.bind("<<TreeviewSelect>>", self._on_extra_peak_select)

        self.extra_peak_energy = tk.StringVar(value="2.2600")
        self.extra_peak_amplitude = tk.StringVar(value="0.500")
        self.extra_peak_S = tk.StringVar(value=f"{self.DEFAULTS['S']:.3f}")
        self.extra_peak_phonon_meV = tk.StringVar(
            value=f"{self.DEFAULTS['phonon_meV']:.2f}"
        )
        self.extra_peak_fwhm_meV = tk.StringVar(
            value=f"{self.DEFAULTS['fwhm_meV']:.2f}"
        )
        self.extra_peak_broadening_multiplier = tk.StringVar(
            value=f"{self.DEFAULTS['broadening_multiplier']:.3f}"
        )
        self.extra_peak_active = tk.BooleanVar(value=True)
        fields = (
            ("Energy (eV)", self.extra_peak_energy),
            ("Amplitude", self.extra_peak_amplitude),
            ("Huang–Rhys S", self.extra_peak_S),
            ("Phonon (meV)", self.extra_peak_phonon_meV),
            ("FWHM₀ (meV)", self.extra_peak_fwhm_meV),
            ("Width m", self.extra_peak_broadening_multiplier),
        )
        for index, (label, variable) in enumerate(fields):
            block_row = 2 + 2 * (index // 3)
            column = index % 3
            ttk.Label(frame, text=label, style="Muted.TLabel").grid(
                row=block_row,
                column=column,
                sticky="w",
                padx=(0, 3) if column < 2 else 0,
            )
            entry = ttk.Entry(
                frame,
                textvariable=variable,
                width=8,
                justify="center",
                style="Value.TEntry",
            )
            entry.grid(
                row=block_row + 1,
                column=column,
                sticky="ew",
                padx=(0, 3) if column < 2 else 0,
                pady=(2, 5),
            )
            entry.bind("<Return>", lambda _event: self.add_extra_peak())

        ttk.Checkbutton(
            frame,
            text="Selected/additional branch active in the summed model",
            variable=self.extra_peak_active,
            command=self._on_extra_peak_active_changed,
            style="Clean.TCheckbutton",
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(1, 4))

        buttons = ttk.Frame(frame, style="Card.TFrame")
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        for column in range(3):
            buttons.columnconfigure(column, weight=1)
        ttk.Button(
            buttons,
            text="Add",
            command=self.add_extra_peak,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            buttons,
            text="Update",
            command=self.update_extra_peak,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(
            buttons,
            text="Remove",
            command=self.remove_extra_peak,
            style="Secondary.TButton",
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        ttk.Label(
            frame,
            text=(
                "Each branch has its own S, ℏω, Γ₀, width multiplier, and color. "
                "Muted branches are excluded from the summed spectrum."
            ),
            style="Muted.TLabel",
            wraplength=310,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _read_extra_peak_entries(self) -> dict[str, float | bool] | None:
        try:
            energy = float(self.extra_peak_energy.get().strip())
            amplitude = float(self.extra_peak_amplitude.get().strip())
            factor = float(self.extra_peak_S.get().strip())
            phonon_meV = float(self.extra_peak_phonon_meV.get().strip())
            fwhm_meV = float(self.extra_peak_fwhm_meV.get().strip())
            multiplier = float(
                self.extra_peak_broadening_multiplier.get().strip()
            )
        except ValueError:
            messagebox.showerror(
                "Invalid ZPL branch",
                "Every ZPL branch field must be numeric.",
                parent=self.root,
            )
            return None
        if not self.parameters["zpl"].minimum <= energy <= self.parameters["zpl"].maximum:
            messagebox.showerror(
                "Invalid ZPL branch",
                "ZPL energy must lie between 1.5 and 3.5 eV.",
                parent=self.root,
            )
            return None
        if not 0.0 < amplitude <= 10.0:
            messagebox.showerror(
                "Invalid ZPL branch",
                "Relative amplitude must be greater than 0 and no larger than 10.",
                parent=self.root,
            )
            return None
        if not 0.0 <= factor <= 10.0:
            messagebox.showerror(
                "Invalid ZPL branch", "Huang–Rhys S must be between 0 and 10.", parent=self.root
            )
            return None
        if not 0.1 <= phonon_meV <= 100.0:
            messagebox.showerror(
                "Invalid ZPL branch",
                "Phonon energy must be between 0.1 and 100 meV.",
                parent=self.root,
            )
            return None
        if not 0.1 <= fwhm_meV <= 100.0:
            messagebox.showerror(
                "Invalid ZPL branch", "FWHM₀ must be between 0.1 and 100 meV.", parent=self.root
            )
            return None
        if not 1.0 <= multiplier <= 3.0:
            messagebox.showerror(
                "Invalid ZPL branch",
                "Width multiplier m must be between 1 and 3.",
                parent=self.root,
            )
            return None
        return {
            "energy_eV": energy,
            "amplitude": amplitude,
            "S": factor,
            "phonon_meV": phonon_meV,
            "fwhm_meV": fwhm_meV,
            "broadening_multiplier": multiplier,
            "active": bool(self.extra_peak_active.get()),
        }

    def _refresh_extra_peaks_tree(self, select_index: int | None = None) -> None:
        children = self.extra_peaks_tree.get_children()
        if children:
            self.extra_peaks_tree.delete(*children)
        for index, peak in enumerate(self.extra_peaks):
            iid = f"peak-{index}"
            self.extra_peaks_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    "Active" if peak.get("active", True) else "Muted",
                    f"{peak['energy_eV']:.4f}",
                    f"{peak['amplitude']:.3f}",
                    f"{peak['S']:.2f}",
                    f"{peak['phonon_meV']:.1f}",
                    f"{peak['fwhm_meV']:.1f}",
                    f"{peak['broadening_multiplier']:.2f}",
                ),
            )
        if select_index is not None and 0 <= select_index < len(self.extra_peaks):
            iid = f"peak-{select_index}"
            self.extra_peaks_tree.selection_set(iid)
            self.extra_peaks_tree.focus(iid)
            self.extra_peaks_tree.see(iid)

    def _selected_extra_peak_index(self) -> int | None:
        selected = self.extra_peaks_tree.selection()
        if not selected:
            return None
        try:
            return int(selected[0].split("-", maxsplit=1)[1])
        except (IndexError, ValueError):
            return None

    def _on_extra_peak_select(self, _event=None) -> None:
        index = self._selected_extra_peak_index()
        if index is None or index >= len(self.extra_peaks):
            return
        peak = self.extra_peaks[index]
        self.extra_peak_energy.set(f"{peak['energy_eV']:.4f}")
        self.extra_peak_amplitude.set(f"{peak['amplitude']:.3f}")
        self.extra_peak_S.set(f"{peak['S']:.3f}")
        self.extra_peak_phonon_meV.set(f"{peak['phonon_meV']:.2f}")
        self.extra_peak_fwhm_meV.set(f"{peak['fwhm_meV']:.2f}")
        self.extra_peak_broadening_multiplier.set(
            f"{peak['broadening_multiplier']:.3f}"
        )
        self.extra_peak_active.set(bool(peak.get("active", True)))

    def _on_primary_zpl_active_changed(self) -> None:
        if not self.primary_zpl_active.get() and not any(
            bool(peak.get("active", True)) for peak in self.extra_peaks
        ):
            self.primary_zpl_active.set(True)
            messagebox.showinfo(
                "One active ZPL required",
                "Activate an additional ZPL before muting the primary branch.",
                parent=self.root,
            )
            return
        self.schedule_update()

    def _on_extra_peak_active_changed(self) -> None:
        index = self._selected_extra_peak_index()
        if index is None or index >= len(self.extra_peaks):
            return
        requested = bool(self.extra_peak_active.get())
        if (
            not requested
            and not self.primary_zpl_active.get()
            and not any(
                bool(peak.get("active", True))
                for other_index, peak in enumerate(self.extra_peaks)
                if other_index != index
            )
        ):
            self.extra_peak_active.set(True)
            messagebox.showinfo(
                "One active ZPL required",
                "At least one ZPL branch must remain active in the summed model.",
                parent=self.root,
            )
            return
        self.extra_peaks[index]["active"] = requested
        self._refresh_extra_peaks_tree(index)
        self.schedule_update()

    def add_extra_peak(
        self, energy_eV: float | None = None, amplitude: float | None = None
    ) -> None:
        if len(self.extra_peaks) >= 8:
            messagebox.showwarning(
                "ZPL branch limit",
                "The interactive app supports up to eight additional ZPL branches.",
                parent=self.root,
            )
            return
        if energy_eV is None or amplitude is None:
            branch = self._read_extra_peak_entries()
            if branch is None:
                return
        else:
            branch = {
                "energy_eV": float(
                    np.clip(
                        energy_eV,
                        self.parameters["zpl"].minimum,
                        self.parameters["zpl"].maximum,
                    )
                ),
                "amplitude": float(np.clip(amplitude, 1e-6, 10.0)),
                "S": self.parameters["S"].get(),
                "phonon_meV": self.parameters["phonon_meV"].get(),
                "fwhm_meV": self.parameters["fwhm_meV"].get(),
                "broadening_multiplier": self.parameters[
                    "broadening_multiplier"
                ].get(),
                "active": True,
            }
        self.extra_peaks.append(branch)
        self._refresh_extra_peaks_tree(len(self.extra_peaks) - 1)
        self.control_notebook.select(self.peaks_tab)
        self.schedule_update()

    def update_extra_peak(self) -> None:
        index = self._selected_extra_peak_index()
        if index is None:
            messagebox.showinfo(
                "Select a ZPL branch",
                "Select an additional branch in the table before updating it.",
                parent=self.root,
            )
            return
        values = self._read_extra_peak_entries()
        if values is None:
            return
        if (
            not bool(values["active"])
            and not self.primary_zpl_active.get()
            and not any(
                bool(peak.get("active", True))
                for other_index, peak in enumerate(self.extra_peaks)
                if other_index != index
            )
        ):
            messagebox.showinfo(
                "One active ZPL required",
                "At least one ZPL branch must remain active in the summed model.",
                parent=self.root,
            )
            return
        self.extra_peaks[index] = values
        self._refresh_extra_peaks_tree(index)
        self.schedule_update()

    def remove_extra_peak(self) -> None:
        index = self._selected_extra_peak_index()
        if index is None:
            return
        self.extra_peaks.pop(index)
        if not self.primary_zpl_active.get() and not any(
            bool(peak.get("active", True)) for peak in self.extra_peaks
        ):
            self.primary_zpl_active.set(True)
        next_index = min(index, len(self.extra_peaks) - 1) if self.extra_peaks else None
        self._refresh_extra_peaks_tree(next_index)
        self.schedule_update()

    def _build_recycling_section(self, parent: tk.Misc) -> None:
        parent.columnconfigure(0, weight=1)
        self.recycling_enabled = tk.BooleanVar(value=True)
        self.recycling_mode = tk.StringVar(value="Beer–Lambert")
        self.show_absorption = tk.BooleanVar(value=False)

        top = ttk.Frame(parent, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        top.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            top,
            text="Enable recycling",
            variable=self.recycling_enabled,
            command=self._on_recycling_controls_changed,
            style="Clean.TCheckbutton",
        ).grid(row=0, column=0, sticky="w")
        self.show_absorption_check = ttk.Checkbutton(
            top,
            text="Show absorption",
            variable=self.show_absorption,
            command=self.schedule_update,
            style="Clean.TCheckbutton",
        )
        self.show_absorption_check.grid(row=0, column=1, sticky="e")

        self._segmented_row(
            parent,
            row=1,
            label="Recycling model",
            variable=self.recycling_mode,
            values=("Beer–Lambert", "Simple"),
            command=self._on_recycling_controls_changed,
        )

        source_buttons = ttk.Frame(parent, style="Card.TFrame")
        source_buttons.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        source_buttons.columnconfigure(0, weight=1)
        source_buttons.columnconfigure(1, weight=1)
        self.load_absorption_button = ttk.Button(
            source_buttons,
            text="Load absorption",
            command=self.load_absorption,
            style="Primary.TButton",
        )
        self.load_absorption_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.reference_absorption_button = ttk.Button(
            source_buttons,
            text="Use CsPbBr₃ reference",
            command=self.use_reference_absorption,
            style="Secondary.TButton",
        )
        self.reference_absorption_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.absorption_summary = tk.StringVar(
            value="Built-in CsPbBr₃ reference • 47 digitized points"
        )
        ttk.Label(
            parent,
            textvariable=self.absorption_summary,
            style="Muted.TLabel",
            wraplength=310,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

        parameter_frame = ttk.Frame(parent, style="Card.TFrame")
        parameter_frame.grid(row=5, column=0, sticky="ew")
        parameter_frame.columnconfigure(0, weight=1)
        specs = (
            ("abs_shift_meV", "Absorption shift, ΔE(T)", -100.0, 100.0, 0.0, 1, " meV"),
            ("abs_broadening_meV", "Extra absorption broadening", 0.0, 100.0, 0.0, 1, " meV"),
            ("peak_optical_depth", "Peak optical depth, αL", 0.0, 8.0, 1.0, 3, ""),
            ("plqy", "Internal PLQY", 0.0, 1.0, 0.90, 3, ""),
            ("max_reemissions", "Maximum re-emissions", 0.0, 30.0, 12.0, 0, ""),
            ("simple_strength", "Recycling strength", 0.0, 0.98, 0.50, 3, ""),
        )
        self.recycling_parameters: dict[str, ParameterControl] = {}
        for row, (key, label, minimum, maximum, initial, decimals, suffix) in enumerate(specs):
            control = ParameterControl(
                parameter_frame,
                label=label,
                minimum=minimum,
                maximum=maximum,
                initial=initial,
                decimals=decimals,
                suffix=suffix,
                command=self.schedule_update,
            )
            control.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            self.recycling_parameters[key] = control

        self.recycling_metrics = tk.StringVar(
            value="Escape and loss metrics will update with the spectrum."
        )
        ttk.Label(
            parent,
            textvariable=self.recycling_metrics,
            style="Muted.TLabel",
            wraplength=310,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(7, 0))
        ttk.Label(
            parent,
            text=(
                "Temperature controls are phenomenological; +ΔE shifts "
                "absorption to higher energy."
            ),
            style="Muted.TLabel",
            wraplength=310,
            justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(5, 0))
        self._sync_recycling_states()

    def _on_recycling_controls_changed(self) -> None:
        self._sync_recycling_states()
        self.schedule_update()

    def _sync_recycling_states(self) -> None:
        enabled = bool(self.recycling_enabled.get())
        physical = enabled and self.recycling_mode.get() == "Beer–Lambert"
        simple = enabled and self.recycling_mode.get() == "Simple"
        for key in ("abs_shift_meV", "abs_broadening_meV"):
            self.recycling_parameters[key].set_enabled(enabled)
        for key in ("peak_optical_depth", "plqy", "max_reemissions"):
            self.recycling_parameters[key].set_enabled(physical)
        self.recycling_parameters["simple_strength"].set_enabled(simple)
        if enabled:
            self.load_absorption_button.state(["!disabled"])
            self.reference_absorption_button.state(["!disabled"])
            self.show_absorption_check.state(["!disabled"])
        else:
            self.load_absorption_button.state(["disabled"])
            self.reference_absorption_button.state(["disabled"])
            self.show_absorption_check.state(["disabled"])

    def _build_action_section(self) -> None:
        action = ttk.Frame(self.sidebar, style="Card.TFrame")
        action.grid(row=6, column=0, sticky="sew", pady=(12, 0))
        action.columnconfigure(0, weight=1)
        action.columnconfigure(1, weight=1)
        action.columnconfigure(2, weight=1)
        ttk.Button(
            action, text="Reset", command=self.reset, style="Secondary.TButton"
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            action, text="Save PNG", command=self.export_plot, style="Secondary.TButton"
        ).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(
            action, text="Save CSV", command=self.export_model, style="Secondary.TButton"
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))

    def _bind_shortcuts(self) -> None:
        modifier = "Command" if platform.system() == "Darwin" else "Control"
        self.root.bind_all(f"<{modifier}-o>", lambda _event: self.load_data())
        self.root.bind_all(f"<{modifier}-s>", lambda _event: self.export_plot())
        self.root.bind_all(f"<{modifier}-r>", lambda _event: self.reset())

    def _connect_plot_events(self) -> None:
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)
        self.canvas.mpl_connect("button_release_event", self._on_plot_button_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("axes_leave_event", self._on_plot_leave)

    def _toolbar_is_active(self) -> bool:
        return bool(str(self.toolbar.mode))

    def _on_plot_scroll(self, event) -> None:
        if event.xdata is None or event.inaxes not in (self.ax, self.abs_ax):
            return
        left, right = self.ax.get_xlim()
        scale = 1.0 / 1.22 if event.button == "up" else 1.22
        cursor = float(event.xdata)
        new_left = cursor - (cursor - left) * scale
        new_right = cursor + (right - cursor) * scale
        self.ax.set_xlim(max(0.001, new_left), new_right)
        self.canvas.draw_idle()
        self._sync_axis_entries()

    def _on_plot_button_release(self, event) -> None:
        if event.inaxes in (self.ax, self.abs_ax):
            self.root.after_idle(self._sync_axis_entries)

    def _on_plot_click(self, event) -> None:
        if (
            event.button != 1
            or event.xdata is None
            or event.inaxes not in (self.ax, self.abs_ax)
            or self._toolbar_is_active()
        ):
            return
        energy = float(event.xdata)
        key = str(event.key or "").lower()
        if "shift" in key:
            self.extra_peak_energy.set(f"{energy:.4f}")
            self.extra_peak_amplitude.set("0.500")
            self.add_extra_peak(energy_eV=energy, amplitude=0.5)
            return
        if event.dblclick:
            self.parameters["zpl"].set(energy)
            self.schedule_update()

    def _on_plot_motion(self, event) -> None:
        if event.xdata is None or event.inaxes not in (self.ax, self.abs_ax):
            self._on_plot_leave()
            return
        energy = float(event.xdata)
        pieces = [f"E = {energy:.5f} eV"]
        if self.last_result is not None and self.last_model_display is not None:
            intensity = float(
                np.interp(
                    energy,
                    self.last_result.energy_eV,
                    self.last_model_display,
                    left=np.nan,
                    right=np.nan,
                )
            )
            if np.isfinite(intensity):
                pieces.append(f"I = {intensity:.4g}")
        if self.last_absorption_profile is not None and self.last_result is not None:
            absorption = float(
                np.interp(
                    energy,
                    self.last_result.energy_eV,
                    self.last_absorption_profile,
                    left=np.nan,
                    right=np.nan,
                )
            )
            if np.isfinite(absorption):
                pieces.append(f"A = {absorption:.3f}")
        self.cursor_status.set("  •  ".join(pieces))

    def _on_plot_leave(self, _event=None) -> None:
        self.cursor_status.set(
            "Scroll: zoom x  •  Double-click: set primary ZPL  •  Shift-click: add ZPL"
        )

    def schedule_update(self, *_args) -> None:
        if self.last_update_id is not None:
            self.root.after_cancel(self.last_update_id)
        self.last_update_id = self.root.after(35, self.update_plot)

    def _on_preserve_axes_changed(self) -> None:
        if self.preserve_axes.get():
            self._sync_axis_entries()
        else:
            self._fit_axes_once = True
            self.schedule_update()

    def fit_axes(self) -> None:
        """Recalculate axis limits once, then resume preserving the view."""

        self.preserve_axes.set(True)
        self._fit_axes_once = True
        self.schedule_update()

    def _sync_axis_entries(self) -> None:
        if not self._plot_initialized:
            return
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        values = (*x_limits, *y_limits)
        for key, value in zip(
            ("x_min", "x_max", "y_min", "y_max"), values, strict=True
        ):
            self.axis_limit_values[key].set(f"{float(value):.6g}")

    def apply_axis_limits(self, _event=None) -> None:
        """Apply explicit plot limits and preserve them across later updates."""

        try:
            x_min, x_max, y_min, y_max = (
                float(self.axis_limit_values[key].get().strip())
                for key in ("x_min", "x_max", "y_min", "y_max")
            )
        except ValueError:
            messagebox.showerror(
                "Invalid axis limits",
                "X min, X max, Y min, and Y max must all be numeric.",
                parent=self.root,
            )
            return
        if not np.all(np.isfinite([x_min, x_max, y_min, y_max])):
            messagebox.showerror(
                "Invalid axis limits",
                "Axis limits must contain finite numbers.",
                parent=self.root,
            )
            return
        if x_min >= x_max or y_min >= y_max:
            messagebox.showerror(
                "Invalid axis limits",
                "Each minimum must be smaller than its matching maximum.",
                parent=self.root,
            )
            return
        if self.y_scale.get() == "Log" and y_min <= 0:
            messagebox.showerror(
                "Invalid logarithmic limit",
                "Y min must be greater than zero on a logarithmic scale.",
                parent=self.root,
            )
            return

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.preserve_axes.set(True)
        self.canvas.draw_idle()
        self._sync_axis_entries()
        self.status.set("Applied manual axis limits; the view is now preserved.")

    def _absorption_source_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.absorption_data is not None:
            return self.absorption_data.energy_eV, self.absorption_data.intensity
        return CSPBBR3_ABSORPTION_ENERGY_EV, CSPBBR3_ABSORPTION_INTENSITY

    def _energy_range(self) -> tuple[float, float]:
        if self.data is not None:
            low = float(np.min(self.data.energy_eV))
            high = float(np.max(self.data.energy_eV))
            pad = max(0.01, 0.025 * (high - low))
            low, high = max(0.001, low - pad), high + pad
        else:
            low, high = np.inf, -np.inf

        active_branches = []
        if self.primary_zpl_active.get():
            active_branches.append(
                (
                    self.parameters["zpl"].get(),
                    self.parameters["S"].get(),
                    self.parameters["phonon_meV"].get(),
                    self.parameters["fwhm_meV"].get(),
                )
            )
        active_branches.extend(
            (
                float(peak["energy_eV"]),
                float(peak["S"]),
                float(peak["phonon_meV"]),
                float(peak["fwhm_meV"]),
            )
            for peak in self.extra_peaks
            if bool(peak.get("active", True))
        )
        for zpl, factor, phonon_meV, fwhm_meV in active_branches:
            phonon = phonon_meV / 1000.0
            fwhm = fwhm_meV / 1000.0
            visible_order = max(
                4, int(np.ceil(factor + 4.0 * np.sqrt(factor + 1.0)))
            )
            progression = visible_order * phonon
            pad = max(0.06, 5.0 * fwhm)
            if self.direction.get() == "Emission":
                branch_low, branch_high = zpl - progression - pad, zpl + pad
            else:
                branch_low, branch_high = zpl - pad, zpl + progression + pad
            low = min(low, branch_low)
            high = max(high, branch_high)

        if self.recycling_enabled.get() and self.show_absorption.get():
            source_energy, _ = self._absorption_source_arrays()
            shift_eV = self.recycling_parameters["abs_shift_meV"].get() / 1000.0
            broadening_eV = self.recycling_parameters["abs_broadening_meV"].get() / 1000.0
            low = min(low, float(np.min(source_energy)) + shift_eV - 2.0 * broadening_eV - 0.02)
            high = max(high, float(np.max(source_energy)) + shift_eV + 2.0 * broadening_eV + 0.02)
        return max(0.001, low), high

    def update_plot(self) -> None:
        self.last_update_id = None
        preserve_view = (
            self._plot_initialized
            and self.preserve_axes.get()
            and not self._fit_axes_once
        )
        preserved_xlim = self.ax.get_xlim() if preserve_view else None
        preserved_ylim = (
            self.ax.get_ylim()
            if preserve_view and self._last_y_scale == self.y_scale.get()
            else None
        )
        low, high = self._energy_range()
        energy = np.linspace(low, high, 2600)

        zpl_energies = [self.parameters["zpl"].get()] + [
            peak["energy_eV"] for peak in self.extra_peaks
        ]
        active_branches = [bool(self.primary_zpl_active.get())] + [
            bool(peak.get("active", True)) for peak in self.extra_peaks
        ]
        relative_amplitudes = [1.0 if active_branches[0] else 0.0] + [
            float(peak["amplitude"]) if active else 0.0
            for peak, active in zip(
                self.extra_peaks, active_branches[1:], strict=True
            )
        ]
        huang_rhys_factors = [self.parameters["S"].get()] + [
            peak["S"] for peak in self.extra_peaks
        ]
        phonon_energies_eV = [self.parameters["phonon_meV"].get() / 1000.0] + [
            peak["phonon_meV"] / 1000.0 for peak in self.extra_peaks
        ]
        base_fwhm_eV = [self.parameters["fwhm_meV"].get() / 1000.0] + [
            peak["fwhm_meV"] / 1000.0 for peak in self.extra_peaks
        ]
        broadening_multipliers = [
            self.parameters["broadening_multiplier"].get()
        ] + [peak["broadening_multiplier"] for peak in self.extra_peaks]
        result = multi_zpl_spectrum(
            energy,
            huang_rhys_factors,
            zpl_energies,
            relative_amplitudes,
            base_fwhm_eV,
            phonon_energies_eV,
            line_shape=self.line_shape.get(),
            direction=self.direction.get(),
            successive_broadening_multiplier=broadening_multipliers,
        )
        model_peak = float(np.max(result.total))
        if model_peak <= 0:
            model_peak = 1.0
        model_scale = self.parameters["model_scale"].get()
        intrinsic_display = result.total / model_peak * model_scale
        display_replicas = result.replicas / model_peak * model_scale
        self.last_branch_displays = result.branch_totals / model_peak * model_scale

        self.last_result = result
        self.last_intrinsic_display = intrinsic_display
        self.last_recycling = None
        self.last_absorption_profile = None

        absorption_profile: np.ndarray | None = None
        absorption_error: str | None = None
        if self.recycling_enabled.get():
            source_energy, source_absorption = self._absorption_source_arrays()
            try:
                absorption_profile = prepare_absorption_profile(
                    energy,
                    source_energy,
                    source_absorption,
                    shift_meV=self.recycling_parameters["abs_shift_meV"].get(),
                    broadening_fwhm_meV=self.recycling_parameters[
                        "abs_broadening_meV"
                    ].get(),
                )
                self.last_absorption_profile = absorption_profile
            except ValueError as exc:
                absorption_error = str(exc)

        recycling_active = (
            self.recycling_enabled.get()
            and self.direction.get() == "Emission"
            and absorption_profile is not None
        )
        displayed_model = intrinsic_display
        if recycling_active:
            recycling = photon_recycle(
                energy,
                result.total,
                absorption_profile,
                mode=self.recycling_mode.get(),
                peak_optical_depth=self.recycling_parameters[
                    "peak_optical_depth"
                ].get(),
                plqy=self.recycling_parameters["plqy"].get(),
                max_reemissions=int(
                    round(self.recycling_parameters["max_reemissions"].get())
                ),
                simple_strength=self.recycling_parameters[
                    "simple_strength"
                ].get(),
            )
            self.last_recycling = recycling
            intrinsic_density_peak = max(
                float(np.max(recycling.intrinsic_density)), np.finfo(float).eps
            )
            displayed_model = (
                recycling.escaped_spectrum / intrinsic_density_peak * model_scale
            )
            self.recycling_metrics.set(
                f"Escape {100 * recycling.escape_fraction:.1f}%  •  "
                f"loss {100 * recycling.nonradiative_loss_fraction:.1f}%  •  "
                f"residual {100 * recycling.residual_fraction:.2f}%\n"
                f"Recycling enhancement ×{recycling.recycling_enhancement:.2f}  •  "
                f"mean re-emissions {recycling.mean_reemissions:.2f}  •  "
                f"peak shift {recycling.peak_shift_meV:+.1f} meV"
                + (
                    "\nSimple mode assumes PLQY = 1 and up to 25 re-emissions."
                    if self.recycling_mode.get() == "Simple"
                    else ""
                )
            )
        elif absorption_error:
            self.recycling_metrics.set(f"Absorption error: {absorption_error}")
        elif self.recycling_enabled.get() and self.direction.get() != "Emission":
            self.recycling_metrics.set(
                "Photon recycling applies to emission. Switch Process to Emission to enable it."
            )
        else:
            self.recycling_metrics.set("Photon recycling is disabled.")

        self.last_model_display = displayed_model
        self.ax.clear()
        self.abs_ax.clear()
        self.ax.set_facecolor(COLORS["plot"])
        color_map = mpl.colormaps["tab10"]
        branch_colors = [
            color_map(index % color_map.N)
            for index in range(result.zpl_energies_eV.size)
        ]

        if self.show_replicas.get():
            for branch_index, color in enumerate(branch_colors):
                if not active_branches[branch_index]:
                    continue
                branch_replica_indices = np.flatnonzero(
                    result.replica_branch_indices == branch_index
                )
                branch_weights = result.weights[branch_replica_indices]
                threshold = max(
                    1e-7, float(np.max(branch_weights, initial=0.0)) * 1e-3
                )
                relevant = branch_replica_indices[branch_weights >= threshold]
                for index in relevant:
                    replica = display_replicas[index]
                    order = int(result.replica_orders[index])
                    alpha = max(0.30, 0.68 - 0.025 * order)
                    self.ax.fill_between(
                        energy, 0, replica, color=color, alpha=0.10, linewidth=0
                    )
                    self.ax.plot(
                        energy, replica, color=color, alpha=alpha, linewidth=0.75
                    )

        show_components = (
            self.show_zpl_components.get() and sum(active_branches) > 1
        )
        if show_components:
            branch_prefix = self.plot_text["branch_prefix"].get().strip()
            for branch_index, (branch, color) in enumerate(
                zip(self.last_branch_displays, branch_colors, strict=True), start=1
            ):
                if not active_branches[branch_index - 1]:
                    continue
                self.ax.plot(
                    energy,
                    branch,
                    color=color,
                    linewidth=1.25,
                    alpha=0.88,
                    linestyle=":",
                    label=(
                        f"{branch_prefix} {branch_index} progression"
                        if branch_prefix
                        else "_nolegend_"
                    ),
                    zorder=6,
                )

        if recycling_active and self.show_intrinsic_spectrum.get():
            self.ax.plot(
                energy,
                intrinsic_display,
                color=COLORS["intrinsic"],
                linewidth=1.7,
                linestyle="--",
                label=self.plot_text["intrinsic_legend"].get(),
                zorder=8,
            )
        if not recycling_active and self.show_final_spectrum.get():
            model_label = (
                self.plot_text["combined_legend"].get()
                if sum(active_branches) > 1
                else self.plot_text["model_legend"].get()
            )
            self.ax.plot(
                energy,
                intrinsic_display,
                color=COLORS["model"],
                linewidth=2.4,
                label=model_label,
                zorder=8,
            )
        if recycling_active and self.show_final_spectrum.get():
            self.ax.plot(
                energy,
                displayed_model,
                color=COLORS["accent"],
                linewidth=2.5,
                label=self.plot_text["escaped_legend"].get(),
                zorder=9,
            )

        if self.show_zpl_markers.get():
            for branch_number, (zpl, color, active) in enumerate(
                zip(
                    result.zpl_energies_eV,
                    branch_colors,
                    active_branches,
                    strict=True,
                ),
                start=1,
            ):
                if not active:
                    continue
                self.ax.plot(
                    [zpl],
                    [0.985],
                    marker="v",
                    markersize=6,
                    color=color,
                    transform=self.ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=12,
                )
                self.ax.annotate(
                    str(branch_number),
                    xy=(zpl, 0.985),
                    xycoords=self.ax.get_xaxis_transform(),
                    xytext=(0, -11),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=color,
                )

        plotted_data: np.ndarray | None = None
        if self.data is not None and self.show_data_layer.get():
            plotted_data = self.data.intensity * self.parameters["data_scale"].get()
            if self.y_scale.get() == "Log":
                plotted_data = np.where(plotted_data > 0, plotted_data, np.nan)
            self.ax.plot(
                self.data.energy_eV,
                plotted_data,
                color=COLORS["data"],
                linewidth=1.7,
                alpha=0.9,
                label=self.plot_text["data_legend"].get().strip()
                or self.data.source_name,
                zorder=7,
            )

        absorption_visible = (
            self.recycling_enabled.get()
            and self.show_absorption.get()
            and absorption_profile is not None
        )
        if absorption_visible:
            self.abs_ax.set_visible(True)
            self.abs_ax.plot(
                energy,
                absorption_profile,
                color=COLORS["absorption"],
                linewidth=1.8,
                alpha=0.9,
                label=self.plot_text["absorption_legend"].get(),
                zorder=3,
            )
            self.abs_ax.fill_between(
                energy,
                0,
                absorption_profile,
                color=COLORS["absorption"],
                alpha=0.06,
                linewidth=0,
            )
        else:
            self.abs_ax.set_visible(False)

        visible_model_curves: list[np.ndarray] = []
        if self.show_final_spectrum.get():
            visible_model_curves.append(
                displayed_model if recycling_active else intrinsic_display
            )
        if recycling_active and self.show_intrinsic_spectrum.get():
            visible_model_curves.append(intrinsic_display)
        if show_components:
            visible_model_curves.extend(
                branch
                for branch, active in zip(
                    self.last_branch_displays, active_branches, strict=True
                )
                if active
            )
        if self.show_replicas.get():
            visible_replica_indices = np.asarray(
                [
                    active_branches[int(branch_index)]
                    for branch_index in result.replica_branch_indices
                ],
                dtype=bool,
            )
            if np.any(visible_replica_indices):
                visible_model_curves.append(
                    np.max(display_replicas[visible_replica_indices], axis=0)
                )
        model_for_limits = (
            np.maximum.reduce(visible_model_curves)
            if visible_model_curves
            else np.full_like(energy, 1e-6)
        )
        self._format_axes(
            model_for_limits,
            plotted_data,
            low,
            high,
            absorption_visible=absorption_visible,
            recycling_active=recycling_active,
        )
        if preserved_xlim is not None:
            self.ax.set_xlim(*preserved_xlim)
        if preserved_ylim is not None:
            self.ax.set_ylim(*preserved_ylim)
        self._plot_initialized = True
        self._last_y_scale = self.y_scale.get()
        self._fit_axes_once = False
        self._sync_axis_entries()
        self.canvas.draw_idle()
        status = (
            f"S = {self.parameters['S'].get():.3f}   •   "
            f"ZPL = {self.parameters['zpl'].get():.4f} eV   •   "
            f"ℏω = {self.parameters['phonon_meV'].get():.2f} meV   •   "
            f"FWHM₀ = {self.parameters['fwhm_meV'].get():.2f} meV   •   "
            f"mΓ = {self.parameters['broadening_multiplier'].get():.3f}   •   "
            f"active ZPLs = {sum(active_branches)}/{len(zpl_energies)}"
        )
        if recycling_active and self.last_recycling is not None:
            status += (
                f"   •   escape = {100 * self.last_recycling.escape_fraction:.1f}%"
                f"   •   ΔEpeak = {self.last_recycling.peak_shift_meV:+.1f} meV"
            )
        self.status.set(status)

    def _format_axes(
        self,
        model: np.ndarray,
        plotted_data: np.ndarray | None,
        low: float,
        high: float,
        *,
        absorption_visible: bool,
        recycling_active: bool,
    ) -> None:
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self.ax.spines[side].set_color(COLORS["border"])
            self.ax.spines[side].set_linewidth(1.0)

        self.ax.set_xlim(low, high)
        self.ax.set_xlabel(self.plot_text["x_label"].get(), labelpad=8)
        self.ax.set_ylabel(self.plot_text["y_label"].get(), labelpad=8)
        automatic_title = (
            f"Emission · {self.recycling_mode.get()} photon recycling"
            if recycling_active
            else f"{self.direction.get()} progression · {self.line_shape.get()} broadening"
        )
        self.ax.set_title(
            self.plot_text["title"].get().strip() or automatic_title,
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=14,
        )
        self.ax.grid(True, which="major", color=COLORS["grid"], linewidth=0.8)
        self.ax.grid(True, which="minor", color=COLORS["grid"], linewidth=0.45, alpha=0.5)
        self.ax.tick_params(length=4, width=0.8)
        self.ax.minorticks_on()

        values = [model[np.isfinite(model)]]
        if plotted_data is not None:
            values.append(plotted_data[np.isfinite(plotted_data)])
        finite = np.concatenate([value for value in values if value.size])
        maximum = max(float(np.max(finite)), 1e-6)

        if self.y_scale.get() == "Log":
            self.ax.set_yscale("log")
            positive = finite[finite > 0]
            if positive.size:
                floor = max(float(np.min(positive)) * 0.7, maximum * 1e-5)
            else:
                floor = maximum * 1e-5
            self.ax.set_ylim(floor, maximum * 1.35)
        else:
            self.ax.set_yscale("linear")
            minimum = min(float(np.min(finite)), 0.0)
            lower = minimum * 1.1 if minimum < 0 else 0.0
            self.ax.set_ylim(lower, maximum * 1.12)

        if absorption_visible:
            self.abs_ax.set_ylim(0.0, 1.08)
            self.abs_ax.set_ylabel(
                self.plot_text["absorption_y_label"].get(),
                color=COLORS["absorption"],
                labelpad=9,
            )
            self.abs_ax.tick_params(
                axis="y",
                colors=COLORS["absorption"],
                length=3,
                width=0.8,
            )
            self.abs_ax.spines["right"].set_color(COLORS["absorption"])
            self.abs_ax.spines["right"].set_alpha(0.55)
            self.abs_ax.spines["top"].set_visible(False)
            self.abs_ax.spines["left"].set_visible(False)
            self.abs_ax.spines["bottom"].set_visible(False)
            self.abs_ax.grid(False)

        handles, labels = self.ax.get_legend_handles_labels()
        if absorption_visible:
            absorption_handles, absorption_labels = self.abs_ax.get_legend_handles_labels()
            handles += absorption_handles
            labels += absorption_labels
        if self.show_legend.get() and handles:
            legend = self.ax.legend(
                handles,
                labels,
                loc=self.legend_location.get(),
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor=COLORS["border"],
                framealpha=0.95,
                fontsize=9,
            )
            legend.get_frame().set_linewidth(0.8)
        self.figure.tight_layout(pad=1.3)

    def load_data(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load a spectrum",
            filetypes=(
                ("Spectral data", "*.csv *.tsv *.txt *.dat"),
                ("CSV", "*.csv"),
                ("Text", "*.txt *.dat *.tsv"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return
        try:
            self.data = load_spectrum(path, self.import_unit.get())
        except SpectrumImportError as exc:
            messagebox.showerror("Could not import spectrum", str(exc), parent=self.root)
            self.status.set(f"Import failed: {exc}")
            return

        self.data_summary.set(f"{self.data.source_name}\n{self.data.summary}")
        self.status.set(f"Loaded {self.data.source_name}")
        self.update_plot()

    def clear_data(self) -> None:
        self.data = None
        self.data_summary.set("No spectrum loaded")
        self.update_plot()

    def load_absorption(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load an absorption spectrum",
            filetypes=(
                ("Spectral data", "*.csv *.tsv *.txt *.dat"),
                ("CSV", "*.csv"),
                ("Text", "*.txt *.dat *.tsv"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return
        try:
            self.absorption_data = load_spectrum(path, "Auto")
        except SpectrumImportError as exc:
            messagebox.showerror(
                "Could not import absorption spectrum", str(exc), parent=self.root
            )
            self.status.set(f"Absorption import failed: {exc}")
            return

        self.absorption_summary.set(
            f"Imported: {self.absorption_data.source_name}\n"
            f"{self.absorption_data.summary}"
        )
        self.recycling_enabled.set(True)
        self._sync_recycling_states()
        self.status.set(f"Loaded absorption: {self.absorption_data.source_name}")
        self.update_plot()

    def use_reference_absorption(self) -> None:
        self.absorption_data = None
        self.absorption_summary.set(
            "Built-in CsPbBr₃ reference • 47 digitized points"
        )
        self.recycling_enabled.set(True)
        self._sync_recycling_states()
        self.update_plot()

    def reset(self) -> None:
        for key, value in self.DEFAULTS.items():
            self.parameters[key].set(value)
        for key, value in self.RECYCLING_DEFAULTS.items():
            self.recycling_parameters[key].set(value)
        self.line_shape.set("Gaussian")
        self.direction.set("Emission")
        self.y_scale.set("Linear")
        self.show_replicas.set(False)
        self.show_final_spectrum.set(True)
        self.show_intrinsic_spectrum.set(False)
        self.show_data_layer.set(True)
        self.show_zpl_components.set(False)
        self.show_zpl_markers.set(True)
        for key, value in self.TEXT_DEFAULTS.items():
            self.plot_text[key].set(value)
        self.show_legend.set(True)
        self.legend_location.set("upper right")
        self.preserve_axes.set(True)
        self._fit_axes_once = True
        self.primary_zpl_active.set(True)
        self.extra_peaks.clear()
        self._refresh_extra_peaks_tree()
        self.extra_peak_energy.set("2.2600")
        self.extra_peak_amplitude.set("0.500")
        self.extra_peak_S.set(f"{self.DEFAULTS['S']:.3f}")
        self.extra_peak_phonon_meV.set(
            f"{self.DEFAULTS['phonon_meV']:.2f}"
        )
        self.extra_peak_fwhm_meV.set(f"{self.DEFAULTS['fwhm_meV']:.2f}")
        self.extra_peak_broadening_multiplier.set(
            f"{self.DEFAULTS['broadening_multiplier']:.3f}"
        )
        self.extra_peak_active.set(True)
        self.recycling_enabled.set(True)
        self.recycling_mode.set("Beer–Lambert")
        self.show_absorption.set(False)
        self.absorption_data = None
        self.absorption_summary.set(
            "Built-in CsPbBr₃ reference • 47 digitized points"
        )
        self._sync_recycling_states()
        self.update_plot()

    def export_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save figure",
            defaultextension=".png",
            initialfile="huang_rhys_spectrum.png",
            filetypes=(("PNG image", "*.png"), ("PDF document", "*.pdf"), ("SVG vector", "*.svg")),
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
        except OSError as exc:
            messagebox.showerror("Could not save figure", str(exc), parent=self.root)
            return
        self.status.set(f"Saved figure: {Path(path).name}")

    def export_model(self) -> None:
        if (
            self.last_result is None
            or self.last_model_display is None
            or self.last_intrinsic_display is None
            or self.last_branch_displays is None
        ):
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save modeled spectrum",
            defaultextension=".csv",
            initialfile="huang_rhys_model.csv",
            filetypes=(("CSV", "*.csv"),),
        )
        if not path:
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                absorption = (
                    self.last_absorption_profile
                    if self.last_absorption_profile is not None
                    else np.zeros_like(self.last_result.energy_eV)
                )
                reabsorption = (
                    self.last_recycling.reabsorption_probability
                    if self.last_recycling is not None
                    else np.zeros_like(self.last_result.energy_eV)
                )
                branch_headers = [
                    (
                        f"branch_{index + 1}_E{zpl:.4f}eV_"
                        f"S{self.last_result.huang_rhys_factors[index]:.3f}_"
                        f"phonon{1000 * self.last_result.phonon_energies_eV[index]:.2f}meV_"
                        f"FWHM{1000 * self.last_result.base_fwhm_eV[index]:.2f}meV_"
                        f"m{self.last_result.broadening_multipliers[index]:.3f}_"
                        + (
                            "active"
                            if self.last_result.relative_amplitudes[index] > 0
                            else "muted"
                        )
                    )
                    for index, zpl in enumerate(
                        self.last_result.zpl_energies_eV
                    )
                ]
                writer.writerow(
                    [
                        "energy_eV",
                        "intrinsic_emission",
                        "displayed_escaped_emission",
                        "absorption_normalized",
                        "reabsorption_probability",
                    ]
                    + branch_headers
                )
                writer.writerows(
                    zip(
                        self.last_result.energy_eV,
                        self.last_intrinsic_display,
                        self.last_model_display,
                        absorption,
                        reabsorption,
                        *self.last_branch_displays,
                        strict=True,
                    )
                )
        except OSError as exc:
            messagebox.showerror("Could not save model", str(exc), parent=self.root)
            return
        self.status.set(f"Saved model: {Path(path).name}")


def launch() -> None:
    root = tk.Tk()
    HuangRhysExplorer(root)
    root.mainloop()
