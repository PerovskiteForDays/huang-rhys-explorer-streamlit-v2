# Huang–Rhys Spectrum Explorer

A clean Streamlit web app and cross-platform desktop app for exploring Gaussian- or Lorentzian-broadened Huang–Rhys vibronic progressions, experimental spectra, self-absorption, and photon recycling.

## What changed

- Deployment-ready Streamlit interface in `streamlit_app.py`; the Tk desktop app remains available through `app.py`.
- Interactive Plotly spectrum with browser zoom, hover readout, modebar image export, and view preservation across ordinary reruns.
- Editable dynamic ZPL table: add/delete branches, mute them, and independently set energy, amplitude, S, phonon energy, base FWHM, and broadening multiplier.
- Browser uploads for emission and absorption CSV/TSV/TXT/DAT files without server-side temporary files.
- Direct downloads for model CSV, parameters JSON, and a self-contained interactive HTML figure.
- Gaussian/Lorentzian line-shape toggle with a common **FWHM** definition.
- Stable replica calculation across the full `S = 0–10` range; no short-list `IndexError`.
- Cross-platform file dialog (macOS, Windows, and Linux).
- Defensive CSV/TSV/TXT/DAT import with header and comment skipping, duplicate-x averaging, finite-value filtering, sorting, and normalization.
- Automatic wavelength (nm) to photon-energy (eV) conversion, with an explicit unit selector when auto-detection is not appropriate.
- Crisp responsive interface with precise type-in values, a high-contrast **LOG Y-SCALE** toggle, emission/absorption controls, and PNG/PDF/SVG or CSV export.
- Stable plot limits by default: zoomed or manually entered X/Y ranges survive slider and model updates.
- A dedicated **Plot** tab with manual limits, one-click fitting, editable title/axis text, custom legend labels, legend visibility, and legend placement.
- Compact plot-layer switches for the final spectrum, intrinsic pre-recycling spectrum, experimental data, ZPL components, individual replicas, ZPL markers, and absorption.
- Per-branch **Active/Muted** controls; muted ZPLs are removed from the summed model and photon-recycling calculation without deleting their settings.
- Installed-font detection with a safe Matplotlib fallback, avoiding missing `SF Pro Text` warnings on macOS systems where that font is not exposed to Matplotlib.
- Beer–Lambert photon recycling with peak optical depth, internal PLQY, and a controlled number of re-emission events.
- A simpler one-slider recycling-strength mode for qualitative exploration.
- Built-in digitized CsPbBr₃ absorption data plus optional measured-absorption import.
- Phenomenological absorption energy-shift and broadening sliders for temperature-dependent comparisons.
- Photon-balance diagnostics: escape fraction, nonradiative loss, one-pass enhancement, mean re-emissions, and spectral peak shift.
- Multiple ZPL branches with independent energy, amplitude, Huang–Rhys factor, phonon energy, base FWHM, and successive-broadening multiplier.
- A unique plot color and optional replica curves for every ZPL phonon progression.
- Successive phonon-replica broadening with a stable linewidth multiplier.
- Interactive navigation toolbar, pan/box zoom, cursor readout, wheel zoom, double-click primary-ZPL placement, and Shift-click ZPL creation.
- Automated model/import tests and GitHub Actions builds for macOS and Windows.

## Run the Streamlit web app locally

Python 3.10–3.13 is supported. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit opens the local web app in your browser. The desktop and web versions share the same numerical model and defensive data importer.

## Deploy on Streamlit Community Cloud

The repository is ready to deploy without restructuring:

1. Put the complete project at the root of a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create a new app and select that repository and branch.
3. Set the main file path to `streamlit_app.py`.
4. In Advanced settings, select Python 3.12 when available, then deploy.

Community Cloud reads the root `requirements.txt` and installs NumPy, SciPy, pandas, Plotly, Streamlit, and Matplotlib. No `packages.txt` or secrets are required for this app. The included `.streamlit/config.toml` supplies the theme and headless server configuration. See Streamlit’s official [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) if you need to change the repository or Python settings.

## Run the desktop app locally

Python 3.10–3.13 is supported.

```bash
git clone <your-repository-url>
cd huang-rhys-explorer

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

You can test the importer immediately with `sample_spectrum.csv`.

### Recommended macOS setup with Miniforge

If a pyenv Python reports `ModuleNotFoundError: No module named '_tkinter'`, a `venv` cannot add Tk afterward because it reuses that same Python build. Create the included Conda environment instead:

```bash
source "$HOME/miniforge3/bin/activate"
conda env create -f environment.yml
conda activate huang-rhys
python -m tkinter
python app.py
```

Close the small Tk test window before running the final command. Do not try `pip install tkinter`; Tk must be provided with the Python distribution.

## Data format

Use a text-based file with x-values in the first numeric column and intensity in the second:

```text
Energy_eV,Intensity_au
2.20,0.15
2.25,0.62
2.30,1.00
```

Supported separators are commas, semicolons, tabs, and whitespace. Header lines, blank lines, and lines beginning with common comment markers are ignored. Files containing wavelength in nm can be converted automatically or by selecting **Wavelength (nm)** before import.

The importer intentionally does not spline-smooth data. That avoids introducing overshoot or artificial negative intensity between experimental points.

Absorption files use the same two-column format. Click **Load absorption** in the **Photon recycling** tab. The bundled CsPbBr₃ reference can be restored at any time with **Use CsPbBr₃ reference**; its source values are also included as `cspbbr3_absorption_reference.csv` for inspection.

## Interactive plot and additional ZPLs

The toolbar below the plot provides Home, Back/Forward, Pan, box Zoom, subplot settings, and image export. The plot itself also supports:

- **Mouse wheel:** zoom the photon-energy axis around the cursor.
- **Hover:** report energy, displayed intensity, and absorption when available.
- **Double-click:** move the primary ZPL to the selected energy.
- **Shift-click:** create an additional ZPL at the selected energy with an initial relative amplitude of 0.5.

Additional branches can also be added, edited, muted, or removed in the **ZPL branches** tab. Every branch has independent energy, relative integrated amplitude, Huang–Rhys factor $S_k$, phonon energy $\hbar\omega_k$, base linewidth $\Gamma_{0,k}$, and successive-width multiplier $m_k$. The primary branch can also be activated or muted there. Muted branches keep their settings but contribute zero intensity to the combined spectrum and photon recycling; at least one branch must remain active. The shared controls are line shape and emission/absorption direction. Each branch and all of its replicas use a unique color. Up to eight additional branches are supported interactively.

The high-contrast **LOG Y-SCALE** button remains beside the plot toolbar, and it stays synchronized with the Linear/Log controls in the **Spectrum** tab.

## Axis limits, graph text, and legends

The plot view is preserved by default. Wheel zoom, toolbar pan/zoom, and manually applied limits therefore remain fixed while sliders and model settings update. Use the **Plot** tab to:

- Enter exact X min, X max, Y min, and Y max values, then click **Apply limits**.
- Click **Fit axes now** to recalculate a clean view once and return to preserved limits.
- Turn off **Preserve zoom and limits when the model updates** when continuous automatic fitting is preferred.
- Edit the title, left and right axis labels, model/data/absorption legend labels, and the branch-label prefix.
- Hide the legend or move it to another standard location.

The **Visible plot layers** panel prevents the common stack of overlapping live curves. The clean default shows the final/model spectrum and ZPL position markers. Intrinsic pre-recycling emission, component progressions, replicas, and absorption are opt-in. Experimental data is shown when loaded and can be muted independently. Layer visibility changes only the graph; ZPL Active/Muted state changes the physical summed model.

A blank title restores the automatic process/line-shape title. A blank data-legend field uses the imported filename. Changing between linear and logarithmic Y scales intentionally recalculates the Y limits so a zero-valued linear lower bound is never reused on a log scale.

## Photon-recycling model

The Beer–Lambert mode converts the normalized absorption trace, $A(E)$, into an energy-dependent reabsorption probability:

$$P_{\mathrm{reabs}}(E)=1-\exp[-\tau_{\mathrm{peak}}A(E)],$$

where $\tau_{\mathrm{peak}}=\alpha L$ is the optical depth at the maximum of the reference absorption. At each generation, escaped photons contribute

$$G_j(E)[1-P_{\mathrm{reabs}}(E)],$$

while the absorbed fraction is re-emitted with the intrinsic Huang–Rhys spectrum according to the internal PLQY. Nonradiative loss and the residual population after the chosen maximum number of re-emissions are tracked explicitly.

In **Simple** mode,

$$P_{\mathrm{reabs}}(E)=sA(E),$$

where $s$ is the single **Recycling strength** slider. The mode assumes unity re-emission yield and uses 25 possible re-emission events. Any remaining truncated population is reported as the residual rather than silently counted as escaped or lost.

The absorption temperature controls are intentionally phenomenological:

- **Absorption shift, ΔE(T):** translates the reference spectrum; positive values move it to higher energy.
- **Extra absorption broadening:** applies an additional Gaussian FWHM.

This avoids imposing a universal linear temperature law across CsPbBr₃ structural phase transitions. The model assumes a homogeneous absorber and the same intrinsic emission spectrum after every recycling event; it is a spectral population model, not a spatial ray-tracing calculation.

## Build a standalone desktop app

Install the build requirements, then run PyInstaller on the same operating system you want to distribute for:

```bash
python -m pip install -r requirements-build.txt
python build_app.py
```

The finished application is placed in `dist/`:

- macOS: `dist/HuangRhysExplorer.app`
- Windows: `dist/HuangRhysExplorer.exe`

PyInstaller builds are operating-system-specific. The included GitHub Actions workflow creates both macOS and Windows artifacts automatically without requiring either computer locally.

## Publish on GitHub

1. Create an empty GitHub repository.
2. Copy this project into it and push the `main` branch.
3. Open the repository's **Actions** tab and run **Test and build desktop apps**.
4. Download the macOS or Windows build artifact from the completed workflow.
5. When ready, create a GitHub Release and attach the downloaded archives.

macOS users may need to right-click the unsigned app and choose **Open** the first time. Public distribution without that warning requires Apple code signing and notarization.

## Model conventions

The nth replica has Poisson weight

$$w_n = e^{-S}\frac{S^n}{n!}.$$

Its center is

$$E_n = E_{\mathrm{ZPL}} - n\hbar\omega$$

for emission and

$$E_n = E_{\mathrm{ZPL}} + n\hbar\omega$$

for absorption. Both line-shape options use the displayed FWHM. The model curve is peak-normalized in the interface before applying **Model scale**, while exported intensity is the displayed value.

For multiple electronic branches, the total intrinsic spectrum is

$$I_{\mathrm{total}}(E)=\sum_k a_k I_k(E),$$

where the primary branch has $a_1=1$ and each additional branch has a user-selected relative amplitude $a_k$. Each $I_k(E)$ uses its own $S_k$, $\hbar\omega_k$, $\Gamma_{0,k}$, and $m_k$. Photon recycling is calculated from this summed intrinsic emission.

The nth phonon replica uses

$$\Gamma_n=\Gamma_0[1+n(m-1)],$$

where $m$ is the **Successive linewidth multiplier**. Thus $m=1$ gives equal FWHM for every replica, while $m=1.5$ produces widths of $\Gamma_0$, $1.5\Gamma_0$, $2\Gamma_0$, $2.5\Gamma_0$, and so on. The linear law avoids the runaway broadening produced by an exponential $m^n$ rule.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover line-shape FWHM, Poisson-tail completeness, independent multi-ZPL progressions, successive linewidth scaling, branch-parameter validation, emission/absorption direction, the former low-`S` replica crash, mixed text formats, wavelength conversion, duplicate points, invalid imports, photon conservation, PLQY losses, absorption shifts, and both recycling modes.
