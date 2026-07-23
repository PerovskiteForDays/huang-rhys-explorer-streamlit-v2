# Streamlit deployment checklist

## Repository layout

Keep these items together at the root of the GitHub repository:

```text
streamlit_app.py
requirements.txt
README.md
.streamlit/config.toml
huang_rhys/
```

Do not upload an extra outer folder around them. Streamlit must be able to find
`streamlit_app.py` at the main file path you select.

## Test locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

1. Push the complete project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select the repository and branch.
4. Enter `streamlit_app.py` as the main file path.
5. In Advanced settings, choose Python 3.12 when available.
6. Click **Deploy**.

No secrets or system `packages.txt` file are required.

## Common deployment errors

- **Main file not found:** `streamlit_app.py` is probably nested inside another
  folder. Move the project contents to the repository root or update the main
  file path.
- **Module not found:** confirm `requirements.txt` is committed beside
  `streamlit_app.py`, then reboot the app from Community Cloud.
- **Import file rejected:** use a text CSV/TSV/TXT/DAT file with energy or
  wavelength in the first numeric column and intensity in the second.

Official references: [app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [deployment steps](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
