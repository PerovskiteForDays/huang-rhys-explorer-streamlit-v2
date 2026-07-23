"""Build the desktop application for the current operating system."""

from __future__ import annotations

from pathlib import Path
import platform
import sys


def main() -> None:
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed. Run: "
            "python -m pip install -r requirements-build.txt"
        ) from exc

    project_dir = Path(__file__).resolve().parent
    args = [
        str(project_dir / "app.py"),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        "HuangRhysExplorer",
        "--distpath",
        str(project_dir / "dist"),
        "--workpath",
        str(project_dir / "build"),
        "--specpath",
        str(project_dir),
    ]
    if platform.system() == "Windows":
        args.append("--onefile")

    PyInstaller.__main__.run(args)
    print(f"Build complete: {project_dir / 'dist'}")


if __name__ == "__main__":
    main()
