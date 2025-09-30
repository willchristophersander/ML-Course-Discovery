#!/usr/bin/env python3
"""Bootstrap a virtual environment and install project dependencies."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
STRAY_NAMES = {
    "PATH",
    "is",
    "if",
    "already",
    "on",
    "only",
    "python3.13",
    "you",
    "haven't",
    "haven’t",
    "#",
}


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def cleanup_stray_artifacts() -> None:
    for name in STRAY_NAMES:
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()


def ensure_virtualenv() -> Path:
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print(f"Virtual environment already exists at {VENV_DIR}")
    return VENV_DIR


def venv_bin(executable: str) -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / executable
    return VENV_DIR / "bin" / executable


def install_requirements() -> None:
    if not REQUIREMENTS.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQUIREMENTS}")
    pip_executable = venv_bin("pip")
    print("Installing dependencies from requirements.txt")
    run([str(pip_executable), "install", "-r", str(REQUIREMENTS)])


def bootstrap_playwright() -> None:
    pip_executable = venv_bin("pip")
    playwright_executable = venv_bin("playwright")

    try:
        run([str(pip_executable), "install", "playwright"])
    except subprocess.CalledProcessError as exc:
        print(f"Failed to install playwright: {exc}")
        return

    try:
        run([str(playwright_executable), "install"])
    except subprocess.CalledProcessError as exc:
        print(f"Failed to download playwright browsers: {exc}")


def main() -> None:
    cleanup_stray_artifacts()
    ensure_virtualenv()
    install_requirements()
    bootstrap_playwright()
    cleanup_stray_artifacts()
    print("Environment setup complete. Activate it with 'source .venv/bin/activate'.")


if __name__ == "__main__":
    main()
