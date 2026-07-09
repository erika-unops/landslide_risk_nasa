"""
pipeline_utils.py
Shared paths/helpers for the multi-script pipeline (setup_environment.py,
fetch_plugin_data.py, feature_engineering.py, stanley_kirschbaum_heuristic_logic.py).

Environment layout:
    <repo_root>/.venv/            master venv — runs feature_engineering.py and
                                   stanley_kirschbaum_heuristic_logic.py
    <repo_root>/.venvs/<plugin>/  one venv per plugin — runs that plugin's script
    <repo_root>/../<plugin>/      cloned plugin repo (sibling of the repo root)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PLUGINS_PATH = ROOT / "plugins.yaml"
REQUIREMENTS_PATH = ROOT / "requirements.txt"

WORKSPACE_DIR = ROOT.parent
MASTER_VENV_DIR = ROOT / ".venv"
PLUGIN_VENVS_DIR = ROOT / ".venvs"


def run(cmd, cwd=None, env=None) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> Path:
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
    return venv_python(venv_dir)


def install_requirements(python_exe: Path, requirements_file: Path) -> None:
    if requirements_file.exists():
        run([str(python_exe), "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print(f"Skipping dependency install: {requirements_file} not found.")


def load_plugin_registry() -> list[dict]:
    with open(PLUGINS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("plugins", [])


def load_master_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Master config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def plugin_repo_dir(plugin: dict) -> Path:
    return WORKSPACE_DIR / plugin["name"]


def plugin_venv_dir(plugin: dict) -> Path:
    return PLUGIN_VENVS_DIR / plugin["name"]
