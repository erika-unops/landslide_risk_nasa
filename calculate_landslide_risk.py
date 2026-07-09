from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
PLUGINS_PATH = ROOT / "plugins.yaml"


def run(cmd, cwd=None, env=None):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_venv(pyvenv_dir: Path) -> Path:
    if not pyvenv_dir.exists():
        run([sys.executable, "-m", "venv", str(pyvenv_dir)])

    if os.name == "nt":
        return pyvenv_dir / "Scripts" / "python.exe"
    return pyvenv_dir / "bin" / "python"


def load_plugin_registry() -> list[dict]:
    with open(PLUGINS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("plugins", [])


def clone_or_update_plugin(plugin: dict, workspace_dir: Path) -> Path:
    name = plugin["name"]
    repo = plugin["repo"]
    target_dir = workspace_dir / name

    if not target_dir.exists():
        run(["git", "clone", repo, str(target_dir)])
    else:
        run(["git", "-C", str(target_dir), "pull", "--ff-only"])

    return target_dir


def install_requirements(python_exe: Path, plugin_dir: Path) -> None:
    req = plugin_dir / "requirements.txt"
    if req.exists():
        run([str(python_exe), "-m", "pip", "install", "-r", str(req)])
    else:
        print(f"Skipping dependency install for {plugin_dir} because requirements.txt is missing.")


def run_plugin_script(python_exe: Path, plugin_dir: Path, script_name: str, config_path: Path) -> None:
    script_path = plugin_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Plugin script not found: {script_path}")

    run([str(python_exe), str(script_path), "--config", str(config_path)])


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Master config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        master_config = yaml.safe_load(f) or {}

    workspace_dir = ROOT.parent
    venv_dir = ROOT / ".venv"
    python_exe = ensure_venv(venv_dir)

    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

    plugins = load_plugin_registry()

    for plugin in plugins:
        plugin_dir = clone_or_update_plugin(plugin, workspace_dir)
        install_requirements(python_exe, plugin_dir)

        plugin_config = master_config.get("plugins", {}).get(plugin["config_key"], {})
        plugin_config_path = ROOT / f"{plugin['name']}_plugin_config.yaml"

        with open(plugin_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(plugin_config, f, sort_keys=False)

        run_plugin_script(
            python_exe=python_exe,
            plugin_dir=plugin_dir,
            script_name=plugin["script"],
            config_path=plugin_config_path,
        )

    run([str(python_exe), str(ROOT / "feature_engineering.py"), "--config", str(CONFIG_PATH)])
    run([str(python_exe), str(ROOT / "stanley_kirschbaum_heuristic_logic.py"), "--config", str(CONFIG_PATH)])


if __name__ == "__main__":
    main()