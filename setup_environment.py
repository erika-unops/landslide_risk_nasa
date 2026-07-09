"""
setup_environment.py

Step 1 of 4. Clones/updates every plugin repo listed in plugins.yaml and
builds the venvs the rest of the pipeline runs in:
    - one venv per plugin (isolates each plugin's own requirements.txt)
    - one master venv for this repo (feature_engineering.py and
      stanley_kirschbaum_heuristic_logic.py)

Usage:
    python setup_environment.py
"""

from __future__ import annotations

from utils.pipeline_utils import (
    MASTER_VENV_DIR,
    REQUIREMENTS_PATH,
    ensure_venv,
    install_requirements,
    load_plugin_registry,
    plugin_repo_dir,
    plugin_venv_dir,
    run,
)


def clone_or_update_plugin(plugin: dict) -> None:
    target_dir = plugin_repo_dir(plugin)
    if not target_dir.exists():
        run(["git", "clone", plugin["repo"], str(target_dir)])
    else:
        run(["git", "-C", str(target_dir), "pull", "--ff-only"])


def build_master_env() -> None:
    print(f"\n[master env] {MASTER_VENV_DIR}")
    python_exe = ensure_venv(MASTER_VENV_DIR)
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    install_requirements(python_exe, REQUIREMENTS_PATH)


def build_plugin_env(plugin: dict) -> None:
    venv_dir = plugin_venv_dir(plugin)
    print(f"\n[plugin env] {plugin['name']} -> {venv_dir}")
    python_exe = ensure_venv(venv_dir)
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    install_requirements(python_exe, plugin_repo_dir(plugin) / "requirements.txt")


def main() -> None:
    build_master_env()

    for plugin in load_plugin_registry():
        clone_or_update_plugin(plugin)
        build_plugin_env(plugin)

    print("\nEnvironment ready. Next: python fetch_plugin_data.py")


if __name__ == "__main__":
    main()
