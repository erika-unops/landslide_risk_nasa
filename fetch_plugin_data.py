"""
fetch_plugin_data.py

Step 2 of 4. Runs each plugin's script (in that plugin's own venv) to
download its raw data, passing config.yaml straight through as --config.

Requires setup_environment.py to have been run first.

Usage:
    python fetch_plugin_data.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.pipeline_utils import (
    load_master_config,
    load_plugin_registry,
    plugin_repo_dir,
    plugin_venv_dir,
    run,
    venv_python,
)


def run_plugin_script(plugin: dict, config_path: Path) -> None:
    plugin_dir = plugin_repo_dir(plugin)
    venv_dir = plugin_venv_dir(plugin)
    python_exe = venv_python(venv_dir)

    if not python_exe.exists():
        raise FileNotFoundError(
            f"No venv found for plugin '{plugin['name']}' ({venv_dir}). "
            "Run setup_environment.py first."
        )

    script_path = plugin_dir / plugin["script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Plugin script not found: {script_path}")

    # Run with config.yaml's own directory as cwd so relative paths in
    # config.yaml (output_dir, tags_file, ...) resolve consistently regardless
    # of where fetch_plugin_data.py itself was invoked from.
    run(
        [str(python_exe), str(script_path), "--config", str(config_path.resolve())],
        cwd=str(config_path.resolve().parent),
    )


def main(config_path: Path) -> None:
    load_master_config(config_path)  # fail fast if config.yaml is missing/invalid

    for plugin in load_plugin_registry():
        print(f"\n[fetch] {plugin['name']}")
        run_plugin_script(plugin, config_path)

    print(f"\nRaw data fetched. Next: python feature_engineering.py --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to top-level config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
