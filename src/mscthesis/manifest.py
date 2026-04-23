from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .log import log_call


@log_call()
def dump_manifest(
    target_path: Path,
    command_name: str,
    sample_id: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    metadata: dict[str, Any],
    tool_version: str,
) -> None:
    """
    Dump a manifest JSON file summarizing the command execution and output contents.
    Args:
        target_path (Path): Path where the manifest file will be saved.
        command_name (str): Name of the command executed.
        sample_id (str): Identifier for the generated sample.
        inputs (dict[str, Any]): Dictionary of input file paths used.
        outputs (dict[str, Any]): Dictionary of output file paths generated.
        metadata (dict[str, Any]): Additional metadata to include in the manifest.
        tool_version (str): Version of the tool used.
    """
    manifest: dict[str, Any] = {}
    manifest["time_stamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    manifest["command"] = command_name
    manifest["sample_id"] = sample_id
    manifest["inputs"] = inputs
    manifest["outputs"] = outputs
    manifest["meta"] = metadata
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    manifest["git_commit"] = git_commit
    manifest["tool"] = f"mscthesis version {tool_version}"

    with open(target_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, default=str)

    return


@log_call()
def fetch_from_manifest(
    manifest_path: Path, *keys: str, subdict: str = "meta"
) -> dict[str, Any]:
    """
    Fetch and return the contents of a manifest JSON file.
    Args:
        manifest_path (Path): Path to the manifest file to read.
        *keys (str): The keys to fetch from the manifest.
        subdict (str): The subdictionary to fetch keys from (default is "meta").
    Returns:
        dict[str, Any]: The contents of the manifest file as a dictionary.
    """
    quantities: dict[str, Any] = {}
    with open(manifest_path) as manifest_file:
        manifest = json.load(manifest_file)
        for key in keys:
            if key in manifest[subdict]:
                quantities[key] = manifest[subdict][key]
            else:
                raise KeyError(f"Key '{key}' not found in manifest {subdict}.")
    return quantities
