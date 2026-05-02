from __future__ import annotations

import argparse
import os
import subprocess

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_voxels, save_surface_mesh
from ....core.meshing.triangulation import triangulate_voxels
from ....ids import sample_id_from_int
from ....paths import ProjectPaths

""" 
RETRY LOGIC:

params = [
    {"a": 1, "b": 2},
    {"a": 3, "b": 4},
]

for param in params:
    try: 
        result = some_function(**param)
        break # stop looping further if successful
    except SomeError:
        continue # try the next set of parameters
else: # only executed if the loop completes without a break
    raise RuntimeError("All attempts failed.")
"""


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    index = load_dataframe(paths.index.require())
    selected_ids = index[index["selected"]]["sample_id"].values.tolist()
    selected_ids = [
        sample_id_from_int(int(sample_id), config.behavior.sample_id_digits)
        for sample_id in selected_ids
    ]
    print(selected_ids, type(selected_ids[0]))

    # prepare tasks as the set of sample ids where .brep does not exist but .voxels does
    # for each sample, repeatedly attempt triangulation with increasing elements_per_cell until success or max_attempts is reached
    # as the config, save the parameters that succeeded, not the baseline
    # print a final report of status and the list of failed samples

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "triangulate-selected", help="Triangulate the selected samples."
    )
    parser.set_defaults(cmd=_cmd)
    return
