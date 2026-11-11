"""Where things live. One place, overridable by environment, never hard-coded twice.

The journal repo used to sit beside the frozen workshop tree as
``<parent>/forcesketch`` and every script derived that path from its own
location. After the repository moved, that derivation resolved to the journal
repo itself and every import from the frozen package failed. The roots below are
therefore read from the environment, with defaults that match the Hekstra-lab
Cannon layout (code in home, software and raw inputs on holylabs, regenerable
job output on netscratch):

    FORCESKETCH_FROZEN   frozen workshop tree (src/forcesketch, configs, caches)
    FORCESKETCH_DATA     raw inputs: data/{3bpa,rmd17}, models/zenodo/...
    FORCESKETCH_CACHES   regenerated head-force caches and scans (purgeable)
    FORCESKETCH_SCRATCH  everything else bulky: downloads, training runs, logs

``FROZEN / "data"`` and ``FROZEN / "models"`` are symlinks into ``DATA`` so that
the frozen scripts' relative paths keep working unchanged.
"""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

_HOME = pathlib.Path("/n/home05/ianpoon")
_HOLY = pathlib.Path("/n/holylabs/hekstra_lab/Everyone/ianpoon")
_NETSCRATCH = pathlib.Path("/n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch")


def _env_path(var: str, default: pathlib.Path) -> pathlib.Path:
    v = os.environ.get(var)
    return pathlib.Path(v).expanduser().resolve() if v else default


FROZEN = _env_path("FORCESKETCH_FROZEN", _HOME / "forcesketch-workshop/forcesketch")
DATA = _env_path("FORCESKETCH_DATA", _HOLY / "forcesketch-data")
SCRATCH = _env_path("FORCESKETCH_SCRATCH", _NETSCRATCH)
CACHES = _env_path("FORCESKETCH_CACHES", SCRATCH / "caches")

MODELS = DATA / "models/zenodo"
FOUNDATION = MODELS / "foundation"
FROZEN_CACHES = FROZEN / "results/processed"   # the six git-tracked workshop caches
NAIVE_CACHES = CACHES / "naive"
LOGS = SCRATCH / "logs"

PY_GPU = _HOLY / "envs/fs-gpu/bin/python"
PY_CPU = _HOLY / "envs/fs-cpu/bin/python"


def cache_dirs() -> list[pathlib.Path]:
    """Every directory that may hold a ``head_forces_*.pt`` cache, journal first."""
    return [CACHES, NAIVE_CACHES, FROZEN_CACHES]


def describe() -> str:
    return "\n".join(f"{k:<14}{v}" for k, v in {
        "ROOT": ROOT, "FROZEN": FROZEN, "DATA": DATA, "CACHES": CACHES,
        "SCRATCH": SCRATCH, "MODELS": MODELS}.items())


if __name__ == "__main__":
    print(describe())
