from __future__ import annotations

from importlib.metadata import version

from cde.lib.eval.mteb.benchmarks import (
    MTEB_MAIN_EN,
    MTEB_RETRIEVAL_LAW,
    MTEB_RETRIEVAL_WITH_INSTRUCTIONS,
)
from cde.lib.eval.mteb.evaluation import *
from cde.lib.eval.mteb.overview import TASKS_REGISTRY, get_tasks

__version__ = "1.10.3"  # fetch version from install metadata


__all__ = [
    "MTEB_MAIN_EN",
    "MTEB_RETRIEVAL_LAW",
    "MTEB_RETRIEVAL_WITH_INSTRUCTIONS",
    "TASKS_REGISTRY",
    "get_tasks",
]
