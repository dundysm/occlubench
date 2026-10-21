"""OccluBench: GPU-native VLA robustness benchmark for moving physical occluders."""

from occlubench.conditions import CAMERAS, Condition, condition_grid
from occlubench.env import OccluEnv
from occlubench.metrics import bootstrap_ci, summarize
from occlubench.runner import run_paired

__all__ = [
    "CAMERAS",
    "Condition",
    "condition_grid",
    "OccluEnv",
    "bootstrap_ci",
    "summarize",
    "run_paired",
]
__version__ = "0.0.1"
