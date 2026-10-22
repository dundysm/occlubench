"""OccluBench: GPU-native VLA robustness benchmark for moving physical occluders.

Exports are imported lazily so that light submodules (for example ``occlubench.rlds``, which
loads TensorFlow) can be used without importing MuJoCo: TensorFlow and MuJoCo crash when both
are loaded into one process on some builds.
"""

import importlib

_EXPORTS = {
    "CAMERAS": "occlubench.conditions",
    "Condition": "occlubench.conditions",
    "condition_grid": "occlubench.conditions",
    "OccluEnv": "occlubench.env",
    "bootstrap_ci": "occlubench.metrics",
    "summarize": "occlubench.metrics",
    "run_paired": "occlubench.runner",
}
__all__ = list(_EXPORTS)
__version__ = "0.0.1"


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'occlubench' has no attribute {name!r}")
