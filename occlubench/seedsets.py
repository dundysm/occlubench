"""Frozen evaluation seed sets, one file per grid cell.

Benchmark v1 is the 2 cameras x 2 severities x 2 onsets x 2 anchors grid below (16 cells,
duration 0.25 of the horizon). Each cell has its own frozen list of ``SEEDS_PER_CELL`` seeds;
every seed is one *pair* (a clean rollout and a perturbed rollout from the same initial state).
The files ship in ``occlubench/seedsets_v1/`` with a ``manifest.json`` holding each file's
SHA-256, so a results file can prove which seeds it used. Never regenerate v1; add v2 instead.

Seeds are drawn without replacement from ``[0, DEMO_SEED_BASE)``, so they never overlap the
demonstration seeds used for fine-tuning (``occlubench.demos``).
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np

from occlubench.conditions import Condition, condition_grid
from occlubench.demos import DEMO_SEED_BASE

VERSION = "v1"
SEEDS_PER_CELL = 1024
DURATION = 0.25


def cells_v1() -> list[Condition]:
    return [
        c
        for anchor in ("camera", "world")
        for c in condition_grid(severities=(0.5, 1.0), onsets=(0.15, 0.45), duration=DURATION, anchor=anchor)
    ]


def _dir() -> Path:
    return Path(str(resources.files("occlubench") / f"seedsets_{VERSION}"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_v1(out_dir: str | Path | None = None, master_seed: int = 20261022) -> Path:
    """Generate the v1 files. Run once (already done; kept for provenance)."""
    out = Path(out_dir) if out_dir else _dir()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(master_seed)
    manifest = dict(version=VERSION, seeds_per_cell=SEEDS_PER_CELL, master_seed=master_seed, cells={})
    for c in cells_v1():
        seeds = np.sort(rng.choice(DEMO_SEED_BASE, SEEDS_PER_CELL, replace=False)).tolist()
        f = out / f"{c.name}.json"
        f.write_text(json.dumps(dict(cell=c.name, condition=c.to_dict(), seeds=seeds)) + "\n")
        manifest["cells"][c.name] = dict(file=f.name, sha256=_sha(f), condition=c.to_dict())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def manifest() -> dict:
    return json.loads((_dir() / "manifest.json").read_text())


def load(cell: str, verify: bool = True) -> tuple[Condition, list[int]]:
    """(condition, frozen seeds) for one cell name, e.g. ``wrist_sev1.00_on0.15_dur0.25``."""
    entry = manifest()["cells"][cell]
    f = _dir() / entry["file"]
    if verify and _sha(f) != entry["sha256"]:
        raise ValueError(f"seed file {f} does not match the manifest hash")
    d = json.loads(f.read_text())
    return Condition(**d["condition"]), d["seeds"]


def cell_names() -> list[str]:
    return list(manifest()["cells"])
