"""Fetch the Franka Emika Panda model from MuJoCo Menagerie at runtime (not vendored).

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
License: Apache-2.0 (see the LICENSE file in that directory; it is fetched alongside the model).

Resolution order:
1. ``$OCCLUBENCH_MENAGERIE``: an existing Menagerie checkout (directory containing franka_emika_panda/).
2. ``$OCCLUBENCH_CACHE`` or ``~/.cache/occlubench``: a sparse git checkout at ``MENAGERIE_COMMIT``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"  # pinned for reproducibility
PANDA_DIR = "franka_emika_panda"


def cache_root() -> Path:
    return Path(os.environ.get("OCCLUBENCH_CACHE", Path.home() / ".cache" / "occlubench"))


def panda_dir(fetch: bool = True) -> Path:
    env = os.environ.get("OCCLUBENCH_MENAGERIE")
    if env:
        d = Path(env) / PANDA_DIR
        if not (d / "panda.xml").exists():
            raise FileNotFoundError(f"$OCCLUBENCH_MENAGERIE set but {d / 'panda.xml'} is missing")
        return d
    root = cache_root() / f"mujoco_menagerie-{MENAGERIE_COMMIT[:12]}"
    d = root / PANDA_DIR
    if (d / "panda.xml").exists():
        return d
    if not fetch:
        raise FileNotFoundError(f"Panda model not cached at {d}; run `python -m occlubench.menagerie`")
    _sparse_checkout(root)
    return d


def _sparse_checkout(root: Path):
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch MuJoCo Menagerie (or set $OCCLUBENCH_MENAGERIE)")
    tmp = root.with_name(root.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp, check=True, capture_output=True)

    tmp.mkdir()
    git("init", "-q")
    git("remote", "add", "origin", MENAGERIE_URL)
    git("sparse-checkout", "set", PANDA_DIR)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", MENAGERIE_COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    shutil.rmtree(root, ignore_errors=True)
    tmp.rename(root)


if __name__ == "__main__":
    print(panda_dir())
