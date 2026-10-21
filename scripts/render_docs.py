"""Render clean vs occluded frames for docs/images (CPU renderer)."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from occlubench.conditions import Condition  # noqa: E402
from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")


def frames(cond, t_snap, size=(240, 320)):
    env = OccluEnv(n_envs=1, horizon=200, image_size=size)
    obs = env.reset([3], [cond])
    pol = ScriptedOracle()
    pol.reset(1)
    for _ in range(t_snap):
        obs, _, _ = env.step(pol.act(obs, env))
    cov = env.coverage[0]
    env.close()
    return obs["rgb"]["third_person"][0], obs["rgb"]["wrist"][0], cov


def main():
    os.makedirs(OUT, exist_ok=True)
    t = 40  # inside the hold window for onset 0.15, duration 0.25
    rows = []
    for name, cond in [
        ("clean", Condition()),
        ("third_person_sev0.5", Condition("third_person", 0.5, 0.15, 0.25)),
        ("third_person_sev1.0", Condition("third_person", 1.0, 0.15, 0.25)),
        ("wrist_sev1.0", Condition("wrist", 1.0, 0.15, 0.25)),
    ]:
        tp, wr, cov = frames(cond, t)
        print(name, "coverage", cov)
        rows.append(np.concatenate([tp, np.full((tp.shape[0], 6, 3), 255, np.uint8), wr], 1))
        Image.fromarray(rows[-1]).save(os.path.join(OUT, f"{name}.png"))
    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.concatenate(sum(([r, gap] for r in rows), [])[:-1], 0)
    Image.fromarray(grid).save(os.path.join(OUT, "grid.png"))


if __name__ == "__main__":
    main()
