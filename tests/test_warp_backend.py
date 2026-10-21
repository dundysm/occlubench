"""MuJoCo Warp backend. Runs on Warp's CPU device if no GPU is present; skipped if not installed."""

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from occlubench.conditions import Condition  # noqa: E402
from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402


def test_warp_physics_and_batch_render_match_cpu_roughly():
    cond = Condition(camera="third_person", severity=1.0, onset=0.0, duration=0.5)
    out = {}
    for backend in ("warp", "cpu"):
        env = OccluEnv(n_envs=2, horizon=20, physics=backend, render=backend, image_size=(32, 32))
        obs = env.reset([0, 1], [cond, Condition()])
        pol = ScriptedOracle()
        pol.reset(2)
        for _ in range(6):
            obs, _, _ = env.step(pol.act(obs, env))
        out[backend] = obs
        env.close()
    assert np.abs(out["warp"]["proprio"] - out["cpu"]["proprio"]).max() < 0.01
    for cam in ("third_person", "wrist"):
        assert out["warp"]["rgb"][cam].shape == (2, 32, 32, 3)
        assert out["warp"]["rgb"][cam].std() > 1.0
