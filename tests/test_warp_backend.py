"""MuJoCo Warp backend. Runs on Warp's CPU device if no GPU is present; skipped if not installed."""

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from occlubench.conditions import Condition  # noqa: E402
from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402


def _run(robot, backend):
    env = OccluEnv(n_envs=2, horizon=20, physics=backend, render=backend, image_size=(32, 32), robot=robot)
    obs = env.reset([0, 1], [Condition("third_person", 1.0, 0.0, 0.5), Condition()])
    pol = ScriptedOracle()
    pol.reset(2)
    for _ in range(6):
        obs, _, _ = env.step(pol.act(obs, env))
    cov = env.coverage_stats()["coverage_onset"]
    env.close()
    return obs, cov


@pytest.mark.parametrize("robot", ["gantry", pytest.param("panda", marks=pytest.mark.panda)])
def test_warp_matches_cpu(robot):
    w, cw = _run(robot, "warp")
    c, cc = _run(robot, "cpu")
    assert np.abs(w["proprio"] - c["proprio"]).max() < 1e-3
    assert cw[0] == pytest.approx(cc[0], abs=1e-3)
    for cam in ("third_person", "wrist"):
        assert w["rgb"][cam].shape == (2, 32, 32, 3)
        assert w["rgb"][cam].std() > 1.0
