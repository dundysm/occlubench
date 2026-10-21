"""Franka Panda (MuJoCo Menagerie, fetched at runtime). Run separately: `pytest -m panda`.

Kept in a separate pytest process from the gantry tests because the CPU OSMesa renderer
misbehaves when renderers for two different models are created in one process.
"""

import numpy as np
import pytest

from occlubench.conditions import Condition
from occlubench.env import OccluEnv
from occlubench.policies import ScriptedOracle

pytestmark = pytest.mark.panda


def test_panda_model_fetched_and_composed():
    from occlubench.menagerie import panda_dir

    d = panda_dir()
    assert "Apache License" in (d / "LICENSE").read_text()
    env = OccluEnv(n_envs=1, robot="panda", render="none")
    assert env.robot.action_dim == 7 and env.m.nu == 8
    env.close()


def test_panda_ik_reaches_pose():
    env = OccluEnv(n_envs=2, robot="panda", render="none", horizon=60)
    obs = env.reset([0, 1])
    target = np.array([[0.0, -0.1, 0.55], [0.1, 0.1, 0.5]])
    act = env.robot.default_action(target, grip=-1.0, yaw=np.array([0.0, 0.5]))
    for _ in range(60):
        obs, _, _ = env.step(act)
    assert np.linalg.norm(obs["proprio"][:, :3] - target, axis=1).max() < 0.01
    z = env.state["tcp_mat"][:, :, 2]
    assert np.all(z[:, 2] < -0.98)  # gripper pointing down
    env.close()


def test_panda_oracle_solves_task():
    n = 16
    env = OccluEnv(n_envs=n, robot="panda", render="none", horizon=200)
    obs = env.reset(list(range(n)))
    pol = ScriptedOracle()
    pol.reset(n)
    done = False
    while not done:
        obs, done, _ = env.step(pol.act(obs, env))
    assert (env.success_step >= 0).mean() >= 0.9
    env.close()


def test_panda_geometry_matches_segmentation():
    from test_geometry import _compare

    geo, seg = _compare("panda")
    assert np.abs(geo - seg).max() < 0.08


def test_panda_render_and_occluder():
    env = OccluEnv(n_envs=2, robot="panda", image_size=(48, 48), horizon=20)
    obs = env.reset([0, 1], [Condition("wrist", 1.0, 0.0, 0.5), Condition()])
    for _ in range(3):
        obs, _, _ = env.step(env.robot.default_action(obs["proprio"][:, :3]))
    assert obs["rgb"]["wrist"].shape == (2, 48, 48, 3) and obs["proprio"].shape == (2, 12)
    assert env.coverage_onset[0] > 0.95
    env.close()
