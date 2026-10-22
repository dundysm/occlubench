"""On-device IK kernels (occlubench.warp_ik) vs. the NumPy reference in occlubench.robots."""

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402


def ik_parity(robot: str, n: int = 6, steps: int = 25, seed: int = 0):
    """Drive a Warp env with the oracle; at each step compare device ctrl with NumPy ctrl."""
    env = OccluEnv(n_envs=n, horizon=steps, physics="warp", render="none", robot=robot)
    assert env._ik is not None
    obs = env.reset(list(range(n)))
    pol = ScriptedOracle()
    pol.reset(n)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(steps):
        a = pol.act(obs, env)
        a = a + rng.normal(0, 0.01, a.shape)  # off the oracle's path too
        st = env.state
        ref = env.robot.ctrl(a, st["qpos"], st["tcp_pos"], st["tcp_mat"], st["jac"])
        env._ik.write(a)
        dev = env.backend.d.ctrl.numpy().astype(float)
        worst = max(worst, float(np.abs(dev - ref).max()))
        obs, done, _ = env.step(a)
    env.close()
    return worst


def oracle_success(robot: str, device_ik: bool, n: int = 4, horizon: int = 200):
    env = OccluEnv(n_envs=n, horizon=horizon, physics="warp", render="none", robot=robot, device_ik=device_ik)
    obs = env.reset(list(range(n)))
    pol = ScriptedOracle()
    pol.reset(n)
    done = False
    while not done:
        obs, done, info = env.step(pol.act(obs, env))
    s = env.success_step
    env.close()
    return s


def test_gantry_ik_parity():
    assert ik_parity("gantry") < 1e-5


def test_gantry_oracle_device_ik_matches_host_ik():
    dev, host = oracle_success("gantry", True), oracle_success("gantry", False)
    assert (dev >= 0).all()
    np.testing.assert_allclose(dev, host, atol=2)


def test_device_ik_requires_warp():
    with pytest.raises(ValueError):
        OccluEnv(n_envs=1, render="none", device_ik=True)
