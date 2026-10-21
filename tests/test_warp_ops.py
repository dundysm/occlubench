"""On-device occluder/coverage/success kernels vs. the NumPy reference (same Warp physics)."""

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from occlubench.conditions import Condition  # noqa: E402
from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402

CONDS = [
    Condition("third_person", 0.5, 0.05, 0.3),
    Condition("wrist", 1.0, 0.05, 0.3),
    Condition("third_person", 0.75, 0.1, 0.3, anchor="world"),
    Condition(),
]


def _run(device_ops, horizon=40):
    env = OccluEnv(n_envs=4, horizon=horizon, physics="warp", render="none", device_ops=device_ops)
    obs = env.reset([0, 1, 2, 3], CONDS)
    pol = ScriptedOracle()
    pol.reset(4)
    mocap, active = [], []
    done = False
    while not done:
        obs, done, info = env.step(pol.act(obs, env))
        d = env.backend.d
        mocap.append((d.mocap_pos.numpy().copy(), d.mocap_quat.numpy().copy()))
        active.append(np.asarray(info["occluder_active"]))
    out = dict(stats=env.coverage_stats(), succ=env.success_step, mocap=mocap, active=np.array(active))
    env.close()
    return out


def test_device_ops_match_numpy():
    dev, ref = _run(True), _run(False)
    np.testing.assert_array_equal(dev["active"], ref["active"])
    np.testing.assert_array_equal(dev["succ"], ref["succ"])
    for (pd, qd), (pr, qr) in zip(dev["mocap"], ref["mocap"]):
        np.testing.assert_allclose(pd, pr, atol=2e-3)
        dot = np.abs(np.sum(qd * qr, -1))  # q and -q are the same rotation
        assert dot.min() > 1 - 1e-4
    for k, v in ref["stats"].items():
        np.testing.assert_allclose(dev["stats"][k], v, atol=5e-3, equal_nan=True, err_msg=k)
    assert dev["active"].any()


def test_device_ops_requires_warp():
    with pytest.raises(ValueError):
        OccluEnv(n_envs=1, physics="cpu", render="none", device_ops=True)
