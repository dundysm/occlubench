"""The vectorised geometric coverage must agree with CPU segmentation renders."""

import numpy as np

from occlubench import geometry
from occlubench.conditions import Condition
from occlubench.env import OccluEnv
from occlubench.occluder import CoverageMeter, set_occluder


def _compare(robot):
    n = 6
    env = OccluEnv(n_envs=n, render="none", robot=robot)
    conds = [Condition(("third_person", "wrist")[i % 2], (0.3, 0.6, 1.0)[i % 3], 0.0, 0.5) for i in range(n)]
    env.reset(list(range(n)), conds)
    env._place_occluders()  # calibrates at onset
    cp, cm = env._cam(env.cam_idx)
    pos, rot, _ = geometry.occluder_frame(cp, cm, env.state["cube_pos"], env._hold_lat, np.full(n, 0.45))
    quat = geometry.mat_to_quat(rot)
    geo = np.array([env._coverage(env.cam_idx[i], pos, rot, None)[i] for i in range(n)])
    meter = CoverageMeter(env.m, env.ids, 128, 128)
    seg = []
    for i, d in enumerate(env.backend.datas):
        set_occluder(d, env.ids, pos[i], quat[i])
        seg.append(meter.coverage(d, ("third_person", "wrist")[env.cam_idx[i]]))
    meter.close()
    env.close()
    return geo, np.array(seg)


def test_geometric_coverage_matches_segmentation_gantry():
    geo, seg = _compare("gantry")
    assert np.abs(geo - seg).max() < 0.08


def test_mat_to_quat_roundtrip():
    import mujoco

    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    mats = np.empty((5, 3, 3))
    for i in range(5):
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, q[i])
        mats[i] = m.reshape(3, 3)
    q2 = geometry.mat_to_quat(mats)
    assert np.allclose(np.abs(np.sum(q * q2, 1)), 1.0, atol=1e-6)
