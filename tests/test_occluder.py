import math

import numpy as np
import pytest

from occlubench.conditions import Condition, condition_grid
from occlubench.env import OccluEnv
from occlubench.occluder import schedule_phase


def test_schedule_moves_in_holds_and_leaves():
    seq = [schedule_phase(t, onset_step=10, hold_steps=5, slide_steps=4) for t in range(30)]
    assert all(math.isnan(x) for x in seq[:10])
    assert seq[10] > seq[12] > 0
    assert seq[14:19] == [0.0] * 5
    assert 0 < seq[19] < seq[21]
    assert all(math.isnan(x) for x in seq[23:])


def test_schedule_is_vectorised():
    out = schedule_phase(np.array([0, 5, 9]), np.array([0, 0, 0]), np.array([4, 4, 4]), 2)
    assert out[0] == pytest.approx(0.5) and out[1] == 0.0 and math.isnan(out[2])


@pytest.mark.parametrize("camera", ["third_person", "wrist"])
@pytest.mark.parametrize("severity", [0.3, 0.6, 1.0])
def test_calibration_hits_requested_severity(camera, severity):
    env = OccluEnv(n_envs=3, render="none", horizon=40)
    env.reset([0, 1, 2], [Condition(camera, severity, 0.0, 0.5)] * 3)
    env.step(np.tile([0.0, 0.0, 0.6, -1.0], (3, 1)))
    assert np.all(np.abs(env.coverage_onset - severity) < 0.05)
    env.close()


def test_occluder_activates_at_onset_and_world_anchor_freezes_pose():
    conds = [Condition("wrist", 1.0, 0.25, 0.25, "camera"), Condition("wrist", 1.0, 0.25, 0.25, "world")]
    env = OccluEnv(n_envs=2, horizon=40, render="none")
    env.reset([0, 0], conds)
    active, poses = [], []
    for t in range(40):
        # Move the arm during the hold so the wrist camera moves.
        x = 0.1 * math.sin(t / 3)
        env.step(np.tile([x, 0.0, 0.6, -1.0], (2, 1)))
        active.append(env._active.copy())
        poses.append(env.mocap_pos[:, env.ids.occluder_mocap].copy())
    active = np.array(active)
    assert not active[:10].any() and active[10:20].all()
    hold = slice(19, 28)
    world_motion = np.ptp(np.array(poses)[hold, 1], axis=0).max()
    cam_motion = np.ptp(np.array(poses)[hold, 0], axis=0).max()
    assert world_motion < 1e-9 < cam_motion
    env.close()


def test_condition_grid_and_validation():
    g = condition_grid()
    assert len(g) == 8 and len({c.name for c in g}) == 8
    assert condition_grid(anchor="world")[0].name.endswith("_world")
    with pytest.raises(ValueError):
        Condition(camera="top", severity=0.5)
    with pytest.raises(ValueError):
        Condition(camera="wrist", severity=0.0)
    with pytest.raises(ValueError):
        Condition(camera="wrist", severity=0.5, anchor="robot")
