import math

import numpy as np
import pytest

from occlubench.conditions import Condition, condition_grid
from occlubench.env import OccluEnv
from occlubench.occluder import occluder_pose, schedule_phase, set_occluder


@pytest.mark.parametrize("camera", ["third_person", "wrist"])
def test_occluder_on_ray_hides_object(camera):
    env = OccluEnv(n_envs=1, render="none")
    env.reset([0])
    d = env.datas[0]
    assert env.meter.coverage(d, camera) == pytest.approx(0.0)
    pos, quat = occluder_pose(d, env.ids, camera, 0.0)
    set_occluder(d, env.ids, pos, quat)
    assert env.meter.coverage(d, camera) > 0.95
    env.close()


@pytest.mark.parametrize("severity", [0.3, 0.6])
def test_calibration_hits_requested_severity(severity):
    env = OccluEnv(n_envs=1, render="none", coverage_res=96)
    env.reset([0])
    lat, achieved = env.meter.calibrate(env.datas[0], "third_person", severity)
    assert lat > 0
    assert abs(achieved - severity) < 0.12
    env.close()


def test_schedule_moves_in_holds_and_leaves():
    seq = [schedule_phase(t, onset_step=10, hold_steps=5, slide_steps=4) for t in range(30)]
    assert all(math.isnan(x) for x in seq[:10])
    assert seq[10] > seq[12] > 0  # sliding in
    assert seq[14:19] == [0.0] * 5  # holding over the object
    assert 0 < seq[19] < seq[21]  # sliding out
    assert all(math.isnan(x) for x in seq[23:])


def test_occluder_activates_at_onset_in_env():
    cond = Condition(camera="third_person", severity=1.0, onset=0.25, duration=0.25)
    env = OccluEnv(n_envs=1, horizon=40, render="none")
    obs = env.reset([0], [cond])
    active = []
    for _ in range(40):
        obs, done, info = env.step(np.tile([0.0, 0.0, 0.6, -1.0], (1, 1)))
        active.append(bool(info["occluder_active"][0]))
    assert not any(active[:9]) and any(active[10:20])
    assert env.coverage[0] > 0.9
    env.close()


def test_condition_grid_and_validation():
    g = condition_grid()
    assert len(g) == 8 and len({c.name for c in g}) == 8
    with pytest.raises(ValueError):
        Condition(camera="top", severity=0.5)
    with pytest.raises(ValueError):
        Condition(camera="wrist", severity=0.0)
