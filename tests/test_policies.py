import numpy as np
import pytest

from occlubench.conditions import Condition
from occlubench.env import OccluEnv
from occlubench.policies import ColorCuePolicy, OpenVLAOFTPolicy, ScriptedOracle, unnormalize_actions
from occlubench.runner import run_paired


def test_oracle_solves_gantry_task():
    env = OccluEnv(n_envs=8, render="none", horizon=170)
    obs = env.reset(list(range(8)))
    pol = ScriptedOracle()
    pol.reset(8)
    done = False
    while not done:
        obs, done, info = env.step(pol.act(obs, env))
    assert (env.success_step >= 0).all()
    env.close()


def test_color_cue_localises_from_pixels():
    env = OccluEnv(n_envs=3, image_size=(64, 64))
    obs = env.reset([0, 1, 2])
    est = ColorCuePolicy()._localise(obs["rgb"]["third_person"], env.camera_calibration("third_person"))
    err = np.linalg.norm(est[:, :2] - env.cube_pos()[:, :2], axis=1)
    assert (err < 0.012).all()
    env.close()


def test_paired_runner_oracle_has_zero_drop_and_records_leakage():
    cond = Condition(camera="third_person", severity=1.0, onset=0.1, duration=0.2)
    recs = run_paired(ScriptedOracle, [cond], seeds=[0, 1], n_envs=2, render="none", horizon=170)
    assert len(recs) == 2
    assert all(r["clean_success"] and r["pert_success"] for r in recs)
    assert all(r["coverage_onset"] > 0.95 for r in recs)
    assert all("coverage_other_max" in r for r in recs)


def test_full_occlusion_at_start_delays_pixel_policy():
    cond = Condition(camera="third_person", severity=1.0, onset=0.0, duration=0.3)
    recs = run_paired(ColorCuePolicy, [cond], seeds=[0, 1], n_envs=2, image_size=(48, 48), horizon=170)
    for r in recs:
        assert r["clean_success"]
        assert (not r["pert_success"]) or r["pert_step"] > r["clean_step"]


def test_unnormalize_actions_bounds():
    a = unnormalize_actions(np.array([[-1.0, 1.0, 0.3]]), [0, -2, 5], [2, 2, 9], mask=[True, True, False])
    np.testing.assert_allclose(a, [[0.0, 2.0, 0.3]])


def test_openvla_oft_mock_chunking_and_actions():
    env = OccluEnv(n_envs=2, horizon=20, image_size=(32, 32))
    obs = env.reset([0, 1])
    pol = OpenVLAOFTPolicy(backend="mock", chunk=8, open_loop_steps=4)
    pol.reset(2)
    calls = []
    real = pol.predict_chunk

    def spy(*a):
        calls.append(1)
        return real(*a)

    pol.predict_chunk = spy
    for _ in range(9):
        tcp = obs["proprio"][:, :3]
        act = pol.act(obs, env)
        assert act.shape == (2, env.robot.action_dim)
        assert np.abs(act[:, :3] - tcp).max() <= 0.01 + 1e-9  # mock deltas are within q01/q99
        assert np.isin(np.sign(act[:, -1]), [-1, 0, 1]).all() and np.abs(act[:, -1]).max() <= 1
        obs, _, _ = env.step(act)
    assert len(calls) == 2 * 3  # 9 steps, 4 open-loop steps per chunk -> 3 queries per env
    env.close()


def test_openvla_oft_gripper_convention():
    pol = OpenVLAOFTPolicy(backend="mock")
    from occlubench.robots import GantryRobot

    d = np.zeros((2, 7))
    d[:, 6] = [1.0, 0.0]  # model: 1 = open, 0 = closed
    a = pol.to_env_action(d, np.zeros((2, 3)), np.zeros(2), GantryRobot())
    np.testing.assert_allclose(a[:, -1], [-1.0, 1.0])  # OccluBench: -1 open, +1 close


def test_openvla_oft_rejects_unknown_backend():
    with pytest.raises(ValueError):
        OpenVLAOFTPolicy(backend="nope")
