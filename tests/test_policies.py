import numpy as np

from occlubench.conditions import Condition
from occlubench.env import OccluEnv
from occlubench.policies import ColorCuePolicy, OpenVLAOFTPolicy, ScriptedOracle
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


def test_openvla_stub_raises():
    p = OpenVLAOFTPolicy()
    p.reset(1)
    try:
        p.act({}, None)
    except NotImplementedError:
        return
    raise AssertionError("stub should raise")
