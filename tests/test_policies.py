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


def test_eef_state_axis_angle():
    from occlubench.env import rotmat_to_axisangle
    from occlubench.robots import euler_to_mat

    rpy = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.3], [0.2, -0.4, 1.1]])
    aa = rotmat_to_axisangle(euler_to_mat(rpy))
    np.testing.assert_allclose(aa[0], 0.0, atol=1e-9)
    assert np.all(np.linalg.norm(aa, axis=1) <= np.pi + 1e-9)
    # round trip through MuJoCo's axis-angle -> quaternion -> matrix
    import mujoco

    for v, m in zip(aa, euler_to_mat(rpy)):
        q, r = np.zeros(4), np.zeros(9)
        ang = np.linalg.norm(v)
        mujoco.mju_axisAngle2Quat(q, v / ang if ang > 0 else np.array([1.0, 0, 0]), ang)
        mujoco.mju_quat2Mat(r, q)
        np.testing.assert_allclose(r.reshape(3, 3), m, atol=1e-9)
    env = OccluEnv(n_envs=2, horizon=5, render="none")
    obs = env.reset([0, 1])
    assert obs["eef_state"].shape == (2, 8)
    np.testing.assert_allclose(obs["eef_state"][:, :3], obs["proprio"][:, :3])
    env.close()


def test_openvla_oft_backend_calls_match_openvla_oft(monkeypatch):
    """Fake ``experiments.robot.openvla_utils`` with the signatures of moojink/openvla-oft main."""
    import sys
    import types

    seen = {}

    class FakeVLA:
        llm_dim = 4096
        norm_stats = {"libero_spatial_no_noops": {"action": {}, "proprio": {}}}

    def get_vla(cfg):
        seen["cfg"] = cfg
        return FakeVLA()

    def get_processor(cfg):
        return "proc"

    def get_action_head(cfg, llm_dim):
        assert cfg.use_l1_regression and not cfg.use_diffusion and llm_dim == 4096
        return "head"

    def get_proprio_projector(cfg, llm_dim, proprio_dim):
        assert proprio_dim == 8
        return "proj"

    def resize_image_for_policy(img, resize_size):
        assert isinstance(resize_size, int)
        return np.zeros((resize_size, resize_size, 3), np.uint8)

    def get_vla_action(
        cfg,
        vla,
        processor,
        obs,
        task_label,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        use_film=False,
    ):
        assert set(obs) == {"full_image", "wrist_image", "state"} and obs["state"].shape == (8,)
        assert obs["full_image"].shape == (224, 224, 3) and action_head == "head" and proprio_projector == "proj"
        seen["state"] = obs["state"]
        return [np.full(7, 0.5) for _ in range(8)]

    ou = types.ModuleType("experiments.robot.openvla_utils")
    for f in (get_vla, get_processor, get_action_head, get_proprio_projector, resize_image_for_policy, get_vla_action):
        setattr(ou, f.__name__, f)
    robot = types.ModuleType("experiments.robot")
    robot.openvla_utils = ou
    exp = types.ModuleType("experiments")
    exp.robot = robot
    monkeypatch.setitem(sys.modules, "experiments", exp)
    monkeypatch.setitem(sys.modules, "experiments.robot", robot)
    monkeypatch.setitem(sys.modules, "experiments.robot.openvla_utils", ou)

    pol = OpenVLAOFTPolicy(backend="oft", unnorm_key="libero_spatial")
    ch = pol.predict_chunk(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32, 3), np.uint8), np.arange(8.0))
    assert ch.shape == (8, 7)
    assert pol.unnorm_key == "libero_spatial_no_noops"  # run_libero_eval-style fallback
    cfg = seen["cfg"]
    for k in ("model_family", "num_images_in_input", "use_proprio", "center_crop", "num_open_loop_steps"):
        assert hasattr(cfg, k)
    assert cfg.num_images_in_input == 2 and cfg.model_family == "openvla"
    np.testing.assert_array_equal(seen["state"], np.arange(8.0))
