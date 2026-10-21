import mujoco
import numpy as np

from occlubench.env import OccluEnv
from occlubench.scene import SceneIds, load_model


def test_scene_loads_with_both_cameras_and_mocap_occluder():
    m = load_model()
    ids = SceneIds.from_model(m)
    assert set(ids.cams) == {"third_person", "wrist"}
    assert ids.occluder_mocap >= 0
    # Occluder must not collide with anything.
    assert m.geom_contype[ids.occluder_geom] == 0 and m.geom_conaffinity[ids.occluder_geom] == 0
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)


def test_reset_is_seeded_and_deterministic():
    env = OccluEnv(n_envs=3, render="none")
    env.reset([1, 2, 1])
    p = env.cube_pos()
    assert np.allclose(p[0], p[2]) and not np.allclose(p[0], p[1])
    env.close()


def test_render_shapes():
    env = OccluEnv(n_envs=2, image_size=(32, 40))
    obs = env.reset([0, 1])
    for cam in ("third_person", "wrist"):
        assert obs["rgb"][cam].shape == (2, 32, 40, 3)
        assert obs["rgb"][cam].dtype == np.uint8
        assert obs["rgb"][cam].std() > 1.0  # not a blank frame
    assert obs["proprio"].shape == (2, 5)
    env.close()
