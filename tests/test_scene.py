import numpy as np

from occlubench.env import OccluEnv
from occlubench.robots import make_robot
from occlubench.scene import SceneIds, build_model


def test_gantry_scene_has_cameras_and_collision_free_occluder():
    robot = make_robot("gantry")
    m = build_model(robot)
    ids = SceneIds.from_model(m, robot)
    assert set(ids.cams) == {"third_person", "wrist"}
    assert ids.occluder_mocap >= 0
    assert m.geom_contype[ids.occluder_geom] == 0 and m.geom_conaffinity[ids.occluder_geom] == 0


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
        assert obs["rgb"][cam].std() > 1.0
    assert obs["proprio"].shape == (2, 5)
    env.close()
