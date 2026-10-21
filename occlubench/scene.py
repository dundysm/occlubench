"""Scene composition (base tabletop + robot) and named-index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import mujoco
import numpy as np

SCENE_XML = "scene_base.xml"
TABLE_TOP_Z = 0.40
CUBE_HALF = 0.02
SPAWN_LOW = np.array([-0.2, -0.15])
SPAWN_HIGH = np.array([0.0, 0.05])
GOAL_RADIUS = 0.05
PARK_POS = np.array([0.0, 0.0, 3.0])
CAMERAS = ("third_person", "wrist")


def scene_path() -> str:
    return str(resources.files("occlubench") / "assets" / SCENE_XML)


def build_model(robot) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(scene_path())
    robot.attach(spec)
    m = spec.compile()
    robot.setup(m)
    return m


def load_model(robot_name: str = "gantry") -> mujoco.MjModel:
    from occlubench.robots import make_robot

    return build_model(make_robot(robot_name))


@dataclass
class SceneIds:
    cube_body: int
    cube_geom: int
    cube_qpos: int
    occluder_body: int
    occluder_mocap: int
    occluder_geom: int
    goal_site: int
    tcp_site: int
    cams: dict  # logical name ("third_person", "wrist") -> camera id

    @classmethod
    def from_model(cls, m: mujoco.MjModel, robot) -> SceneIds:
        occ = m.body("occluder").id
        return cls(
            cube_body=m.body("cube").id,
            cube_geom=m.geom("cube_geom").id,
            cube_qpos=int(m.jnt_qposadr[m.joint("cube").id]),
            occluder_body=occ,
            occluder_mocap=int(m.body_mocapid[occ]),
            occluder_geom=m.geom("occluder_geom").id,
            goal_site=m.site("goal").id,
            tcp_site=m.site(robot.tcp_site).id,
            cams={"third_person": m.camera("third_person").id, "wrist": m.camera(robot.wrist_cam).id},
        )
