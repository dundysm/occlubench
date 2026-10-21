"""Scene loading and named-index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import mujoco
import numpy as np

SCENE_XML = "tabletop.xml"
TABLE_TOP_Z = 0.40
CUBE_HALF = 0.02
# Cube spawn region (x, y) on the table, kept away from the goal.
SPAWN_LOW = np.array([-0.2, -0.15])
SPAWN_HIGH = np.array([0.0, 0.05])
GOAL_RADIUS = 0.05
PARK_POS = np.array([0.0, 0.0, 3.0])


def scene_path() -> str:
    return str(resources.files("occlubench") / "assets" / SCENE_XML)


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(scene_path())


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
    cams: dict

    @classmethod
    def from_model(cls, m: mujoco.MjModel) -> SceneIds:
        occ = m.body("occluder").id
        return cls(
            cube_body=m.body("cube").id,
            cube_geom=m.geom("cube_geom").id,
            cube_qpos=int(m.jnt_qposadr[m.joint("cube").id]),
            occluder_body=occ,
            occluder_mocap=int(m.body_mocapid[occ]),
            occluder_geom=m.geom("occluder_geom").id,
            goal_site=m.site("goal").id,
            tcp_site=m.site("tcp").id,
            cams={n: m.camera(n).id for n in ("third_person", "wrist")},
        )
