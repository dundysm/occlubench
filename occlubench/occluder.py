"""Occluder schedule and the CPU segmentation coverage meter (reference implementation).

Runtime placement and coverage are vectorised in ``occlubench.geometry`` and run for all envs
at once on either backend. ``CoverageMeter`` renders segmentation masks with the CPU renderer
and is used to validate the geometric estimate (see tests/test_geometry.py).
"""

from __future__ import annotations

import mujoco
import numpy as np

from occlubench.scene import PARK_POS, SceneIds

DEPTH_FRAC = 0.45
ENTRY_OFFSET = 0.35  # metres beyond the hold offset where the slide starts/ends
SLIDE_STEPS = 8
OCC_HALF = np.array([0.06, 0.06])


def schedule_phase(t, onset_step, hold_steps, slide_steps=SLIDE_STEPS):
    """Vectorised. Lateral blend in [0, 1] (0 = fully in), NaN when parked.

    Timeline: parked before onset, slide in over ``slide_steps``, hold for ``hold_steps``,
    slide out over ``slide_steps``, then parked.
    """
    t = np.asarray(t, float)
    t0 = np.asarray(onset_step, float)
    t1 = t0 + slide_steps
    t2 = t1 + np.asarray(hold_steps, float)
    t3 = t2 + slide_steps
    out = np.full(np.broadcast(t, t0, t2).shape, np.nan)
    out = np.where((t >= t0) & (t < t1), 1.0 - (t - t0 + 1) / slide_steps, out)
    out = np.where((t >= t1) & (t < t2), 0.0, out)
    out = np.where((t >= t2) & (t < t3), (t - t2 + 1) / slide_steps, out)
    return out if out.ndim else float(out)


def set_occluder(d: mujoco.MjData, ids: SceneIds, pos, quat=None):
    d.mocap_pos[ids.occluder_mocap] = pos
    d.mocap_quat[ids.occluder_mocap] = [1, 0, 0, 0] if quat is None else quat


def park_occluder(d: mujoco.MjData, ids: SceneIds):
    set_occluder(d, ids, PARK_POS)


class CoverageMeter:
    """Fraction of task-object pixels hidden in a camera, from CPU segmentation renders."""

    def __init__(self, m: mujoco.MjModel, ids: SceneIds, height: int = 64, width: int = 64):
        self.m, self.ids = m, ids
        self.r = mujoco.Renderer(m, height, width)
        self.r.enable_segmentation_rendering()

    def object_pixels(self, d: mujoco.MjData, camera: str) -> int:
        self.r.update_scene(d, camera=self.ids.cams[camera])
        seg = self.r.render()
        mask = (seg[..., 0] == self.ids.cube_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        return int(mask.sum())

    def coverage(self, d: mujoco.MjData, camera: str) -> float:
        saved_p = d.mocap_pos[self.ids.occluder_mocap].copy()
        saved_q = d.mocap_quat[self.ids.occluder_mocap].copy()
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        with_occ = self.object_pixels(d, camera)
        park_occluder(d, self.ids)
        mujoco.mj_kinematics(self.m, d)
        full = self.object_pixels(d, camera)
        set_occluder(d, self.ids, saved_p, saved_q)
        mujoco.mj_kinematics(self.m, d)
        if full == 0:
            return float("nan")
        return 1.0 - with_occ / full

    def close(self):
        self.r.close()
