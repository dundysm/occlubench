"""Moving physical occluder: placement geometry and coverage measurement.

The occluder is a mocap plate with no collision. When active, it is placed on the
ray from the occluded camera to the task object at ``depth_frac`` of that distance,
oriented to face the camera, and shifted sideways by a lateral offset. The offset is
calibrated per episode (bisection on segmentation renders) so that the occluder hides
the requested fraction of the object's pixels. Entry and exit are a straight-line
slide from far off to the side, so the occluder visibly moves in and out of view.
"""

from __future__ import annotations

import mujoco
import numpy as np

from occlubench.scene import PARK_POS, SceneIds

DEPTH_FRAC = 0.45
ENTRY_OFFSET = 0.35  # metres beyond the hold offset where the slide starts/ends


def _cam_basis(d: mujoco.MjData, cam_id: int):
    pos = d.cam_xpos[cam_id].copy()
    rot = d.cam_xmat[cam_id].reshape(3, 3)
    return pos, rot  # columns: cam x (right), y (up), z (backward)


def _plate_quat(ray: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Quaternion whose local z axis is along ``ray`` (plate faces the camera)."""
    z = ray / np.linalg.norm(ray)
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel())
    return q


def occluder_pose(d: mujoco.MjData, ids: SceneIds, camera: str, lateral: float, depth_frac: float = DEPTH_FRAC):
    """World pose for the occluder given a lateral offset (metres) from the cam->object ray."""
    cpos, rot = _cam_basis(d, ids.cams[camera])
    target = d.xpos[ids.cube_body]
    ray = target - cpos
    right = rot[:, 0]
    # Make the lateral direction exactly perpendicular to the ray.
    side = right - ray * (right @ ray) / (ray @ ray)
    side /= np.linalg.norm(side) + 1e-9
    pos = cpos + depth_frac * ray + lateral * side
    return pos, _plate_quat(ray, rot[:, 1])


def set_occluder(d: mujoco.MjData, ids: SceneIds, pos, quat=None):
    d.mocap_pos[ids.occluder_mocap] = pos
    d.mocap_quat[ids.occluder_mocap] = [1, 0, 0, 0] if quat is None else quat


def park_occluder(d: mujoco.MjData, ids: SceneIds):
    set_occluder(d, ids, PARK_POS)


class CoverageMeter:
    """Measures the fraction of task-object pixels hidden in a camera (CPU segmentation)."""

    def __init__(self, m: mujoco.MjModel, ids: SceneIds, height: int = 64, width: int = 64):
        self.m, self.ids = m, ids
        self.r = mujoco.Renderer(m, height, width)
        self.r.enable_segmentation_rendering()

    def object_pixels(self, d: mujoco.MjData, camera: str) -> int:
        self.r.update_scene(d, camera=camera)
        seg = self.r.render()
        mask = (seg[..., 0] == self.ids.cube_geom) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        return int(mask.sum())

    def coverage(self, d: mujoco.MjData, camera: str) -> float:
        """Hidden fraction = 1 - visible(with occluder) / visible(occluder parked)."""
        saved_p = d.mocap_pos[self.ids.occluder_mocap].copy()
        saved_q = d.mocap_quat[self.ids.occluder_mocap].copy()
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        with_occ = self.object_pixels(d, camera)
        park_occluder(d, self.ids)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        full = self.object_pixels(d, camera)
        set_occluder(d, self.ids, saved_p, saved_q)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        if full == 0:
            return float("nan")
        return 1.0 - with_occ / full

    def calibrate(
        self, d: mujoco.MjData, camera: str, severity: float, iters: int = 12, max_lateral: float = 0.3
    ) -> tuple[float, float]:
        """Bisection on lateral offset so coverage ~= severity. Returns (lateral, achieved)."""

        def cov_at(lat):
            pos, quat = occluder_pose(d, self.ids, camera, lat)
            saved = d.mocap_pos[self.ids.occluder_mocap].copy(), d.mocap_quat[self.ids.occluder_mocap].copy()
            set_occluder(d, self.ids, pos, quat)
            c = self.coverage(d, camera)
            set_occluder(d, self.ids, *saved)
            return c

        if severity >= 1.0 or cov_at(0.0) <= severity:
            return 0.0, cov_at(0.0)
        lo, hi = 0.0, max_lateral  # coverage decreases with lateral offset
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if cov_at(mid) > severity:
                lo = mid
            else:
                hi = mid
        lat = 0.5 * (lo + hi)
        return lat, cov_at(lat)

    def close(self):
        self.r.close()


def schedule_phase(t: int, onset_step: int, hold_steps: int, slide_steps: int) -> float:
    """Return the lateral blend in [0, 1]: 0 = at hold offset (fully in), 1 = off to the side.

    Timeline: parked before onset, slide in over ``slide_steps``, hold for ``hold_steps``,
    slide out over ``slide_steps``, then parked. Returns NaN when parked.
    """
    t0 = onset_step
    t1 = t0 + slide_steps
    t2 = t1 + hold_steps
    t3 = t2 + slide_steps
    if t < t0 or t >= t3:
        return float("nan")
    if t < t1:
        return 1.0 - (t - t0 + 1) / slide_steps
    if t < t2:
        return 0.0
    return (t - t2 + 1) / slide_steps
