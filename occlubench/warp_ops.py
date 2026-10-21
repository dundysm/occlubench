"""On-device occluder placement, coverage and success for the MuJoCo Warp path.

These kernels mirror the NumPy reference in ``occlubench.geometry`` / ``OccluEnv`` (the CPU
fallback) and read poses straight from ``mujoco_warp`` Data arrays. Per control step they
write the occluder mocap pose into Data and keep all per-env bookkeeping (calibrated lateral
offset, world anchor, coverage sums, success step) on the device, so nothing about the
occluder or scoring crosses to the host until results are requested.

Precision is float32 (the NumPy path is float64). tests/test_warp_ops.py checks parity.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from occlubench import geometry
from occlubench.occluder import DEPTH_FRAC, ENTRY_OFFSET, OCC_HALF, SLIDE_STEPS
from occlubench.scene import CUBE_HALF, GOAL_RADIUS, PARK_POS, TABLE_TOP_Z

CAL_ITERS = 14
MAX_LATERAL = 0.3


@wp.func
def _frame(cp: wp.vec3, cm: wp.mat33, tgt: wp.vec3, lat: float, depth: float):
    ray = tgt - cp
    z = wp.normalize(ray)
    right = wp.vec3(cm[0, 0], cm[1, 0], cm[2, 0])
    side = right - z * wp.dot(right, z)
    side = side / (wp.length(side) + 1.0e-9)
    y = wp.cross(z, side)
    rot = wp.matrix_from_cols(side, y, z)
    pos = cp + depth * ray + lat * side
    return pos, rot, side


@wp.func
def _coverage(
    cp: wp.vec3,
    cm: wp.mat33,
    tan_y: float,
    aspect: float,
    cube_p: wp.vec3,
    cube_m: wp.mat33,
    occ_p: wp.vec3,
    occ_m: wp.mat33,
    half: wp.vec2,
    pts: wp.array(dtype=wp.vec3),
    nrms: wp.array(dtype=wp.vec3),
) -> float:
    total = wp.float32(0.0)
    hidden = wp.float32(0.0)
    cmt = wp.transpose(cm)
    omt = wp.transpose(occ_m)
    o = omt @ (cp - occ_p)
    for k in range(pts.shape[0]):
        p = cube_p + cube_m @ pts[k]
        n = cube_m @ nrms[k]
        to_cam = cp - p
        dist = wp.length(to_cam)
        cos = wp.dot(n, to_cam) / dist
        pc = cmt @ (p - cp)
        dep = -pc[2]
        in_view = dep > 1.0e-3 and wp.abs(pc[1]) < tan_y * dep and wp.abs(pc[0]) < tan_y * aspect * dep
        if cos > 0.0 and in_view:
            w = cos / (dist * dist)
            total += w
            d = omt @ (p - cp)
            if d[2] != 0.0:
                t = -o[2] / d[2]
                hx = o[0] + t * d[0]
                hy = o[1] + t * d[1]
                if t > 0.0 and t < 1.0 and wp.abs(hx) < half[0] and wp.abs(hy) < half[1]:
                    hidden += w
    if total > 0.0:
        return hidden / wp.max(total, 1.0e-12)
    return wp.nan


@wp.func
def _phase(t: float, t0: float, hold: float, slide: float) -> float:
    t1 = t0 + slide
    t2 = t1 + hold
    t3 = t2 + slide
    if t >= t0 and t < t1:
        return 1.0 - (t - t0 + 1.0) / slide
    if t >= t1 and t < t2:
        return 0.0
    if t >= t2 and t < t3:
        return (t - t2 + 1.0) / slide
    return wp.nan


@wp.func
def _quat_wxyz(rot: wp.mat33) -> wp.quat:
    q = wp.quat_from_matrix(rot)  # warp stores (x, y, z, w)
    return wp.quat(q[3], q[0], q[1], q[2])  # MuJoCo Warp stores wxyz in a quat slot


@wp.kernel
def _occluder_step(
    t: int,
    # mujoco_warp Data
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
    # constants
    cam_ids: wp.array(dtype=int),
    tan_y: wp.array(dtype=float),
    aspect: float,
    cube_body: int,
    occ_mocap: int,
    half: wp.vec2,
    park: wp.vec3,
    pts: wp.array(dtype=wp.vec3),
    nrms: wp.array(dtype=wp.vec3),
    # per-env condition
    cam_idx: wp.array(dtype=int),
    clean: wp.array(dtype=int),
    onset: wp.array(dtype=int),
    hold: wp.array(dtype=int),
    severity: wp.array(dtype=float),
    world_anchor: wp.array(dtype=int),
    # per-env state (in/out)
    hold_lat: wp.array(dtype=float),
    anchor_pos: wp.array(dtype=wp.vec3),
    anchor_rot: wp.array(dtype=wp.mat33),
    anchor_side: wp.array(dtype=wp.vec3),
    cov_onset: wp.array(dtype=float),
    cov_sum: wp.array2d(dtype=float),
    cov_max: wp.array2d(dtype=float),
    cov_cnt: wp.array(dtype=float),
    active_out: wp.array(dtype=int),
):
    i = wp.tid()
    k = cam_idx[i]
    cid = cam_ids[k]
    cp = cam_xpos[i, cid]
    cm = cam_xmat[i, cid]
    cube_p = xpos[i, cube_body]
    cube_m = xmat[i, cube_body]
    ph = wp.nan
    if clean[i] == 0:
        ph = _phase(float(t), float(onset[i]), float(hold[i]), float(SLIDE_STEPS))

    if clean[i] == 0 and t == onset[i]:  # calibrate: bisection so target-camera coverage ~= severity
        sev = severity[i]
        p0, r0, s0 = _frame(cp, cm, cube_p, 0.0, DEPTH_FRAC)
        c0 = _coverage(cp, cm, tan_y[k], aspect, cube_p, cube_m, p0, r0, half, pts, nrms)
        lo = wp.float32(0.0)
        hi = float(MAX_LATERAL)
        for _ in range(CAL_ITERS):
            mid = 0.5 * (lo + hi)
            pm, rm, sm = _frame(cp, cm, cube_p, mid, DEPTH_FRAC)
            cmid = _coverage(cp, cm, tan_y[k], aspect, cube_p, cube_m, pm, rm, half, pts, nrms)
            if cmid > sev:
                lo = mid
            else:
                hi = mid
        lat = 0.5 * (lo + hi)
        if c0 <= sev or sev >= 1.0:
            lat = 0.0
        hold_lat[i] = lat
        pa, ra, sa = _frame(cp, cm, cube_p, lat, DEPTH_FRAC)
        anchor_pos[i] = pa
        anchor_rot[i] = ra
        anchor_side[i] = sa
        cov_onset[i] = _coverage(cp, cm, tan_y[k], aspect, cube_p, cube_m, pa, ra, half, pts, nrms)

    if wp.isnan(ph):
        mocap_pos[i, occ_mocap] = park
        mocap_quat[i, occ_mocap] = wp.quat(1.0, 0.0, 0.0, 0.0)
        active_out[i] = 0
        return

    lat = hold_lat[i] + ph * ENTRY_OFFSET
    pos, rot, side = _frame(cp, cm, cube_p, lat, DEPTH_FRAC)
    if world_anchor[i] != 0:
        pos = anchor_pos[i] + (ph * ENTRY_OFFSET) * anchor_side[i]
        rot = anchor_rot[i]
    mocap_pos[i, occ_mocap] = pos
    mocap_quat[i, occ_mocap] = _quat_wxyz(rot)
    active_out[i] = 1

    # coverage in both cameras: column 0 = target camera, column 1 = the other camera
    for j in range(2):
        kk = k
        if j == 1:
            kk = 1 - k
        c2 = cam_ids[kk]
        cv = _coverage(cam_xpos[i, c2], cam_xmat[i, c2], tan_y[kk], aspect, cube_p, cube_m, pos, rot, half, pts, nrms)
        if wp.isnan(cv):
            cv = 0.0
        cov_sum[i, j] = cov_sum[i, j] + cv
        cov_max[i, j] = wp.max(cov_max[i, j], cv)
    cov_cnt[i] = cov_cnt[i] + 1.0


@wp.kernel
def _success_step(
    t: int,
    xpos: wp.array2d(dtype=wp.vec3),
    cube_body: int,
    goal: wp.vec3,
    success: wp.array(dtype=int),
    success_step: wp.array(dtype=int),
):
    i = wp.tid()
    c = xpos[i, cube_body]
    dx = c[0] - goal[0]
    dy = c[1] - goal[1]
    ok = wp.sqrt(dx * dx + dy * dy) < GOAL_RADIUS and c[2] < TABLE_TOP_Z + CUBE_HALF + 0.01
    success[i] = int(ok)
    if ok and success_step[i] < 0:
        success_step[i] = t


class DeviceOccluder:
    """Per-env occluder + scoring state on the Warp device. One instance per OccluEnv."""

    def __init__(self, env, d):
        self.env, self.d, self.n = env, d, env.n
        dev = d.qpos.device
        self.dev = dev
        from occlubench.scene import CAMERAS

        self.cam_ids = wp.array(np.array([env.ids.cams[c] for c in CAMERAS], np.int32), dtype=int, device=dev)
        self.tan_y = wp.array(np.tan(np.radians(env.fovy) / 2).astype(np.float32), dtype=float, device=dev)
        self.pts = wp.array(geometry.CUBE_PTS.astype(np.float32), dtype=wp.vec3, device=dev)
        self.nrms = wp.array(geometry.CUBE_NRM.astype(np.float32), dtype=wp.vec3, device=dev)
        self.half = wp.vec2(*OCC_HALF.astype(np.float32))
        self.park = wp.vec3(*np.asarray(PARK_POS, np.float32))

    def _i(self, a):
        return wp.array(np.asarray(a, np.int32), dtype=int, device=self.dev)

    def _f(self, a):
        return wp.array(np.asarray(a, np.float32), dtype=float, device=self.dev)

    def reset(self, cam_idx, clean, onset, hold, severity, world_anchor, goal):
        n, dev = self.n, self.dev
        self.cond = [self._i(cam_idx), self._i(clean), self._i(onset), self._i(hold), self._f(severity)]
        self.cond.append(self._i(world_anchor))
        self.goal = wp.vec3(*np.asarray(goal, np.float32))
        self.hold_lat = wp.zeros(n, dtype=float, device=dev)
        self.anchor_pos = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.anchor_rot = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33, device=dev)
        self.anchor_side = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.cov_onset = wp.array(np.full(n, np.nan, np.float32), dtype=float, device=dev)
        self.cov_sum = wp.zeros((n, 2), dtype=float, device=dev)
        self.cov_max = wp.zeros((n, 2), dtype=float, device=dev)
        self.cov_cnt = wp.zeros(n, dtype=float, device=dev)
        self.active = wp.zeros(n, dtype=int, device=dev)
        self.success = wp.zeros(n, dtype=int, device=dev)
        self.success_step = wp.array(np.full(n, -1, np.int32), dtype=int, device=dev)

    def place(self, t: int):
        d, e = self.d, self.env
        wp.launch(
            _occluder_step,
            dim=self.n,
            device=self.dev,
            inputs=[
                t,
                d.cam_xpos,
                d.cam_xmat,
                d.xpos,
                d.xmat,
                d.mocap_pos,
                d.mocap_quat,
                self.cam_ids,
                self.tan_y,
                float(e.w / e.h),
                int(e.ids.cube_body),
                int(e.ids.occluder_mocap),
                self.half,
                self.park,
                self.pts,
                self.nrms,
                *self.cond,
                self.hold_lat,
                self.anchor_pos,
                self.anchor_rot,
                self.anchor_side,
                self.cov_onset,
                self.cov_sum,
                self.cov_max,
                self.cov_cnt,
                self.active,
            ],
        )

    def score(self, t: int):
        wp.launch(
            _success_step,
            dim=self.n,
            device=self.dev,
            inputs=[t, self.d.xpos, int(self.env.ids.cube_body), self.goal, self.success, self.success_step],
        )

    # host sync, only when results are requested
    def stats(self) -> dict:
        cnt = self.cov_cnt.numpy().astype(float)
        s, mx = self.cov_sum.numpy().astype(float), self.cov_max.numpy().astype(float)
        has = cnt[:, None] > 0
        mean = np.where(has, s / np.maximum(cnt, 1)[:, None], np.nan)
        mx = np.where(has, mx, np.nan)
        return dict(
            coverage_onset=self.cov_onset.numpy().astype(float),
            coverage=mean[:, 0],
            coverage_max=mx[:, 0],
            coverage_other=mean[:, 1],
            coverage_other_max=mx[:, 1],
        )
