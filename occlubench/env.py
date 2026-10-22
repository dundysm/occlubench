"""Batched OccluBench environment.

robot:    "gantry" (simple, self-authored) or "panda" (Franka Panda from MuJoCo Menagerie).
physics:  "cpu" (one mujoco.MjData per env) or "warp" (MuJoCo Warp, nworld = n_envs).
render:   "cpu" (mujoco.Renderer, OSMesa/EGL), "warp" (MuJoCo Warp batch ray tracer), or "none".

On the Warp path (default ``device_ops=True``) occluder placement, calibration, per-step
coverage and success run as Warp kernels (``occlubench.warp_ops``) that read poses from and
write the occluder mocap pose into MuJoCo Warp Data; that state reaches the host only when
``success_step``/``coverage_stats()``/``info`` values are read. IK is still host-side NumPy,
so each step still reads qpos, tcp pose and the tcp Jacobian and pushes ctrl. The CPU path
(and ``device_ops=False``) uses the NumPy reference in ``occlubench.geometry``.

Policies see only RGB from the ``third_person`` and ``wrist`` cameras plus proprioception.
``env.privileged()`` is for the scripted oracle and for scoring only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np

from occlubench import geometry
from occlubench.conditions import CLEAN, Condition
from occlubench.occluder import DEPTH_FRAC, ENTRY_OFFSET, OCC_HALF, SLIDE_STEPS, schedule_phase
from occlubench.robots import make_robot
from occlubench.scene import (
    CAMERAS,
    CUBE_HALF,
    GOAL_RADIUS,
    PARK_POS,
    SPAWN_HIGH,
    SPAWN_LOW,
    TABLE_TOP_Z,
    SceneIds,
    build_model,
)


def is_success(cube_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """Cube resting on the table within GOAL_RADIUS of the goal centre. Shapes (N, 3)."""
    dxy = np.linalg.norm(cube_pos[:, :2] - goal_pos[None, :2], axis=1)
    resting = cube_pos[:, 2] < TABLE_TOP_Z + CUBE_HALF + 0.01
    return (dxy < GOAL_RADIUS) & resting


class _CpuBackend:
    def __init__(self, env):
        self.env, self.m = env, env.m
        self.datas = [mujoco.MjData(self.m) for _ in range(env.n)]

    def load(self, datas):
        for d, src in zip(self.datas, datas):
            mujoco.mj_copyData(d, self.m, src)

    def write(self, ctrl, mocap_pos, mocap_quat):
        for i, d in enumerate(self.datas):
            d.ctrl[:] = ctrl[i]
            d.mocap_pos[:] = mocap_pos[i]
            d.mocap_quat[:] = mocap_quat[i]

    def step(self, substeps):
        for d in self.datas:
            mujoco.mj_step(self.m, d, nstep=substeps)

    def refresh(self, need_jac):
        """Bring derived quantities up to date with qpos; return batched arrays."""
        e = self.env
        jac = np.zeros((e.n, 6, self.m.nv)) if need_jac else None
        for i, d in enumerate(self.datas):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_camlight(self.m, d)
            if need_jac:
                mujoco.mj_comPos(self.m, d)
                mujoco.mj_jacSite(self.m, d, jac[i, :3], jac[i, 3:], e.ids.tcp_site)
        cams = [e.ids.cams[c] for c in CAMERAS]
        return dict(
            qpos=np.stack([d.qpos for d in self.datas]),
            cube_pos=np.stack([d.xpos[e.ids.cube_body] for d in self.datas]),
            cube_mat=np.stack([d.xmat[e.ids.cube_body].reshape(3, 3) for d in self.datas]),
            tcp_pos=np.stack([d.site_xpos[e.ids.tcp_site] for d in self.datas]),
            tcp_mat=np.stack([d.site_xmat[e.ids.tcp_site].reshape(3, 3) for d in self.datas]),
            cam_pos=np.stack([[d.cam_xpos[c] for c in cams] for d in self.datas]),
            cam_mat=np.stack([[d.cam_xmat[c].reshape(3, 3) for c in cams] for d in self.datas]),
            jac=jac,
        )


class _WarpBackend:
    def __init__(self, env):
        import mujoco_warp as mjw
        import warp as wp

        wp.init()
        self.env, self.mjw, self.wp, self.mjm = env, mjw, wp, env.m
        d0 = mujoco.MjData(env.m)
        mujoco.mj_forward(env.m, d0)
        self.m = mjw.put_model(env.m)
        # Contact buffers sized for the Panda (fingers + cube + table); the default njmax=64 overflows.
        self.d = mjw.put_data(env.m, d0, nworld=env.n, nconmax=64, njmax=256)
        self.cam_idx = [env.ids.cams[c] for c in CAMERAS]
        self.jacp = wp.zeros((env.n, 3, env.m.nv), dtype=float)
        self.jacr = wp.zeros((env.n, 3, env.m.nv), dtype=float)
        self.point = wp.zeros(env.n, dtype=wp.vec3)
        self.body = wp.array(np.full(env.n, env.m.site_bodyid[env.ids.tcp_site], np.int32), dtype=wp.int32)

    def _assign(self, name, host):
        arr = getattr(self.d, name)
        shape = (self.env.n, *arr.shape[1:])
        if arr.dtype in (self.wp.vec3, self.wp.vec3f):
            v = self.wp.array(host.reshape(*shape, 3).astype(np.float32), dtype=arr.dtype, device=arr.device)
        elif arr.dtype in (self.wp.quat, self.wp.quatf, self.wp.vec4, self.wp.vec4f):
            v = self.wp.array(host.reshape(*shape, 4).astype(np.float32), dtype=arr.dtype, device=arr.device)
        else:
            v = self.wp.array(host.reshape(shape).astype(np.float32), dtype=arr.dtype, device=arr.device)
        arr.assign(v)

    def load(self, datas):
        for f in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            self._assign(f, np.stack([getattr(d, f) for d in datas]))
        if self.mjm.na:
            self._assign("act", np.stack([d.act for d in datas]))
        self.d.time.zero_()

    def write(self, ctrl, mocap_pos=None, mocap_quat=None):
        if ctrl is not None:  # None when ctrl was written on-device (warp_ik)
            self._assign("ctrl", ctrl)
        if mocap_pos is not None:  # None when the occluder pose is written on-device (warp_ops)
            self._assign("mocap_pos", mocap_pos)
            self._assign("mocap_quat", mocap_quat)

    def step(self, substeps):
        for _ in range(substeps):
            self.mjw.step(self.m, self.d)

    def refresh(self, need_jac):
        e, mjw, wp = self.env, self.mjw, self.wp
        mjw.kinematics(self.m, self.d)
        mjw.camlight(self.m, self.d)
        if need_jac:
            mjw.com_pos(self.m, self.d)
            from occlubench.warp_ik import tcp_point

            wp.launch(tcp_point, dim=e.n, inputs=[self.d.site_xpos, int(e.ids.tcp_site), self.point])
            mjw.jac(self.m, self.d, self.jacp, self.jacr, self.point, self.body)
        d, tcp, cube, nc = self.d, e.ids.tcp_site, e.ids.cube_body, len(CAMERAS)
        getters = dict(
            qpos=lambda: d.qpos.numpy().astype(float),
            cube_pos=lambda: d.xpos.numpy()[:, cube].astype(float),
            cube_mat=lambda: d.xmat.numpy()[:, cube].reshape(-1, 3, 3).astype(float),
            tcp_pos=lambda: d.site_xpos.numpy()[:, tcp].astype(float),
            tcp_mat=lambda: d.site_xmat.numpy()[:, tcp].reshape(-1, 3, 3).astype(float),
            cam_pos=lambda: d.cam_xpos.numpy()[:, self.cam_idx].astype(float),
            cam_mat=lambda: d.cam_xmat.numpy()[:, self.cam_idx].reshape(-1, nc, 3, 3).astype(float),
            jac=(lambda: np.concatenate([self.jacp.numpy(), self.jacr.numpy()], 1)) if need_jac else (lambda: None),
        )
        return _LazyInfo(**getters)


class _LazyInfo(Mapping):
    """Step info whose device-resident values are copied to the host only when read."""

    def __init__(self, **getters):
        self._g, self._v = getters, {}

    def __getitem__(self, k):
        if k not in self._v:
            self._v[k] = self._g[k]()
        return self._v[k]

    def __iter__(self):
        return iter(self._g)

    def __len__(self):
        return len(self._g)


class OccluEnv:
    def __init__(
        self,
        n_envs: int = 8,
        horizon: int = 200,
        image_size: tuple[int, int] = (96, 96),
        physics: str = "cpu",
        render: str = "cpu",
        robot: str = "gantry",
        substeps: int = 10,
        device_ops: bool | None = None,
        device_ik: bool | None = None,
    ):
        self.n, self.horizon, self.substeps = n_envs, horizon, substeps
        self.h, self.w = image_size
        self.physics_backend, self.render_backend = physics, render
        self.robot = make_robot(robot)
        self.m = build_model(self.robot)
        self.ids = SceneIds.from_model(self.m, self.robot)
        self.fovy = np.array([self.m.cam_fovy[self.ids.cams[c]] for c in CAMERAS])
        self.conditions: list[Condition] = [CLEAN] * n_envs
        self.backend = _WarpBackend(self) if physics == "warp" else _CpuBackend(self)
        # Occluder placement, coverage and success: Warp kernels on the Warp path (default),
        # NumPy reference otherwise. ``device_ops=False`` forces the NumPy fallback.
        if device_ops is None:
            device_ops = physics == "warp"
        if device_ops and physics != "warp":
            raise ValueError("device_ops=True requires physics='warp'")
        self._dev = None
        if device_ops:
            from occlubench.warp_ops import DeviceOccluder

            self._dev = DeviceOccluder(self, self.backend.d)
        # Robot IK (action -> ctrl): Warp kernels on the Warp path (default), NumPy otherwise.
        if device_ik is None:
            device_ik = physics == "warp"
        if device_ik and physics != "warp":
            raise ValueError("device_ik=True requires physics='warp'")
        self._ik = None
        if device_ik:
            from occlubench.warp_ik import DeviceIK

            self._ik = DeviceIK(self, self.backend)
        self._ctrl = None
        self._renderer = mujoco.Renderer(self.m, self.h, self.w) if render == "cpu" else None
        self._host = None  # host MjData mirror used only for CPU rendering on the Warp path
        self._rc = None
        if render == "warp":
            self._init_warp_render()

    def _init_warp_render(self):
        wb = self.backend if isinstance(self.backend, _WarpBackend) else None
        if wb is None:
            raise ValueError("render='warp' requires physics='warp'")
        self._rc = wb.mjw.create_render_context(
            self.m,
            nworld=self.n,
            cam_res=(self.w, self.h),
            render_rgb=True,
            cam_active=[self.ids.cams[c] for c in CAMERAS],
        )
        self._rgb_out = wb.wp.zeros((self.n, self.h, self.w), dtype=wb.wp.vec3)

    # ------------------------------------------------------------------ reset
    def reset(self, seeds: Sequence[int], conditions: Sequence[Condition] | None = None):
        assert len(seeds) == self.n, "one seed per env"
        self.seeds = list(seeds)
        self.conditions = list(conditions) if conditions is not None else [CLEAN] * self.n
        c = self.conditions
        self.cam_idx = np.array([CAMERAS.index(x.camera) if not x.is_clean else 0 for x in c])
        self.clean_mask = np.array([x.is_clean for x in c])
        self.onset_step = np.array([round(x.onset * self.horizon) for x in c])
        self.hold_steps = np.array([max(1, round(x.duration * self.horizon)) for x in c])
        self.severity = np.array([x.severity for x in c])
        self.world_anchor = np.array([x.anchor == "world" for x in c])
        self.t = 0
        self._success_step = np.full(self.n, -1)
        self._hold_lat = np.zeros(self.n)
        self._anchor_pos = np.zeros((self.n, 3))
        self._anchor_rot = np.tile(np.eye(3), (self.n, 1, 1))
        self._anchor_side = np.zeros((self.n, 3))
        self.coverage_onset = np.full(self.n, np.nan)
        self._cov_sum = np.zeros((self.n, 2))
        self._cov_max = np.zeros((self.n, 2))
        self._cov_cnt = np.zeros(self.n)
        datas = []
        for s in self.seeds:
            d = mujoco.MjData(self.m)
            rng = np.random.default_rng(s)
            xy = rng.uniform(SPAWN_LOW, SPAWN_HIGH)
            a = self.ids.cube_qpos
            d.qpos[a : a + 3] = [xy[0], xy[1], TABLE_TOP_Z + CUBE_HALF]
            yaw = rng.uniform(-math.pi / 4, math.pi / 4)
            d.qpos[a + 3 : a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self.robot.home(self.m, d)
            d.mocap_pos[self.ids.occluder_mocap] = PARK_POS
            mujoco.mj_forward(self.m, d)
            datas.append(d)
        self.ctrl = np.stack([d.ctrl.copy() for d in datas])
        self.mocap_pos = np.stack([d.mocap_pos.copy() for d in datas])
        self.mocap_quat = np.stack([d.mocap_quat.copy() for d in datas])
        self.goal = datas[0].site_xpos[self.ids.goal_site].copy()
        self.backend.load(datas)
        self.state = self.backend.refresh(self.robot.needs_jacobian)
        if self._dev is not None:
            self._dev.reset(
                self.cam_idx,
                self.clean_mask,
                self.onset_step,
                self.hold_steps,
                self.severity,
                self.world_anchor,
                self.goal,
            )
        return self._obs()

    # ------------------------------------------------------------------ occluder (vectorised)
    def _cam(self, idx):
        r = np.arange(self.n)
        return self.state["cam_pos"][r, idx], self.state["cam_mat"][r, idx]

    def _coverage(self, cam, occ_pos, occ_mat, active):
        cp, cm = self._cam(np.full(self.n, cam))
        return geometry.coverage(
            cp,
            cm,
            self.fovy[cam],
            self.w / self.h,
            self.state["cube_pos"],
            self.state["cube_mat"],
            occ_pos,
            occ_mat,
            OCC_HALF,
            active,
        )

    def _calibrate(self, mask, iters: int = 14, max_lateral: float = 0.3):
        """Vectorised bisection on the lateral offset so target-camera coverage ~= severity."""
        cp, cm = self._cam(self.cam_idx)
        tgt = self.state["cube_pos"]
        depth = np.full(self.n, DEPTH_FRAC)

        def cov(lat):
            pos, rot, _ = geometry.occluder_frame(cp, cm, tgt, lat, depth)
            out = np.empty(self.n)
            for k in range(len(CAMERAS)):
                sel = self.cam_idx == k
                if sel.any():
                    out[sel] = self._coverage(k, pos, rot, None)[sel]
            return out

        lo, hi = np.zeros(self.n), np.full(self.n, max_lateral)
        full = cov(lo) <= self.severity
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            above = cov(mid) > self.severity
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        lat = np.where(full | (self.severity >= 1.0), 0.0, 0.5 * (lo + hi))
        self._hold_lat = np.where(mask, lat, self._hold_lat)
        pos, rot, side = geometry.occluder_frame(cp, cm, tgt, self._hold_lat, depth)
        self._anchor_pos = np.where(mask[:, None], pos, self._anchor_pos)
        self._anchor_rot = np.where(mask[:, None, None], rot, self._anchor_rot)
        self._anchor_side = np.where(mask[:, None], side, self._anchor_side)
        self.coverage_onset = np.where(mask, cov(self._hold_lat), self.coverage_onset)

    def _place_occluders(self):
        phase = schedule_phase(np.full(self.n, self.t), self.onset_step, self.hold_steps, SLIDE_STEPS)
        phase = np.where(self.clean_mask, np.nan, phase)
        start = (~self.clean_mask) & (self.t == self.onset_step)
        if start.any():
            self._calibrate(start)
        active = ~np.isnan(phase)
        blend = np.nan_to_num(phase)
        cp, cm = self._cam(self.cam_idx)
        lat = self._hold_lat + blend * ENTRY_OFFSET
        track_pos, track_rot, _ = geometry.occluder_frame(
            cp, cm, self.state["cube_pos"], lat, np.full(self.n, DEPTH_FRAC)
        )
        world_pos = self._anchor_pos + (blend * ENTRY_OFFSET)[:, None] * self._anchor_side
        wa = self.world_anchor[:, None]
        pos = np.where(wa, world_pos, track_pos)
        rot = np.where(wa[..., None], self._anchor_rot, track_rot)
        pos = np.where(active[:, None], pos, PARK_POS[None])
        quat = np.where(active[:, None], geometry.mat_to_quat(rot), np.array([[1.0, 0, 0, 0]]))
        self.mocap_pos[:, self.ids.occluder_mocap] = pos
        self.mocap_quat[:, self.ids.occluder_mocap] = quat
        self._active, self._occ_pos, self._occ_rot = active, pos, rot

    def _measure_coverage(self):
        """Per-step coverage in BOTH cameras while the occluder is in the scene."""
        if not self._active.any():
            return
        covs = np.stack(
            [np.nan_to_num(self._coverage(k, self._occ_pos, self._occ_rot, self._active)) for k in range(2)], 1
        )
        r = np.arange(self.n)
        tgt = covs[r, self.cam_idx]
        oth = covs[r, 1 - self.cam_idx]
        both = np.stack([tgt, oth], 1)
        a = self._active
        self._cov_sum[a] += both[a]
        self._cov_max[a] = np.maximum(self._cov_max[a], both[a])
        self._cov_cnt[a] += 1

    @property
    def success_step(self) -> np.ndarray:
        """Control step of first success per env, -1 if never (host copy)."""
        if self._dev is not None:
            return self._dev.success_step.numpy().astype(int)
        return self._success_step

    def coverage_stats(self) -> dict:
        if self._dev is not None:
            return self._dev.stats()
        cnt = np.maximum(self._cov_cnt, 1)[:, None]
        mean = np.where(self._cov_cnt[:, None] > 0, self._cov_sum / cnt, np.nan)
        mx = np.where(self._cov_cnt[:, None] > 0, self._cov_max, np.nan)
        return dict(
            coverage_onset=self.coverage_onset.copy(),
            coverage=mean[:, 0],
            coverage_max=mx[:, 0],
            coverage_other=mean[:, 1],
            coverage_other_max=mx[:, 1],
        )

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray):
        """Action shape (N, robot.action_dim); see occlubench.robots."""
        st = self.state
        if self._dev is not None:
            self._dev.place(self.t)  # writes occluder mocap into Warp Data, accumulates coverage
        else:
            self._place_occluders()
            self._measure_coverage()
        if self._ik is not None:
            self._ik.write(action)  # ctrl computed and written on device
            self._ctrl = None
            if self._dev is None:
                self.backend.write(None, self.mocap_pos, self.mocap_quat)
        else:
            self.ctrl = self.robot.ctrl(action, st["qpos"], st["tcp_pos"], st["tcp_mat"], st["jac"])
            if self._dev is not None:
                self.backend.write(self.ctrl)
            else:
                self.backend.write(self.ctrl, self.mocap_pos, self.mocap_quat)
        self.backend.step(self.substeps)
        self.state = self.backend.refresh(self.robot.needs_jacobian)
        self.t += 1
        done = self.t >= self.horizon
        if self._dev is not None:
            self._dev.score(self.t)
            dev = self._dev
            info = _LazyInfo(
                success=lambda: dev.success.numpy().astype(bool),
                ever_success=lambda: dev.success_step.numpy() >= 0,
                occluder_active=lambda: dev.active.numpy().astype(bool),
            )
            return self._obs(), done, info
        succ = is_success(self.state["cube_pos"], self.goal)
        newly = succ & (self._success_step < 0)
        self._success_step[newly] = self.t
        info = dict(success=succ, ever_success=self._success_step >= 0, occluder_active=self._active.copy())
        return self._obs(), done, info

    @property
    def ctrl(self) -> np.ndarray:
        """Last actuator controls (N, nu). Read back from the device lazily on the device-IK path."""
        if self._ctrl is None:
            self._ctrl = self.backend.d.ctrl.numpy().astype(float)
        return self._ctrl

    @ctrl.setter
    def ctrl(self, v):
        self._ctrl = v

    # ------------------------------------------------------------------ obs
    def occluder_active(self) -> np.ndarray:
        phase = schedule_phase(np.full(self.n, self.t), self.onset_step, self.hold_steps, SLIDE_STEPS)
        return ~np.isnan(phase) & ~self.clean_mask

    def cube_pos(self) -> np.ndarray:
        return self.state["cube_pos"]

    def proprio(self) -> np.ndarray:
        return self.robot.proprio(self.state["qpos"], self.state["tcp_pos"], self.ctrl)

    def privileged(self) -> dict:
        """Ground-truth state. For the scripted oracle and scoring only."""
        return dict(
            cube_pos=self.state["cube_pos"],
            cube_mat=self.state["cube_mat"],
            goal_pos=np.repeat(self.goal[None], self.n, 0),
        )

    def camera_calibration(self, camera: str) -> dict:
        """Intrinsics + per-env extrinsics, as a calibrated real rig would provide."""
        k = CAMERAS.index(camera)
        f = 0.5 * self.h / math.tan(math.radians(self.fovy[k]) / 2)
        return dict(f=f, cx=self.w / 2, cy=self.h / 2, pos=self.state["cam_pos"][:, k], rot=self.state["cam_mat"][:, k])

    def render(self) -> dict:
        if self.render_backend == "none":
            return {}
        if self.render_backend == "warp":
            return self._render_warp()
        out = {c: np.empty((self.n, self.h, self.w, 3), np.uint8) for c in CAMERAS}
        datas = getattr(self.backend, "datas", None)
        if datas is None:  # Warp physics + CPU render: mirror qpos into a host MjData
            datas = self._mirror()
        mp, mq = self.mocap_pos, self.mocap_quat
        if self._dev is not None:  # occluder pose lives on the device; host renderer needs it
            mp, mq = self.backend.d.mocap_pos.numpy(), self.backend.d.mocap_quat.numpy()
        for i, d in enumerate(datas):
            d.mocap_pos[:] = mp[i]
            d.mocap_quat[:] = mq[i]
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_camlight(self.m, d)
            for c in CAMERAS:
                self._renderer.update_scene(d, camera=self.ids.cams[c])
                out[c][i] = self._renderer.render()
        return out

    def _mirror(self):
        if self._host is None:
            self._host = [mujoco.MjData(self.m) for _ in range(self.n)]
        for d, q in zip(self._host, self.state["qpos"]):
            d.qpos[:] = q
        return self._host

    def _render_warp(self) -> dict:
        wb = self.backend
        wb.mjw.refit_bvh(wb.m, wb.d, self._rc)
        wb.mjw.render(wb.m, wb.d, self._rc)
        out = {}
        for k, c in enumerate(CAMERAS):
            wb.mjw.get_rgb(self._rc, k, self._rgb_out)
            rgb = self._rgb_out.numpy()
            if rgb.max() <= 1.0 + 1e-6:
                rgb = rgb * 255.0
            out[c] = np.clip(rgb, 0, 255).astype(np.uint8)
        return out

    def _obs(self) -> dict:
        return dict(rgb=self.render(), proprio=self.proprio(), t=self.t)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
