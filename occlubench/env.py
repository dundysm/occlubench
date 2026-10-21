"""Batched OccluBench environment.

Physics backends:  "cpu" (one mujoco.MjData per env) or "warp" (MuJoCo Warp, nworld = n_envs).
Render backends:   "cpu" (mujoco.Renderer, OSMesa/EGL), "warp" (MuJoCo Warp batch ray tracer),
                   or "none" (state-only, for physics throughput tests).

Policies see only RGB from the ``third_person`` and ``wrist`` cameras plus proprioception.
Privileged state is exposed separately (``env.privileged()``) for the scripted oracle and
for scoring; benchmark submissions must not use it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np

from occlubench.conditions import CAMERAS, CLEAN, Condition
from occlubench.occluder import (
    ENTRY_OFFSET,
    CoverageMeter,
    occluder_pose,
    park_occluder,
    schedule_phase,
    set_occluder,
)
from occlubench.scene import (
    CUBE_HALF,
    GOAL_RADIUS,
    SPAWN_HIGH,
    SPAWN_LOW,
    TABLE_TOP_Z,
    SceneIds,
    load_model,
)

TCP_Z0 = 0.68  # world z of the tcp site when the z slide is at 0
GRIP_OPEN, GRIP_CLOSED = 0.012, -0.028
SLIDE_STEPS = 8


def is_success(cube_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """Cube resting on the table within GOAL_RADIUS of the goal centre. Shapes (N, 3)."""
    dxy = np.linalg.norm(cube_pos[:, :2] - goal_pos[None, :2], axis=1)
    resting = cube_pos[:, 2] < TABLE_TOP_Z + CUBE_HALF + 0.01
    return (dxy < GOAL_RADIUS) & resting


class OccluEnv:
    def __init__(
        self,
        n_envs: int = 8,
        horizon: int = 200,
        image_size: tuple[int, int] = (96, 96),
        physics: str = "cpu",
        render: str = "cpu",
        substeps: int = 10,
        coverage_res: int = 64,
    ):
        self.n, self.horizon, self.substeps = n_envs, horizon, substeps
        self.h, self.w = image_size
        self.physics_backend, self.render_backend = physics, render
        self.m = load_model()
        self.ids = SceneIds.from_model(self.m)
        self.datas = [mujoco.MjData(self.m) for _ in range(n_envs)]
        self.goal = None
        self.meter = CoverageMeter(self.m, self.ids, coverage_res, coverage_res)
        self.conditions: list[Condition] = [CLEAN] * n_envs
        self._renderer = None
        self._warp = None
        if physics == "warp" or render == "warp":
            self._init_warp()
        if render == "cpu":
            self._renderer = mujoco.Renderer(self.m, self.h, self.w)

    # ------------------------------------------------------------------ warp
    def _init_warp(self):
        import mujoco_warp as mjw
        import warp as wp

        wp.init()
        mw = mjw.put_model(self.m)
        dw = mjw.put_data(self.m, self.datas[0], nworld=self.n)
        rc = None
        if self.render_backend == "warp":
            rc = mjw.create_render_context(
                self.m,
                nworld=self.n,
                cam_res=(self.w, self.h),
                render_rgb=True,
                cam_active=list(CAMERAS),
            )
            self._rgb_out = wp.zeros((self.n, self.h, self.w), dtype=wp.vec3)
        self._warp = dict(mjw=mjw, wp=wp, m=mw, d=dw, rc=rc)

    def _push_to_warp(self, fields=("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "act")):
        wp = self._warp["wp"]
        dw = self._warp["d"]
        for f in fields:
            host = np.stack([getattr(d, f) for d in self.datas])
            if host.size == 0:
                continue
            arr = getattr(dw, f)
            ref = arr.numpy()  # host copy gives the full shape incl. vector/quaternion dims
            arr.assign(wp.array(host.reshape(ref.shape).astype(ref.dtype), dtype=arr.dtype, device=arr.device))

    def _pull_from_warp(self):
        dw = self._warp["d"]
        qpos, qvel = dw.qpos.numpy(), dw.qvel.numpy()
        for i, d in enumerate(self.datas):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_camlight(self.m, d)

    # ------------------------------------------------------------------ api
    def reset(self, seeds: Sequence[int], conditions: Sequence[Condition] | None = None):
        assert len(seeds) == self.n, "one seed per env"
        self.seeds = list(seeds)
        self.conditions = list(conditions) if conditions is not None else [CLEAN] * self.n
        self.t = 0
        self.success_step = np.full(self.n, -1)
        self.coverage = np.full(self.n, np.nan)
        self.coverage_other = np.full(self.n, np.nan)
        self._hold_lat = np.zeros(self.n)
        for d, s in zip(self.datas, self.seeds):
            mujoco.mj_resetData(self.m, d)
            rng = np.random.default_rng(s)
            xy = rng.uniform(SPAWN_LOW, SPAWN_HIGH)
            a = self.ids.cube_qpos
            d.qpos[a : a + 3] = [xy[0], xy[1], TABLE_TOP_Z + CUBE_HALF]
            yaw = rng.uniform(-math.pi / 4, math.pi / 4)
            d.qpos[a + 3 : a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.ctrl[:] = [0.0, 0.0, 0.0, GRIP_OPEN]
            park_occluder(d, self.ids)
            mujoco.mj_forward(self.m, d)
        self.goal = self.datas[0].site_xpos[self.ids.goal_site].copy()
        if self._warp is not None:
            self._push_to_warp()
        return self._obs()

    def _onset_step(self, c: Condition) -> int:
        return int(round(c.onset * self.horizon))

    def _update_occluders(self):
        for i, (d, c) in enumerate(zip(self.datas, self.conditions)):
            if c.is_clean:
                continue
            on = self._onset_step(c)
            hold = max(1, int(round(c.duration * self.horizon)))
            if self.t == on:
                lat, achieved = self.meter.calibrate(d, c.camera, c.severity)
                self._hold_lat[i], self.coverage[i] = lat, achieved
                # A physical occluder can also block the other camera; measure that leakage.
                other = CAMERAS[1 - CAMERAS.index(c.camera)]
                pos, quat = occluder_pose(d, self.ids, c.camera, lat)
                set_occluder(d, self.ids, pos, quat)
                self.coverage_other[i] = self.meter.coverage(d, other)
            phase = schedule_phase(self.t, on, hold, SLIDE_STEPS)
            if math.isnan(phase):
                park_occluder(d, self.ids)
            else:
                pos, quat = occluder_pose(d, self.ids, c.camera, self._hold_lat[i] + phase * ENTRY_OFFSET)
                set_occluder(d, self.ids, pos, quat)

    def step(self, action: np.ndarray):
        """action: (N, 4) = absolute tcp target (x, y, z) in world metres + grip in [-1, 1] (+1 closes)."""
        action = np.asarray(action, dtype=np.float64)
        self._update_occluders()
        for d, a in zip(self.datas, action):
            d.ctrl[0] = np.clip(a[0], -0.35, 0.35)
            d.ctrl[1] = np.clip(a[1], -0.35, 0.35)
            d.ctrl[2] = np.clip(a[2] - TCP_Z0, -0.33, 0.05)
            g = np.clip(a[3], -1, 1)
            d.ctrl[3] = GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * (g + 1) / 2
        if self._warp is None:
            for d in self.datas:
                for _ in range(self.substeps):
                    mujoco.mj_step(self.m, d)
        else:
            self._push_to_warp(("ctrl", "mocap_pos", "mocap_quat"))
            mjw = self._warp["mjw"]
            for _ in range(self.substeps):
                mjw.step(self._warp["m"], self._warp["d"])
            self._pull_from_warp()
        self.t += 1
        succ = is_success(self.cube_pos(), self.goal)
        newly = succ & (self.success_step < 0)
        self.success_step[newly] = self.t
        done = self.t >= self.horizon
        info = dict(success=succ, ever_success=self.success_step >= 0, occluder_active=self.occluder_active())
        return self._obs(), done, info

    # ------------------------------------------------------------------ obs
    def occluder_active(self) -> np.ndarray:
        out = np.zeros(self.n, bool)
        for i, c in enumerate(self.conditions):
            if not c.is_clean:
                hold = max(1, int(round(c.duration * self.horizon)))
                out[i] = not math.isnan(schedule_phase(self.t, self._onset_step(c), hold, SLIDE_STEPS))
        return out

    def cube_pos(self) -> np.ndarray:
        return np.stack([d.xpos[self.ids.cube_body] for d in self.datas])

    def proprio(self) -> np.ndarray:
        """(N, 5): tcp xyz, finger opening (m), commanded grip in [0, 1]."""
        rows = []
        for d in self.datas:
            tcp = d.site_xpos[self.ids.tcp_site]
            g = (d.ctrl[3] - GRIP_OPEN) / (GRIP_CLOSED - GRIP_OPEN)
            rows.append([*tcp, d.qpos[3], g])
        return np.asarray(rows)

    def privileged(self) -> dict:
        """Ground-truth state. For the scripted oracle and scoring only."""
        return dict(cube_pos=self.cube_pos(), goal_pos=np.repeat(self.goal[None], self.n, 0))

    def camera_calibration(self, camera: str) -> dict:
        """Intrinsics + per-env extrinsics, as a real calibrated rig would provide."""
        cid = self.ids.cams[camera]
        fovy = math.radians(self.m.cam_fovy[cid])
        f = 0.5 * self.h / math.tan(fovy / 2)
        return dict(
            f=f,
            cx=self.w / 2,
            cy=self.h / 2,
            pos=np.stack([d.cam_xpos[cid] for d in self.datas]),
            rot=np.stack([d.cam_xmat[cid].reshape(3, 3) for d in self.datas]),
        )

    def render(self) -> dict:
        if self.render_backend == "none":
            return {}
        if self.render_backend == "cpu":
            out = {c: np.empty((self.n, self.h, self.w, 3), np.uint8) for c in CAMERAS}
            for i, d in enumerate(self.datas):
                for c in CAMERAS:
                    self._renderer.update_scene(d, camera=c)
                    out[c][i] = self._renderer.render()
            return out
        return self._render_warp()

    def _render_warp(self) -> dict:
        w = self._warp
        if self.physics_backend != "warp":
            self._push_to_warp()
            w["mjw"].forward(w["m"], w["d"])
        else:
            self._push_to_warp(("mocap_pos", "mocap_quat"))
            w["mjw"].kinematics(w["m"], w["d"])
        w["mjw"].refit_bvh(w["m"], w["d"], w["rc"])
        w["mjw"].render(w["m"], w["d"], w["rc"])
        out = {}
        for k, c in enumerate(CAMERAS):
            w["mjw"].get_rgb(w["rc"], k, self._rgb_out)
            rgb = self._rgb_out.numpy()
            if rgb.max() <= 1.0 + 1e-6:
                rgb = rgb * 255.0
            out[c] = np.clip(rgb, 0, 255).astype(np.uint8)
        return out

    def _obs(self) -> dict:
        return dict(rgb=self.render(), proprio=self.proprio(), t=self.t)

    def close(self):
        self.meter.close()
        if self._renderer is not None:
            self._renderer.close()
