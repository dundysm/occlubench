"""On-device action -> actuator-control kernels (the robots' IK) for the MuJoCo Warp path.

``DeviceIK`` mirrors ``GantryRobot.ctrl`` / ``PandaRobot.ctrl`` (the NumPy reference in
``occlubench.robots``) as Warp kernels. They read qpos, the tcp site pose and the tcp Jacobian
straight from MuJoCo Warp Data and write ``Data.ctrl`` in place, so the only per-step upload is
the (N, action_dim) action and the Jacobian never leaves the device.

Panda: damped least squares, ``dq = J^T (J J^T + lambda I)^-1 e`` plus the null-space posture
term ``(I - J^+ J) k (q_home - q)``. The 6x6 system is solved with an in-kernel Cholesky
factorisation (it is symmetric positive definite for lambda > 0). Float32 on device vs. float64
in NumPy; ``tests/test_warp_ik.py`` checks parity.
"""

from __future__ import annotations

import numpy as np
import warp as wp

mat67 = wp.types.matrix(shape=(6, 7), dtype=float)
mat66 = wp.types.matrix(shape=(6, 6), dtype=float)
vec6 = wp.types.vector(6, float)
vec7 = wp.types.vector(7, float)


@wp.func
def _euler_to_mat(r: float, p: float, y: float) -> wp.mat33:
    cr, sr, cp, sp, cy, sy = wp.cos(r), wp.sin(r), wp.cos(p), wp.sin(p), wp.cos(y), wp.sin(y)
    return wp.mat33(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )


@wp.func
def _rot_error(rd: wp.mat33, rc: wp.mat33) -> wp.vec3:
    e = wp.vec3(0.0)
    for i in range(3):
        c = wp.vec3(rc[0, i], rc[1, i], rc[2, i])
        d = wp.vec3(rd[0, i], rd[1, i], rd[2, i])
        e = e + wp.cross(c, d)
    return 0.5 * e


@wp.func
def _chol_solve(a: mat66, b: vec6) -> vec6:
    """Solve a x = b for symmetric positive definite a (in-register Cholesky)."""
    lo = mat66()
    for j in range(6):
        s = a[j, j]
        for k in range(j):
            s -= lo[j, k] * lo[j, k]
        ljj = wp.sqrt(wp.max(s, 1.0e-12))
        lo[j, j] = ljj
        for i in range(j + 1, 6):
            t = a[i, j]
            for k in range(j):
                t -= lo[i, k] * lo[j, k]
            lo[i, j] = t / ljj
    y = vec6()
    for i in range(6):
        t = b[i]
        for k in range(i):
            t -= lo[i, k] * y[k]
        y[i] = t / lo[i, i]
    x = vec6()
    for ii in range(6):
        i = 5 - ii
        t = y[i]
        for k in range(i + 1, 6):
            t -= lo[k, i] * x[k]
        x[i] = t / lo[i, i]
    return x


@wp.kernel
def tcp_point(site_xpos: wp.array2d(dtype=wp.vec3), site: int, point: wp.array(dtype=wp.vec3)):
    w = wp.tid()
    point[w] = site_xpos[w, site]


@wp.kernel
def gantry_ctrl(
    action: wp.array2d(dtype=float),
    tcp_z0: float,
    grip_open: float,
    grip_closed: float,
    ctrl: wp.array2d(dtype=float),
):
    w = wp.tid()
    ctrl[w, 0] = wp.clamp(action[w, 0], -0.35, 0.35)
    ctrl[w, 1] = wp.clamp(action[w, 1], -0.35, 0.35)
    ctrl[w, 2] = wp.clamp(action[w, 2] - tcp_z0, -0.33, 0.05)
    g = wp.clamp(action[w, 3], -1.0, 1.0)
    ctrl[w, 3] = grip_open + (grip_closed - grip_open) * (g + 1.0) * 0.5


@wp.kernel
def panda_ctrl(
    action: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    jacp: wp.array3d(dtype=float),
    jacr: wp.array3d(dtype=float),
    site: int,
    qadr: wp.array(dtype=int),
    dadr: wp.array(dtype=int),
    home: wp.array(dtype=float),
    lo: wp.array(dtype=float),
    hi: wp.array(dtype=float),
    damping: float,
    posture_gain: float,
    lead: float,
    max_dq: float,
    grip_hi: float,
    ctrl: wp.array2d(dtype=float),
):
    w = wp.tid()
    tgt = wp.vec3(action[w, 0], action[w, 1], action[w, 2])
    ep = tgt - site_xpos[w, site]
    er = _rot_error(_euler_to_mat(action[w, 3], action[w, 4], action[w, 5]), site_xmat[w, site])
    e = vec6()
    for i in range(3):
        e[i] = wp.clamp(ep[i], -0.05, 0.05)
        e[i + 3] = er[i]
    jm = mat67()
    q = vec7()
    p = vec7()
    for j in range(7):
        dj = dadr[j]
        for i in range(3):
            jm[i, j] = jacp[w, i, dj]
            jm[i + 3, j] = jacr[w, i, dj]
        q[j] = qpos[w, qadr[j]]
        p[j] = posture_gain * (home[j] - q[j])
    a = mat66()
    for i in range(6):
        for k in range(6):
            s = float(0.0)  # noqa: UP018 (Warp needs a typed, mutable loop variable)
            for j in range(7):
                s += jm[i, j] * jm[k, j]
            a[i, k] = s
        a[i, i] = a[i, i] + damping
    # task term: J^T A^-1 e ; null-space posture term: p - J^T A^-1 J p
    jp = vec6()
    for i in range(6):
        s = float(0.0)  # noqa: UP018 (Warp needs a typed, mutable loop variable)
        for j in range(7):
            s += jm[i, j] * p[j]
        jp[i] = s
    y = _chol_solve(a, e)
    z = _chol_solve(a, jp)
    for j in range(7):
        s = float(0.0)  # noqa: UP018 (Warp needs a typed, mutable loop variable)
        for i in range(6):
            s += jm[i, j] * (y[i] - z[i])
        dq = wp.clamp(lead * (s + p[j]), -max_dq, max_dq)
        ctrl[w, j] = wp.clamp(q[j] + dq, lo[j], hi[j])
    g = wp.clamp(action[w, 6], -1.0, 1.0)
    ctrl[w, 7] = grip_hi * (1.0 - g) * 0.5


class DeviceIK:
    """Writes ``Data.ctrl`` from an action on the Warp device. One instance per OccluEnv."""

    def __init__(self, env, backend):
        self.env, self.b, self.robot = env, backend, env.robot
        d = backend.d
        self.dev = d.qpos.device
        self.n = env.n
        self.action = wp.zeros((self.n, self.robot.action_dim), dtype=float, device=self.dev)
        r = self.robot
        if r.name == "panda":

            def f(a):
                return wp.array(np.asarray(a, np.float32), dtype=float, device=self.dev)

            self.qadr = wp.array(r.arm_qadr.astype(np.int32), dtype=int, device=self.dev)
            self.dadr = wp.array(r.arm_dadr.astype(np.int32), dtype=int, device=self.dev)
            self.home, self.lo, self.hi = f(r.HOME), f(r.lo), f(r.hi)

    def write(self, action: np.ndarray):
        """Upload the action and compute ctrl on device (needs kinematics + Jacobian up to date)."""
        self.action.assign(wp.array(np.asarray(action, np.float32), dtype=float, device=self.dev))
        self.run(self.action)

    def run(self, action):
        r, d, b = self.robot, self.b.d, self.b
        site = int(self.env.ids.tcp_site)
        if r.name == "gantry":
            wp.launch(
                gantry_ctrl,
                dim=self.n,
                device=self.dev,
                inputs=[action, r.TCP_Z0, r.GRIP_OPEN, r.GRIP_CLOSED, d.ctrl],
            )
            return
        wp.launch(
            panda_ctrl,
            dim=self.n,
            device=self.dev,
            inputs=[
                action,
                d.qpos,
                d.site_xpos,
                d.site_xmat,
                b.jacp,
                b.jacr,
                site,
                self.qadr,
                self.dadr,
                self.home,
                self.lo,
                self.hi,
                r.DAMPING,
                r.POSTURE_GAIN,
                r.LEAD,
                r.MAX_DQ,
                r.grip_hi,
                d.ctrl,
            ],
        )
