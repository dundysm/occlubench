"""Robots. Each robot attaches itself to the base scene spec and maps actions to actuator controls.

* ``GantryRobot`` (simple, self-authored): action (N, 4) = tcp target xyz + grip in [-1, 1].
* ``PandaRobot`` (Franka Emika Panda from MuJoCo Menagerie, Apache-2.0, fetched at runtime):
  action (N, 7) = tcp target xyz + tcp orientation as roll, pitch, yaw (world frame, radians;
  (pi, 0, yaw) points the gripper straight down) + grip in [-1, 1]. Converted to the 7 joint
  position targets with batched damped-least-squares IK and a null-space posture term.
"""

from __future__ import annotations

from importlib import resources

import mujoco
import numpy as np

PREFIX = "robot/"


def _asset(name: str) -> str:
    return str(resources.files("occlubench") / "assets" / name)


def euler_to_mat(rpy: np.ndarray) -> np.ndarray:
    """(N, 3) roll-pitch-yaw (extrinsic XYZ) -> (N, 3, 3) rotation matrices."""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    m = np.empty((len(rpy), 3, 3))
    m[:, 0] = np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], 1)
    m[:, 1] = np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], 1)
    m[:, 2] = np.stack([-sp, cp * sr, cp * cr], 1)
    return m


def rot_error(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    """Batched axis-angle-ish orientation error (N, 3): 0.5 * sum_i cross(r_cur[:, i], r_des[:, i])."""
    return 0.5 * np.cross(r_cur.transpose(0, 2, 1), r_des.transpose(0, 2, 1)).sum(1)


class GantryRobot:
    name = "gantry"
    action_dim = 4
    tcp_site = PREFIX + "tcp"
    wrist_cam = PREFIX + "wrist"
    TCP_Z0 = 0.68
    GRIP_OPEN, GRIP_CLOSED = 0.012, -0.028
    needs_jacobian = False
    grasp_dz = 0.004  # tcp height above the cube centre when grasping
    carry_scale = 1.0  # speed multiplier while carrying the cube

    def attach(self, spec: mujoco.MjSpec):
        child = mujoco.MjSpec.from_file(_asset("gantry.xml"))
        spec.attach(child, prefix=PREFIX, frame=spec.worldbody.add_frame())

    def setup(self, m: mujoco.MjModel):
        self.qadr = np.array([m.jnt_qposadr[m.joint(PREFIX + j).id] for j in ("arm_x", "arm_y", "arm_z", "finger_l")])
        self.nu = m.nu

    def home(self, m: mujoco.MjModel, d: mujoco.MjData):
        d.ctrl[:] = [0.0, 0.0, 0.0, self.GRIP_OPEN]

    def ctrl(self, action, qpos, tcp_pos, tcp_mat, jac=None):
        a = np.asarray(action, float)
        c = np.empty((len(a), self.nu))
        c[:, 0] = np.clip(a[:, 0], -0.35, 0.35)
        c[:, 1] = np.clip(a[:, 1], -0.35, 0.35)
        c[:, 2] = np.clip(a[:, 2] - self.TCP_Z0, -0.33, 0.05)
        g = np.clip(a[:, 3], -1, 1)
        c[:, 3] = self.GRIP_OPEN + (self.GRIP_CLOSED - self.GRIP_OPEN) * (g + 1) / 2
        return c

    def proprio(self, qpos, tcp_pos, ctrl):
        width = qpos[:, self.qadr[3]]
        g = (ctrl[:, 3] - self.GRIP_OPEN) / (self.GRIP_CLOSED - self.GRIP_OPEN)
        return np.concatenate([tcp_pos, width[:, None], g[:, None]], 1)

    def default_action(self, tcp_pos, grip=-1.0, yaw=None):
        return np.concatenate([tcp_pos, np.full((len(tcp_pos), 1), grip)], 1)


class PandaRobot:
    name = "panda"
    action_dim = 7
    tcp_site = PREFIX + "tcp"
    wrist_cam = PREFIX + "wrist"
    needs_jacobian = True
    grasp_dz = -0.008
    carry_scale = 0.5
    BASE_POS = (-0.52, 0.0, 0.40)
    HOME = np.array([0.0, 0.1, 0.0, -2.0, 0.0, 2.1, 0.785])
    DAMPING = 1e-2
    MAX_DQ = 0.25
    POSTURE_GAIN = 0.2
    LEAD = 3.0  # position targets lead the IK step so the joint servos track at task speed

    def attach(self, spec: mujoco.MjSpec):
        from occlubench.menagerie import panda_dir

        pdir = panda_dir()
        child = mujoco.MjSpec.from_file(str(pdir / "panda.xml"))
        child.meshdir = str(pdir / "assets")
        hand = child.body("hand")
        # TCP between the fingertips (hand frame z points from the hand towards the fingers).
        hand.add_site(name="tcp", pos=[0, 0, 0.105], size=[0.005, 0, 0], rgba=[0, 0, 0, 0])
        # Wrist camera on the hand, looking along +z towards the fingers, offset to see past them.
        hand.add_camera(name="wrist", pos=[0.0, -0.075, 0.03], xyaxes=[1, 0, 0, 0, -0.94, 0.34], fovy=80)
        # Pedestal under the robot base.
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[self.BASE_POS[0], 0, 0.2],
            size=[0.09, 0.2, 0],
            rgba=[0.3, 0.3, 0.32, 1],
        )
        frame = spec.worldbody.add_frame(pos=list(self.BASE_POS))
        spec.attach(child, prefix=PREFIX, frame=frame)

    def setup(self, m: mujoco.MjModel):
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"{PREFIX}joint{i}").id] for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(f"{PREFIX}joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = m.jnt_qposadr[m.joint(PREFIX + "finger_joint1").id]
        self.lo = np.array([m.actuator_ctrlrange[m.actuator(f"{PREFIX}actuator{i}").id][0] for i in range(1, 8)])
        self.hi = np.array([m.actuator_ctrlrange[m.actuator(f"{PREFIX}actuator{i}").id][1] for i in range(1, 8)])
        self.grip_hi = float(m.actuator_ctrlrange[m.actuator(PREFIX + "actuator8").id][1])  # fully open
        self.nu = m.nu

    def home(self, m: mujoco.MjModel, d: mujoco.MjData):
        d.qpos[self.arm_qadr] = self.HOME
        d.qpos[self.finger_qadr] = 0.04
        d.qpos[self.finger_qadr + 1] = 0.04
        d.ctrl[:7] = self.HOME
        d.ctrl[7] = self.grip_hi

    def ctrl(self, action, qpos, tcp_pos, tcp_mat, jac):
        """Batched DLS IK step. jac: (N, 6, nv) [jacp; jacr] at the tcp."""
        a = np.asarray(action, float)
        n = len(a)
        err = np.concatenate([a[:, :3] - tcp_pos, rot_error(euler_to_mat(a[:, 3:6]), tcp_mat)], 1)
        err[:, :3] = np.clip(err[:, :3], -0.05, 0.05)
        j = jac[:, :, self.arm_dadr]  # (N, 6, 7)
        jjt = j @ j.transpose(0, 2, 1) + self.DAMPING * np.eye(6)[None]
        jpinv = j.transpose(0, 2, 1) @ np.linalg.inv(jjt)  # (N, 7, 6)
        q = qpos[:, self.arm_qadr]
        dq = (jpinv @ err[:, :, None])[:, :, 0]
        null = np.eye(7)[None] - jpinv @ j
        dq += (null @ (self.POSTURE_GAIN * (self.HOME - q))[:, :, None])[:, :, 0]
        dq = np.clip(self.LEAD * dq, -self.MAX_DQ, self.MAX_DQ)
        c = np.empty((n, self.nu))
        c[:, :7] = np.clip(q + dq, self.lo, self.hi)
        g = np.clip(a[:, 6], -1, 1)
        c[:, 7] = self.grip_hi * (1 - g) / 2
        return c

    def proprio(self, qpos, tcp_pos, ctrl):
        width = qpos[:, self.finger_qadr]
        g = 1 - ctrl[:, 7] / self.grip_hi
        return np.concatenate([tcp_pos, width[:, None], g[:, None], qpos[:, self.arm_qadr]], 1)

    def default_action(self, tcp_pos, grip=-1.0, yaw=None):
        n = len(tcp_pos)
        yaw = np.zeros(n) if yaw is None else np.broadcast_to(yaw, (n,))
        rpy = np.stack([np.full(n, np.pi), np.zeros(n), yaw], 1)
        return np.concatenate([tcp_pos, rpy, np.full((n, 1), grip)], 1)


ROBOTS = {"gantry": GantryRobot, "panda": PandaRobot}


def make_robot(name: str):
    return ROBOTS[name]()
