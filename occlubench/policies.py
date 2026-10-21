"""Reference policies.

* ``ScriptedOracle`` uses privileged state. It is blind to occlusion by construction and
  serves as a pipeline sanity check (paired drop should be ~0).
* ``ColorCuePolicy`` is a non-learned, pixels-only baseline: it localises the red cube in
  one camera by colour thresholding and back-projection, then runs the same pick-and-place
  script. It is affected by occlusion of the camera it relies on, which makes it a
  meaningful smoke test for the camera axis of the grid.
* ``OpenVLAOFTPolicy`` is an interface stub for OpenVLA-OFT (no weights bundled).
"""

from __future__ import annotations

import numpy as np

from occlubench.scene import CUBE_HALF, TABLE_TOP_Z

APPROACH_H = 0.12
LIFT_Z = 0.62
PLACE_Z = TABLE_TOP_Z + CUBE_HALF + 0.03
GRASP_Z = TABLE_TOP_Z + CUBE_HALF + 0.004


class _PickPlaceScript:
    """Phase machine shared by the scripted policies. Phases advance when the tcp arrives."""

    def __init__(self, n: int, tol: float = 0.012, max_step: float = 0.03):
        self.n, self.tol, self.max_step = n, tol, max_step
        self.phase = np.zeros(n, int)
        self.timer = np.zeros(n, int)

    def act(self, tcp, cube, goal, cube_known):
        acts = np.zeros((self.n, 4))
        for i in range(self.n):
            p, c, g = self.phase[i], cube[i], goal[i]
            grip = -1.0
            if p == 0:  # above cube
                tgt = np.array([c[0], c[1], c[2] + APPROACH_H]) if cube_known[i] else tcp[i]
            elif p == 1:  # descend
                tgt = np.array([c[0], c[1], GRASP_Z])
            elif p == 2:  # close
                tgt, grip = np.array([c[0], c[1], GRASP_Z]), 1.0
            elif p == 3:  # lift
                tgt, grip = np.array([tcp[i][0], tcp[i][1], LIFT_Z]), 1.0
            elif p == 4:  # over goal
                tgt, grip = np.array([g[0], g[1], LIFT_Z]), 1.0
            elif p == 5:  # lower
                tgt, grip = np.array([g[0], g[1], PLACE_Z]), 1.0
            else:  # release and retreat
                tgt = np.array([g[0], g[1], LIFT_Z])
            step = tgt - tcp[i]
            nrm = np.linalg.norm(step)
            cmd = tcp[i] + step * min(1.0, self.max_step / nrm) if nrm > 1e-9 else tgt
            acts[i, :3], acts[i, 3] = cmd, grip
            arrived = np.linalg.norm(tcp[i] - tgt) < self.tol
            if p == 0 and not cube_known[i]:
                arrived = False
            self.timer[i] += 1
            if p == 2 and self.timer[i] >= 12:
                self._next(i)
            elif p in (0, 1, 3, 4, 5) and arrived:
                self._next(i)
        return acts

    def _next(self, i):
        self.phase[i] += 1
        self.timer[i] = 0


class ScriptedOracle:
    uses_privileged_state = True

    def reset(self, n: int):
        self.script = _PickPlaceScript(n)

    def act(self, obs, env):
        st = env.privileged()
        tcp = obs["proprio"][:, :3]
        return self.script.act(tcp, st["cube_pos"], st["goal_pos"], np.ones(len(tcp), bool))


class ColorCuePolicy:
    """Pixels + proprio only. Re-localises the cube every step while approaching it.

    memory=False: if the cube is not visible it holds position (no object permanence).
    memory=True:  it keeps the last estimate while the cube is hidden.
    """

    uses_privileged_state = False

    def __init__(self, camera: str = "third_person", memory: bool = False, goal=(0.18, 0.15, 0.401)):
        self.camera, self.memory, self.goal = camera, memory, np.asarray(goal)

    def reset(self, n: int):
        self.script = _PickPlaceScript(n)
        self.last = np.full((n, 3), np.nan)

    def _localise(self, rgb, calib):
        n = rgb.shape[0]
        est = np.full((n, 3), np.nan)
        r, g, b = (rgb[..., k].astype(int) for k in range(3))
        mask = (r > 120) & (g < 70) & (b < 70)
        z_plane = TABLE_TOP_Z + CUBE_HALF
        for i in range(n):
            vs, us = np.nonzero(mask[i])
            if len(us) < 3:
                continue
            u, v = us.mean() + 0.5, vs.mean() + 0.5
            d_cam = np.array([(u - calib["cx"]) / calib["f"], -(v - calib["cy"]) / calib["f"], -1.0])
            d = calib["rot"][i] @ d_cam
            o = calib["pos"][i]
            if abs(d[2]) < 1e-6:
                continue
            s = (z_plane - o[2]) / d[2]
            if s > 0:
                est[i] = o + s * d
        return est

    def act(self, obs, env):
        tcp = obs["proprio"][:, :3]
        est = self._localise(obs["rgb"][self.camera], env.camera_calibration(self.camera))
        seen = ~np.isnan(est[:, 0])
        # Re-localise only while approaching; during descent the arm itself hides the cube
        # from the third-person view, and after the grasp the cube moves with the hand.
        pre_grasp = self.script.phase == 0
        self.last[seen & pre_grasp] = est[seen & pre_grasp]
        known = ~np.isnan(self.last[:, 0]) & (seen | self.memory | ~pre_grasp)
        cube = np.where(np.isnan(self.last), tcp, self.last)
        goal = np.repeat(self.goal[None], len(tcp), 0)
        acts = self.script.act(tcp, cube, goal, known)
        hold = pre_grasp & ~known
        acts[hold, :3] = tcp[hold]
        return acts


class OpenVLAOFTPolicy:
    """Interface stub for OpenVLA-OFT (https://github.com/moojink/openvla-oft).

    Not functional in v0: no weights are bundled and the model is not a dependency.
    Intended mapping: third_person -> "full_image", wrist -> "wrist_image" (both resized
    to 224x224), proprio -> "state"; the predicted end-effector delta chunk is integrated
    into OccluBench's absolute tcp target + gripper action. Fine-tune on OccluBench demos
    first (the released LIBERO checkpoints target a different robot and action space).
    """

    uses_privileged_state = False

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        self.checkpoint, self.device = checkpoint, device

    def reset(self, n: int):
        self.n = n

    def act(self, obs, env):
        raise NotImplementedError("OpenVLA-OFT adapter is a stub in v0; see the docstring for the planned mapping.")
