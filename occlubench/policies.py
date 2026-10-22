"""Reference policies.

* ``ScriptedOracle`` uses privileged state. It is blind to occlusion by construction and
  serves as a pipeline sanity check (paired drop should be ~0).
* ``ColorCuePolicy`` is a non-learned, pixels-only baseline: it localises the red cube in
  one camera by colour thresholding and back-projection, then runs the same pick-and-place
  script. It is affected by occlusion of the camera it relies on, which makes it a
  meaningful smoke test for the camera axis of the grid.
* ``OpenVLAOFTPolicy`` adapts OpenVLA-OFT (lazy HF / openvla-oft import, mock mode for CI).
"""

from __future__ import annotations

import numpy as np

from occlubench.scene import CUBE_HALF, TABLE_TOP_Z

APPROACH_H = 0.12
LIFT_Z = 0.58
PLACE_Z = TABLE_TOP_Z + CUBE_HALF + 0.03
CUBE_Z = TABLE_TOP_Z + CUBE_HALF  # grasp height = CUBE_Z + robot.grasp_dz


class _PickPlaceScript:
    """Phase machine shared by the scripted policies. Phases advance when the tcp arrives."""

    def __init__(self, n: int, tol: float = 0.012, max_step: float = 0.03, carry_scale: float = 1.0):
        self.n, self.tol, self.max_step, self.carry_scale = n, tol, max_step, carry_scale
        self.phase = np.zeros(n, int)
        self.timer = np.zeros(n, int)

    def act(self, tcp, cube, goal, cube_known, robot, yaw=None):
        acts = np.zeros((self.n, 4))
        grasp_z = CUBE_Z + robot.grasp_dz
        for i in range(self.n):
            p, c, g = self.phase[i], cube[i], goal[i]
            grip = -1.0
            if p == 0:  # above cube
                tgt = np.array([c[0], c[1], c[2] + APPROACH_H]) if cube_known[i] else tcp[i]
            elif p == 1:  # descend
                tgt = np.array([c[0], c[1], grasp_z])
            elif p == 2:  # close
                tgt, grip = np.array([c[0], c[1], grasp_z]), 1.0
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
            lim = self.max_step * (self.carry_scale if p in (3, 4, 5) else 1.0)
            cmd = tcp[i] + step * min(1.0, lim / nrm) if nrm > 1e-9 else tgt
            acts[i, :3], acts[i, 3] = cmd, grip
            arrived = np.linalg.norm(tcp[i] - tgt) < self.tol
            if p == 0 and not cube_known[i]:
                arrived = False
            self.timer[i] += 1
            if p == 2 and self.timer[i] >= 12:
                self._next(i)
            elif p in (0, 1, 3, 4, 5) and arrived:
                self._next(i)

        yaw = np.zeros(self.n) if yaw is None else yaw
        out = robot.default_action(acts[:, :3], 0.0, yaw)
        out[:, -1] = acts[:, 3]
        return out

    def _next(self, i):
        self.phase[i] += 1
        self.timer[i] = 0


class ScriptedOracle:
    uses_privileged_state = True
    drop_dist = 0.05  # cube further than this from the tcp while carrying -> re-grasp

    def reset(self, n: int):
        self.script = _PickPlaceScript(n)
        self._yaw = np.zeros(n)

    def act(self, obs, env):
        self.script.carry_scale = env.robot.carry_scale
        st = env.privileged()
        tcp = obs["proprio"][:, :3]
        # Safety net: re-grasp if the cube ever slips out of the hand while carrying; with
        # privileged state the oracle can simply start the pick again.
        sc = self.script
        dropped = np.isin(sc.phase, (3, 4, 5)) & (np.linalg.norm(st["cube_pos"] - tcp, axis=1) > self.drop_dist)
        sc.phase[dropped], sc.timer[dropped] = 0, 0
        # Grasp yaw aligned with the cube (square cube: symmetric under 90 degrees).
        m = st["cube_mat"]
        yaw = np.arctan2(m[:, 1, 0], m[:, 0, 0])
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        # Latch the grasp yaw once the fingers start closing. Re-reading it from the cube while
        # carrying let a small in-hand rotation cross the +-45 degree wrap, which snapped the
        # commanded yaw by 90 degrees mid-carry and shook the cube loose (the 2/64 Panda failures in v0.2).
        track = np.isin(self.script.phase, (0, 1))
        self._yaw[track] = yaw[track]
        yaw = self._yaw
        return self.script.act(tcp, st["cube_pos"], st["goal_pos"], np.ones(len(tcp), bool), env.robot, yaw)


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
        self.script.carry_scale = env.robot.carry_scale
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
        acts = self.script.act(tcp, cube, goal, known, env.robot)
        hold = pre_grasp & ~known
        acts[hold, :3] = tcp[hold]
        return acts


def unnormalize_actions(a, q01, q99, mask=None):
    """OpenVLA / OpenVLA-OFT bounds unnormalisation: [-1, 1] -> [q01, q99] on masked dims."""
    a = np.asarray(a, float)
    q01, q99 = np.asarray(q01, float), np.asarray(q99, float)
    mask = np.ones_like(q01, bool) if mask is None else np.asarray(mask, bool)
    return np.where(mask, 0.5 * (a + 1.0) * (q99 - q01) + q01, a)


class OpenVLAOFTPolicy:
    """OpenVLA-OFT adapter (https://github.com/moojink/openvla-oft).

    Observation mapping: ``third_person`` -> ``full_image`` and ``wrist`` -> ``wrist_image``
    (resized to ``image_size``; the ``oft`` backend uses openvla-oft's own JPEG + Lanczos
    ``resize_image_for_policy`` so inputs match its training pipeline), ``obs["eef_state"]`` ->
    8-D ``state`` laid out like LIBERO (tcp xyz, tcp axis-angle, two finger joint positions;
    see ``OccluEnv.eef_state``). The model predicts a chunk of ``chunk`` end-effector
    deltas ``(dx, dy, dz, droll, dpitch, dyaw, gripper)``. The adapter executes the first
    ``open_loop_steps`` of each chunk, integrating the position delta (times ``delta_scale``)
    into OccluBench's absolute tcp target and the yaw delta into the Panda's yaw target.
    The gripper goes from the model's [0, 1] (1 = open) to OccluBench's [-1, 1] (+1 = close).

    backend:
      * ``"mock"``: no model. Deterministic normalised chunks, unnormalised with ``norm_stats``.
        Used by CI to exercise chunking, unnormalisation and action conversion.
      * ``"hf"``: Hugging Face ``transformers`` (lazy import) ``AutoModelForVision2Seq`` with
        ``trust_remote_code=True``; calls ``model.predict_action``. If ``outputs_normalized``
        the adapter unnormalises with ``norm_stats`` (or ``model.norm_stats[unnorm_key]``).
      * ``"oft"``: the openvla-oft repo's own loaders (``experiments.robot.openvla_utils``:
        ``get_vla``, ``get_processor``, ``get_action_head``, ``get_proprio_projector``,
        ``get_vla_action``), which add the L1 action head and proprio projector the OFT
        checkpoints need. The repo must be installed (``pip install -e openvla-oft``).

    Released checkpoints target LIBERO's Franka, cameras and action space, so expect near-zero
    success zero-shot; fine-tune on OccluBench demos first. See docs/openvla_oft.md.
    The ``hf`` and ``oft`` backends have not yet been run end to end on a GPU by this project.
    """

    uses_privileged_state = False

    def __init__(
        self,
        model_id: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
        backend: str = "hf",
        device: str = "cuda",
        instruction: str = "pick up the red cube and place it on the green target",
        unnorm_key: str | None = "libero_spatial_no_noops",
        norm_stats: dict | None = None,
        chunk: int = 8,
        open_loop_steps: int | None = None,
        action_dim: int = 7,
        image_size: int = 224,
        delta_scale: float = 1.0,
        outputs_normalized: bool = False,
        invert_gripper: bool = True,
        binarize_gripper: bool = True,
        seed: int = 0,
    ):
        if backend not in ("mock", "hf", "oft"):
            raise ValueError(f"unknown backend {backend!r}")
        self.model_id, self.backend, self.device = model_id, backend, device
        self.instruction, self.unnorm_key = instruction, unnorm_key
        self.norm_stats = norm_stats
        self.chunk, self.action_dim, self.image_size = chunk, action_dim, image_size
        self.open_loop_steps = open_loop_steps or chunk
        self.delta_scale, self.outputs_normalized, self.invert_gripper = delta_scale, outputs_normalized, invert_gripper
        self.binarize_gripper = binarize_gripper
        self.seed = seed
        self._model = None
        if backend == "mock" and norm_stats is None:
            self.norm_stats = dict(q01=[-0.01] * 6 + [0.0], q99=[0.01] * 6 + [1.0], mask=[True] * 6 + [False])

    # ---------------------------------------------------------------- model loading (lazy)
    def _load(self):
        if self._model is not None or self.backend == "mock":
            return
        if self.backend == "hf":
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            ).to(self.device)
            if self.norm_stats is None and self.outputs_normalized:
                self.norm_stats = self._model.norm_stats[self.unnorm_key]["action"]
        else:  # oft
            from types import SimpleNamespace

            from experiments.robot import openvla_utils as ou

            # Fields read by openvla_utils.get_vla / get_action_head / get_proprio_projector /
            # get_vla_action (checked against moojink/openvla-oft main, experiments/robot/*.py).
            # Note: openvla-oft places the model on its module-level DEVICE (cuda:0), not self.device.
            self._cfg = SimpleNamespace(
                model_family="openvla",
                pretrained_checkpoint=self.model_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_images_in_input=2,
                use_proprio=True,
                center_crop=True,
                num_open_loop_steps=self.open_loop_steps,
                unnorm_key=self.unnorm_key,
                load_in_8bit=False,
                load_in_4bit=False,
                lora_rank=32,
                num_diffusion_steps_train=50,
                num_diffusion_steps_inference=50,
            )
            self._model = ou.get_vla(self._cfg)
            # Same fallback as experiments/robot/libero/run_libero_eval.py::validate_config.
            stats = self._model.norm_stats
            if self.unnorm_key not in stats and f"{self.unnorm_key}_no_noops" in stats:
                self._cfg.unnorm_key = self.unnorm_key = f"{self.unnorm_key}_no_noops"
            if self.unnorm_key not in stats:
                raise KeyError(f"unnorm_key {self.unnorm_key!r} not in checkpoint stats {sorted(stats)}")
            self._processor = ou.get_processor(self._cfg)
            self._action_head = ou.get_action_head(self._cfg, self._model.llm_dim)
            self._proprio = ou.get_proprio_projector(self._cfg, self._model.llm_dim, proprio_dim=8)
            self._ou = ou

    # ---------------------------------------------------------------- inference
    def _state(self, eef_state):
        s = np.asarray(eef_state, float)
        assert s.shape == (8,), "expects OccluEnv.eef_state(): xyz, axis-angle, 2 finger joints"
        return s

    def _resize(self, img):
        if self.backend == "oft":
            return self._ou.resize_image_for_policy(img, self.image_size)
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((self.image_size, self.image_size), Image.BILINEAR))

    def predict_chunk(self, full_image, wrist_image, eef_state) -> np.ndarray:
        """One env. Returns (chunk, action_dim) unnormalised end-effector deltas + gripper in [0, 1]."""
        if self.backend == "mock":
            k = np.arange(self.chunk)[:, None]
            j = np.arange(self.action_dim)[None]
            raw = np.sin(0.7 * k + 1.3 * j + self.seed + self._calls)
            self._calls += 1
            return unnormalize_actions(raw, **self.norm_stats)
        self._load()
        full, wrist = self._resize(full_image), self._resize(wrist_image)
        if self.backend == "oft":
            obs = dict(full_image=full, wrist_image=wrist, state=self._state(eef_state))
            acts = self._ou.get_vla_action(
                self._cfg,
                self._model,
                self._processor,
                obs,
                self.instruction,
                action_head=self._action_head,
                proprio_projector=self._proprio,
                use_film=False,
            )
            return np.asarray(acts, float).reshape(-1, self.action_dim)
        import torch
        from PIL import Image

        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = self._processor(prompt, Image.fromarray(full).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        out = self._model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        out = out[0] if isinstance(out, tuple) else out
        a = np.asarray(out, float).reshape(-1, self.action_dim)
        if self.outputs_normalized:
            a = unnormalize_actions(
                a, **{k: self.norm_stats[k] for k in ("q01", "q99")}, mask=self.norm_stats.get("mask")
            )
        return a

    def to_env_action(self, delta, tcp, yaw, robot):
        """(N, 7) model deltas -> (N, robot.action_dim) absolute OccluBench actions."""
        target = tcp + self.delta_scale * delta[:, :3]
        g = np.clip(delta[:, 6], 0.0, 1.0) * 2.0 - 1.0  # 1 = open in the model convention
        if self.binarize_gripper:  # as experiments/robot/robot_utils.normalize_gripper_action
            g = np.where(g >= 0, 1.0, -1.0)
        grip = -g if self.invert_gripper else g
        out = robot.default_action(target, 0.0, yaw)
        out[:, -1] = grip
        return out

    # ---------------------------------------------------------------- policy API
    def reset(self, n: int):
        self.n = n
        self._calls = 0
        self._queue = [[] for _ in range(n)]
        self._yaw = np.zeros(n)

    def act(self, obs, env):
        rgb, prop = obs["rgb"], obs["proprio"]
        eef = obs["eef_state"]
        deltas = np.empty((self.n, self.action_dim))
        for i in range(self.n):
            if not self._queue[i]:
                ch = self.predict_chunk(rgb["third_person"][i], rgb["wrist"][i], eef[i])
                self._queue[i] = list(ch[: self.open_loop_steps])
            deltas[i] = self._queue[i].pop(0)
        self._yaw = self._yaw + deltas[:, 5]
        return self.to_env_action(deltas, prop[:, :3], self._yaw, env.robot)
