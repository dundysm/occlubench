"""Batched demonstration generator: scripted-oracle rollouts written to disk for fine-tuning.

Layout (LeRobot-v2-like, NumPy instead of parquet/mp4 so it needs no extra dependencies)::

    out_dir/
      meta/info.json        # format, robot, fps, image size, feature shapes and conventions
      meta/episodes.jsonl   # one line per episode: index, seed, length, success, condition, task
      data/episode_000000.npz

Each episode file holds per-step arrays (T = episode length):

* ``observation.images.third_person``, ``observation.images.wrist``: (T, H, W, 3) uint8
* ``observation.state``: (T, 8) ``OccluEnv.eef_state()`` (tcp xyz, tcp axis-angle, 2 finger joints)
* ``observation.proprio``: (T, P) ``OccluEnv.proprio()``
* ``action``: (T, 7) OpenVLA-OFT/LIBERO-style deltas: dxyz (commanded tcp target minus current tcp),
  droll/dpitch/dyaw (change in the commanded orientation, zero for the gantry), gripper in [0, 1]
  with 1 = open (the RLDS convention openvla-oft trains on)
* ``action.absolute``: (T, action_dim) the raw OccluBench action that was executed

Demo seeds come from a range disjoint from the frozen evaluation seed sets (``DEMO_SEED_BASE``),
so fine-tuning never sees an evaluation initial state. By default only successful clean episodes
are kept and each is truncated a few steps after success.

Occluded demos: with ``occluded_frac > 0`` each episode is, with that probability, run under a
condition drawn uniformly from ``occluded_conditions`` (default: the training grid
``TRAIN_OCCLUSIONS``, whose severities/onsets differ from the v1 evaluation cells). The oracle
uses privileged state, so it keeps acting correctly while a camera is blocked; the demos therefore
show the behaviour a robust policy should have. The condition of each episode is stored in
``meta/episodes.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from occlubench.conditions import CLEAN, Condition, condition_grid
from occlubench.env import OccluEnv
from occlubench.policies import ScriptedOracle

DEMO_SEED_BASE = 1_000_000  # frozen evaluation seed sets live below this
FORMAT = "occlubench-demos-v1"
# Training occlusions: severities and onsets chosen to differ from the v1 evaluation cells
# (severity 0.5/1.0, onset 0.15/0.45), so no evaluation cell is seen verbatim in training.
TRAIN_OCCLUSIONS = condition_grid(
    severities=(0.35, 0.7, 0.9), onsets=(0.05, 0.25, 0.35), duration=0.25
) + condition_grid(severities=(0.35, 0.7, 0.9), onsets=(0.05, 0.25, 0.35), duration=0.25, anchor="world")
TASK = "pick up the red cube and place it on the green target"


def _delta_actions(abs_actions: np.ndarray, tcp: np.ndarray, robot) -> np.ndarray:
    """(T, A) absolute OccluBench actions + (T, 3) tcp -> (T, 7) delta actions."""
    t = len(abs_actions)
    out = np.zeros((t, 7))
    out[:, :3] = abs_actions[:, :3] - tcp
    if robot.action_dim == 7:
        rpy = abs_actions[:, 3:6]
        prev = np.concatenate([rpy[:1], rpy[:-1]], 0)
        out[:, 3:6] = (rpy - prev + np.pi) % (2 * np.pi) - np.pi
    out[:, 6] = (1.0 - np.clip(abs_actions[:, -1], -1, 1)) / 2.0  # +1 close -> 0, -1 open -> 1
    return out


def generate(
    out_dir: str | Path,
    n_episodes: int,
    robot: str = "gantry",
    n_envs: int = 16,
    horizon: int = 200,
    image_size: tuple[int, int] = (224, 224),
    physics: str = "cpu",
    render: str = "cpu",
    condition: Condition = CLEAN,
    seed_base: int = DEMO_SEED_BASE,
    keep_failures: bool = False,
    tail_steps: int = 10,
    occluded_frac: float = 0.0,
    occluded_conditions: list[Condition] | None = None,
) -> dict:
    """Roll out the scripted oracle in batches of ``n_envs`` and write episodes. Returns a summary."""
    out = Path(out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    env = OccluEnv(n_envs=n_envs, horizon=horizon, image_size=image_size, physics=physics, render=render, robot=robot)
    pol = ScriptedOracle()
    written, tried, seed = 0, 0, seed_base
    occ = list(occluded_conditions or TRAIN_OCCLUSIONS)
    rng = np.random.default_rng(seed_base)
    episodes = []
    while written < n_episodes:
        seeds = list(range(seed, seed + n_envs))
        seed += n_envs
        conds = [
            occ[rng.integers(len(occ))] if occluded_frac > 0 and rng.random() < occluded_frac else condition
            for _ in range(n_envs)
        ]
        obs = env.reset(seeds, conds)
        pol.reset(n_envs)
        buf = {k: [] for k in ("tp", "wr", "state", "proprio", "act", "tcp")}
        done = False
        while not done:
            a = pol.act(obs, env)
            buf["tp"].append(obs["rgb"]["third_person"])
            buf["wr"].append(obs["rgb"]["wrist"])
            buf["state"].append(obs["eef_state"])
            buf["proprio"].append(obs["proprio"])
            buf["tcp"].append(obs["proprio"][:, :3])
            buf["act"].append(np.asarray(a, float))
            obs, done, _ = env.step(a)
        succ = env.success_step
        arr = {k: np.stack(v, 1) for k, v in buf.items()}  # (N, T, ...)
        for i in range(n_envs):
            tried += 1
            ok = bool(succ[i] >= 0)
            if not ok and not keep_failures:
                continue
            if written >= n_episodes:
                break
            t_end = min(horizon, int(succ[i]) + tail_steps) if ok else horizon
            ep = {
                "observation.images.third_person": arr["tp"][i, :t_end].astype(np.uint8),
                "observation.images.wrist": arr["wr"][i, :t_end].astype(np.uint8),
                "observation.state": arr["state"][i, :t_end].astype(np.float32),
                "observation.proprio": arr["proprio"][i, :t_end].astype(np.float32),
                "action": _delta_actions(arr["act"][i, :t_end], arr["tcp"][i, :t_end], env.robot).astype(np.float32),
                "action.absolute": arr["act"][i, :t_end].astype(np.float32),
            }
            np.savez_compressed(out / "data" / f"episode_{written:06d}.npz", **ep)
            episodes.append(
                dict(
                    episode_index=written,
                    seed=seeds[i],
                    length=t_end,
                    success=ok,
                    success_step=int(succ[i]),
                    condition=conds[i].to_dict(),
                    task=TASK,
                )
            )
            written += 1
    env.close()
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    info = dict(
        format=FORMAT,
        robot=robot,
        physics=physics,
        render=render,
        image_size=list(image_size),
        horizon=horizon,
        control_dt=float(env.m.opt.timestep * env.substeps),
        n_episodes=written,
        n_tried=tried,
        n_occluded=int(sum(e["condition"]["camera"] != "none" for e in episodes)),
        occluded_frac=occluded_frac,
        n_frames=int(sum(e["length"] for e in episodes)),
        seed_base=seed_base,
        task=TASK,
        action_convention="dxyz, drpy, gripper in [0,1] with 1=open (OpenVLA-OFT / RLDS)",
        state_convention="tcp xyz, tcp axis-angle (world), 2 finger joints (LIBERO layout)",
    )
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return info


def load_episode(out_dir: str | Path, index: int) -> dict:
    with np.load(Path(out_dir) / "data" / f"episode_{index:06d}.npz") as z:
        return {k: z[k] for k in z.files}
