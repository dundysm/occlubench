"""Throughput + memory smoke test.

Examples
  # CPU (any machine; needs OSMesa or EGL for rendering)
  MUJOCO_GL=osmesa python scripts/smoke_test.py --n-envs 8 --physics cpu --render cpu
  # NVIDIA GPU (e.g. NVIDIA Brev H100 / A100 instance), MuJoCo Warp physics + batch renderer
  pip install -e ".[warp]"
  python scripts/smoke_test.py --n-envs 64 128 256 512 --physics warp --render warp --image 128 128

Reports env-steps/s (one env-step = one control step = `substeps` physics steps, plus two
camera renders when rendering is on), host peak RSS, and GPU memory via nvidia-smi if present.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from occlubench.conditions import Condition  # noqa: E402
from occlubench.env import OccluEnv  # noqa: E402
from occlubench.policies import ScriptedOracle  # noqa: E402


def gpu_mem_mib():
    if not shutil.which("nvidia-smi"):
        return None
    out = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return [dict(used=int(u), total=int(t), name=n.strip()) for u, t, n in (line.split(",") for line in out)]


def run(n, args):
    env = OccluEnv(n_envs=n, horizon=args.steps, image_size=tuple(args.image), physics=args.physics, render=args.render)
    cond = Condition(camera="third_person", severity=0.5, onset=0.2, duration=0.3)
    obs = env.reset(list(range(n)), [cond] * n)
    pol = ScriptedOracle()
    pol.reset(n)
    obs, _, _ = env.step(pol.act(obs, env))  # warm-up / JIT compile
    t0 = time.perf_counter()
    k = 0
    done = False
    while not done:
        obs, done, _ = env.step(pol.act(obs, env))
        k += 1
    dt = time.perf_counter() - t0
    env.close()
    return dict(
        n_envs=n,
        physics=args.physics,
        render=args.render,
        image=args.image,
        steps=k,
        env_steps_per_s=round(n * k / dt, 1),
        wall_s=round(dt, 2),
        host_peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        gpu=gpu_mem_mib(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, nargs="+", default=[8])
    p.add_argument("--physics", choices=["cpu", "warp"], default="cpu")
    p.add_argument("--render", choices=["cpu", "warp", "none"], default="cpu")
    p.add_argument("--image", type=int, nargs=2, default=[96, 96])
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    for n in args.n_envs:
        print(json.dumps(run(n, args)))


if __name__ == "__main__":
    main()
