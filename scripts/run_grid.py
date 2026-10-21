"""Run a paired camera x severity x onset grid and print the metrics table.

MUJOCO_GL=osmesa python scripts/run_grid.py --policy colorcue --seeds 32 --n-envs 16
python scripts/run_grid.py --policy colorcue --seeds 1024 --n-envs 256 --physics warp --render warp
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from occlubench import condition_grid, run_paired, summarize  # noqa: E402
from occlubench.metrics import format_table  # noqa: E402
from occlubench.policies import ColorCuePolicy, ScriptedOracle  # noqa: E402

POLICIES = {
    "oracle": ScriptedOracle,
    "colorcue": ColorCuePolicy,
    "colorcue_memory": lambda: ColorCuePolicy(memory=True),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=list(POLICIES), default="colorcue")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--severities", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--onsets", type=float, nargs="+", default=[0.0, 0.15])
    p.add_argument("--duration", type=float, default=0.25)
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--image", type=int, nargs=2, default=[64, 64])
    p.add_argument("--physics", default="cpu")
    p.add_argument("--render", default="cpu")
    p.add_argument("--out", default=None, help="write raw paired records as JSONL")
    a = p.parse_args()
    conds = condition_grid(severities=a.severities, onsets=a.onsets, duration=a.duration)
    recs = run_paired(
        POLICIES[a.policy],
        conds,
        seeds=range(a.seeds),
        n_envs=a.n_envs,
        horizon=a.horizon,
        image_size=tuple(a.image),
        physics=a.physics,
        render=a.render,
    )
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    print(format_table(summarize(recs)))


if __name__ == "__main__":
    main()
