"""Generate scripted-oracle demonstrations for fine-tuning (see occlubench/demos.py for the format).

MUJOCO_GL=egl python scripts/generate_demos.py --out data/demos_panda --robot panda --episodes 500 --n-envs 64
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from occlubench.demos import generate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--robot", choices=["gantry", "panda"], default="panda")
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--image", type=int, nargs=2, default=[224, 224])
    p.add_argument("--physics", default="cpu")
    p.add_argument("--render", default="cpu")
    p.add_argument("--keep-failures", action="store_true")
    a = p.parse_args()
    info = generate(
        a.out,
        a.episodes,
        robot=a.robot,
        n_envs=a.n_envs,
        horizon=a.horizon,
        image_size=tuple(a.image),
        physics=a.physics,
        render=a.render,
        keep_failures=a.keep_failures,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
