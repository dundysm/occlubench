"""Run a policy on the frozen v1 seed sets and write a leaderboard results file.

python scripts/run_benchmark.py --policy colorcue --robot gantry --physics warp --render warp --n-envs 256 \
    --out results/colorcue_gantry.json
python scripts/run_benchmark.py --policy oracle --max-seeds 16 --out results/smoke_oracle.json   # partial
python scripts/build_leaderboard.py
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from run_grid import POLICIES  # noqa: E402

from occlubench.leaderboard import run_benchmark  # noqa: E402


def _write(res, path):
    """``.json.gz`` paths are gzip-compressed."""
    import gzip

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = json.dumps(res, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(gzip.compress(data) if path.endswith(".gz") else data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=list(POLICIES), default="colorcue")
    p.add_argument("--robot", choices=["gantry", "panda"], default="gantry")
    p.add_argument("--cells", nargs="*", default=None, help="subset of v1 cell names (default: all 16)")
    p.add_argument("--max-seeds", type=int, default=None, help="truncate each cell (marks the run partial)")
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--image", type=int, nargs=2, default=[64, 64])
    p.add_argument("--physics", default="cpu")
    p.add_argument("--render", default="cpu")
    p.add_argument("--no-records", action="store_true", help="omit per-pair records (smaller file, no overall CI)")
    p.add_argument("--merge", nargs="+", default=None, help="merge these results files into --out and exit")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.merge:
        from occlubench.leaderboard import load_results, merge_results

        res = merge_results([load_results(f) for f in a.merge])
        _write(res, a.out)
        print(f"merged {len(a.merge)} files into {a.out} (complete={res['complete']}, cells={len(res['rows'])})")
        return
    res = run_benchmark(
        a.policy,
        POLICIES[a.policy],
        robot=a.robot,
        cells=a.cells,
        max_seeds=a.max_seeds,
        n_envs=a.n_envs,
        keep_records=not a.no_records,
        horizon=a.horizon,
        image_size=tuple(a.image),
        physics=a.physics,
        render=a.render,
    )
    _write(res, a.out)
    print(f"wrote {a.out} (complete={res['complete']}, cells={len(res['rows'])})")


if __name__ == "__main__":
    main()
