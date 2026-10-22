"""Rebuild results/leaderboard.md from every results JSON under results/.

python scripts/build_leaderboard.py [--results results] [--out results/leaderboard.md]
"""

import argparse

from occlubench.leaderboard import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/leaderboard.md")
    a = p.parse_args()
    print(build(a.results, a.out))


if __name__ == "__main__":
    main()
