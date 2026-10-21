"""Paired rollout runner.

For every seed, a clean rollout and one perturbed rollout per condition start from the
identical initial state (same seed -> same cube pose). Pairing follows MAIL-Bench
(arXiv:2609.35003), which pairs healthy and interrupted rollouts per scene.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from occlubench.conditions import CLEAN, Condition
from occlubench.env import OccluEnv


def _rollout(env: OccluEnv, policy, seeds, conds, frames: list | None = None):
    obs = env.reset(seeds, conds)
    policy.reset(env.n)
    done = False
    while not done:
        obs, done, info = env.step(policy.act(obs, env))
        if frames is not None:
            frames.append({k: v.copy() for k, v in obs["rgb"].items()})
    return env.success_step.copy(), np.stack([env.coverage, env.coverage_other], 1)


def run_paired(
    policy_factory: Callable[[], object],
    conditions: Sequence[Condition],
    seeds: Sequence[int],
    n_envs: int = 8,
    **env_kwargs,
) -> list[dict]:
    """Returns one record per (condition, seed) with clean and perturbed outcomes."""
    seeds = list(seeds)
    env = OccluEnv(n_envs=n_envs, **env_kwargs)
    policy = policy_factory()
    horizon = env.horizon
    batches = [seeds[i : i + n_envs] for i in range(0, len(seeds), n_envs)]

    def run_all(cond: Condition):
        steps, covs = [], []
        for b in batches:
            pad = b + [b[-1]] * (n_envs - len(b))
            s, c = _rollout(env, policy, pad, [cond] * n_envs)
            steps.append(s[: len(b)])
            covs.append(c[: len(b)])
        return np.concatenate(steps), np.concatenate(covs)

    clean_steps, _ = run_all(CLEAN)
    records = []
    for cond in conditions:
        pert_steps, cov = run_all(cond)
        for k, seed in enumerate(seeds):
            records.append(
                dict(
                    condition=cond.name,
                    **cond.to_dict(),
                    seed=seed,
                    clean_success=bool(clean_steps[k] >= 0),
                    clean_step=int(clean_steps[k]),
                    pert_success=bool(pert_steps[k] >= 0),
                    pert_step=int(pert_steps[k]),
                    coverage=float(cov[k, 0]),
                    coverage_other=float(cov[k, 1]),
                    horizon=horizon,
                )
            )
    env.close()
    return records
