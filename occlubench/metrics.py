"""Paired metrics with bootstrap 95% confidence intervals (resampling seeds)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap over the leading axis. Returns (point, lo, hi); NaNs if empty."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def _nanmean(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def summarize(records: list[dict], n_boot: int = 10_000) -> list[dict]:
    """Per condition:

    * clean_sr / pert_sr       success rates
    * paired_drop              mean(clean - perturbed) over paired seeds, with CI
    * conditional_success      P(perturbed success | clean success), with CI
    * recovery_steps           mean extra steps to success (perturbed - clean) when both succeed
    * coverage                 mean achieved occlusion fraction (target pixels hidden) in the targeted camera
    * coverage_other           leakage: fraction hidden in the *other* camera by the same physical occluder
    """
    by = defaultdict(list)
    for r in records:
        by[r["condition"]].append(r)
    rows = []
    for name, rs in by.items():
        c = np.array([r["clean_success"] for r in rs], float)
        p = np.array([r["pert_success"] for r in rs], float)
        both = [r["pert_step"] - r["clean_step"] for r in rs if r["clean_success"] and r["pert_success"]]
        cond = p[c == 1]
        rows.append(
            dict(
                condition=name,
                camera=rs[0]["camera"],
                severity=rs[0]["severity"],
                onset=rs[0]["onset"],
                n=len(rs),
                clean_sr=float(c.mean()),
                pert_sr=float(p.mean()),
                paired_drop=bootstrap_ci(c - p, n_boot=n_boot),
                conditional_success=bootstrap_ci(cond, n_boot=n_boot),
                recovery_steps=bootstrap_ci(both, n_boot=n_boot),
                coverage=_nanmean([r["coverage"] for r in rs]),
                coverage_other=_nanmean([r.get("coverage_other", np.nan) for r in rs]),
            )
        )
    return rows


def format_table(rows: list[dict]) -> str:
    def ci(t):
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"

    head = "| condition | n | coverage (other cam) | clean SR | pert SR | paired drop [95% CI] | P(pert ok given clean ok) [95% CI] | recovery steps [95% CI] |"
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['n']} | {r['coverage']:.2f} ({r['coverage_other']:.2f}) | {r['clean_sr']:.2f} | {r['pert_sr']:.2f} | "
            f"{ci(r['paired_drop'])} | {ci(r['conditional_success'])} | {ci(r['recovery_steps'])} |"
        )
    return "\n".join(lines)
