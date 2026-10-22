"""Benchmark runs on the frozen seed sets and the leaderboard built from their results files.

A results file (schema ``occlubench-results-v1``) is one policy on one robot over the v1 cells:
``{"schema", "policy", "robot", "seedset", "physics", "render", "image_size", "horizon",
"git_sha", "created", "complete", "rows": summarize(...) per cell, "records": [...] (optional)}``.
``complete`` is true only when every v1 cell ran with its full frozen seed list; partial runs
are listed on the leaderboard but marked and never ranked above complete ones.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

from occlubench import seedsets
from occlubench.metrics import bootstrap_ci, summarize
from occlubench.runner import run_paired

SCHEMA = "occlubench-results-v1"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run_benchmark(
    policy_name: str,
    policy_factory: Callable[[], object],
    robot: str = "gantry",
    cells: list[str] | None = None,
    max_seeds: int | None = None,
    n_envs: int = 64,
    keep_records: bool = True,
    **env_kwargs,
) -> dict:
    """Run ``policy`` on the frozen v1 cells (all by default). ``max_seeds`` truncates for smoke runs."""
    all_cells = seedsets.cell_names()
    cells = cells or all_cells
    records = []
    for name in cells:
        cond, seeds = seedsets.load(name)
        seeds = seeds[:max_seeds] if max_seeds else seeds
        records += run_paired(policy_factory, [cond], seeds=seeds, n_envs=n_envs, robot=robot, **env_kwargs)
    complete = set(cells) == set(all_cells) and not (max_seeds and max_seeds < seedsets.SEEDS_PER_CELL)
    return dict(
        schema=SCHEMA,
        policy=policy_name,
        robot=robot,
        seedset=seedsets.VERSION,
        physics=env_kwargs.get("physics", "cpu"),
        render=env_kwargs.get("render", "cpu"),
        image_size=list(env_kwargs.get("image_size", (96, 96))),
        horizon=env_kwargs.get("horizon", 200),
        git_sha=_git_sha(),
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        complete=bool(complete),
        rows=_jsonable(summarize(records)),
        records=records if keep_records else None,
    )


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _entry(res: dict, path: Path) -> dict:
    rows = res["rows"]
    n = sum(r["n"] for r in rows)
    clean = float(np.average([r["clean_sr"] for r in rows], weights=[r["n"] for r in rows]))
    pert = float(np.average([r["pert_sr"] for r in rows], weights=[r["n"] for r in rows]))
    # Overall paired drop with a CI over all pairs, when the raw records are present.
    recs = res.get("records") or []
    if recs:
        d = [float(r["clean_success"]) - float(r["pert_success"]) for r in recs]
        drop = bootstrap_ci(d, n_boot=2000)
    else:
        drop = (clean - pert, None, None)

    def cam(c):
        v = [r["paired_drop"][0] for r in rows if r["camera"] == c and r["paired_drop"][0] is not None]
        return float(np.mean(v)) if v else None

    worst = max(rows, key=lambda r: -1 if r["paired_drop"][0] is None else r["paired_drop"][0])
    return dict(
        policy=res["policy"],
        robot=res["robot"],
        seedset=res["seedset"],
        complete=res["complete"],
        cells=len(rows),
        pairs=n,
        clean_sr=clean,
        pert_sr=pert,
        drop=drop,
        drop_third=cam("third_person"),
        drop_wrist=cam("wrist"),
        worst_cell=worst["condition"],
        worst_drop=worst["paired_drop"][0],
        physics=f"{res.get('physics')}/{res.get('render')}",
        git_sha=res.get("git_sha", ""),
        file=str(path),
    )


def collect(results_dir: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).rglob("*.json")):
        try:
            res = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(res, dict) and res.get("schema") == SCHEMA and res.get("rows"):
            out.append(_entry(res, p))
    # complete runs first, then smaller overall paired drop (more robust), then higher clean SR
    out.sort(key=lambda e: (not e["complete"], e["drop"][0], -e["clean_sr"]))
    return out


def _f(x, signed=False):
    if x is None:
        return "–"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def render_markdown(entries: list[dict]) -> str:
    lines = [
        "# OccluBench leaderboard",
        "",
        "Generated by `python scripts/build_leaderboard.py` from `results/**/*.json` "
        f"(schema `{SCHEMA}`). Do not edit by hand.",
        "",
        "Seed set v1: 16 cells (2 cameras × 2 severities × 2 onsets × 2 anchors), 1,024 frozen pairs per cell "
        "(`occlubench/seedsets_v1`). Ranked by overall paired success drop (clean − occluded), lower is better; "
        "partial runs are listed after complete ones and are not comparable.",
        "",
        "| # | policy | robot | clean SR | occluded SR | paired drop [95% CI] | drop 3rd-person | drop wrist "
        "| worst cell (drop) | pairs | status | sim | commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if not entries:
        lines.append("| – | *no results yet* | | | | | | | | | | | |")
    for i, e in enumerate(entries, 1):
        lo, hi = e["drop"][1], e["drop"][2]
        ci = f"{_f(e['drop'][0], True)} [{_f(lo, True)}, {_f(hi, True)}]" if lo is not None else _f(e["drop"][0], True)
        lines.append(
            f"| {i} | {e['policy']} | {e['robot']} | {_f(e['clean_sr'])} | {_f(e['pert_sr'])} | {ci} "
            f"| {_f(e['drop_third'], True)} | {_f(e['drop_wrist'], True)} | {e['worst_cell']} ({_f(e['worst_drop'], True)}) "
            f"| {e['pairs']} | {'complete' if e['complete'] else 'partial'} | {e['physics']} | {e['git_sha']} |"
        )
    lines += [
        "",
        "Reference policies (`oracle` uses privileged state; `colorcue` is a pixel heuristic) are sanity checks, "
        "not competitors.",
        "",
    ]
    return "\n".join(lines)


def build(results_dir: str | Path = "results", out: str | Path = "results/leaderboard.md") -> str:
    md = render_markdown(collect(results_dir))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(md)
    return md
