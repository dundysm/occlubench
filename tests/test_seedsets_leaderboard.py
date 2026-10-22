import json

import numpy as np

from occlubench import seedsets
from occlubench.demos import DEMO_SEED_BASE
from occlubench.leaderboard import SCHEMA, build, collect, run_benchmark
from occlubench.policies import ScriptedOracle


def test_v1_seedsets_frozen_and_valid():
    m = seedsets.manifest()
    names = seedsets.cell_names()
    assert len(names) == 16 and names == [c.name for c in seedsets.cells_v1()]
    for n in names:
        cond, seeds = seedsets.load(n)  # verifies the SHA-256 against the manifest
        assert cond.name == n
        assert len(seeds) == len(set(seeds)) == m["seeds_per_cell"] >= 1000
        assert 0 <= min(seeds) and max(seeds) < DEMO_SEED_BASE


def test_v1_seedsets_regenerate_identically(tmp_path):
    seedsets.write_v1(tmp_path)
    fresh = json.loads((tmp_path / "manifest.json").read_text())
    assert fresh == seedsets.manifest()


def test_benchmark_smoke_and_leaderboard(tmp_path):
    name = seedsets.cell_names()[0]
    res = run_benchmark(
        "oracle", ScriptedOracle, cells=[name], max_seeds=2, n_envs=2, horizon=160, render="none", image_size=(16, 16)
    )
    assert res["schema"] == SCHEMA and not res["complete"] and len(res["records"]) == 2
    (tmp_path / "oracle.json").write_text(json.dumps(res))
    # a fake complete entry ranks first regardless of its drop
    fake = dict(res, policy="fake", complete=True, records=None)
    fake["rows"] = [dict(r, paired_drop=[0.5, 0.4, 0.6]) for r in res["rows"]]
    (tmp_path / "fake.json").write_text(json.dumps(fake))
    (tmp_path / "junk.json").write_text("[1, 2]")
    entries = collect(tmp_path)
    assert [e["policy"] for e in entries] == ["fake", "oracle"]
    md = build(tmp_path, tmp_path / "leaderboard.md")
    assert "| 1 | fake |" in md and "partial" in md
    assert np.isclose(entries[1]["clean_sr"], 1.0)


def test_empty_leaderboard(tmp_path):
    assert "no results yet" in build(tmp_path, tmp_path / "lb.md")
