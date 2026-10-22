import json

import numpy as np

from occlubench.demos import DEMO_SEED_BASE, generate, load_episode


def test_generate_demos_roundtrip(tmp_path):
    info = generate(tmp_path, 3, robot="gantry", n_envs=2, horizon=160, image_size=(24, 24))
    assert info["n_episodes"] == 3 and info["format"] == "occlubench-demos-v1"
    eps = [json.loads(x) for x in (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 3 and all(e["success"] for e in eps)
    assert all(e["seed"] >= DEMO_SEED_BASE for e in eps)
    ep = load_episode(tmp_path, 0)
    t = eps[0]["length"]
    assert ep["observation.images.third_person"].shape == (t, 24, 24, 3)
    assert ep["observation.images.wrist"].dtype == np.uint8
    assert ep["observation.state"].shape == (t, 8)
    assert ep["action"].shape == (t, 7)
    g = ep["action"][:, 6]
    assert set(np.unique(g)) <= {0.0, 1.0} and g[0] == 1.0 and g.min() == 0.0  # opens, then closes
    np.testing.assert_allclose(
        ep["action"][:, :3], ep["action.absolute"][:, :3] - ep["observation.proprio"][:, :3], atol=1e-5
    )
