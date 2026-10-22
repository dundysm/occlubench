"""RLDS converter. Needs tensorflow-datasets; MuJoCo work runs in a subprocess because TF and
MuJoCo must not share a process (see occlubench/rlds.py). CI runs this file in its own job."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

tfds = pytest.importorskip("tensorflow_datasets")

from occlubench.rlds import convert, libero_action, load_episode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_libero_action_gripper():
    a = np.zeros((2, 7), np.float32)
    a[:, 6] = [1.0, 0.0]  # demos: 1 = open
    np.testing.assert_allclose(libero_action(a)[:, 6], [-1.0, 1.0])  # LIBERO raw: -1 open, +1 close


def test_convert_to_rlds(tmp_path):
    demo = tmp_path / "demos"
    if "mujoco" in sys.modules:
        pytest.skip("MuJoCo already loaded in this process; run tests/test_rlds.py on its own")
    args = ["--out", str(demo), "--robot", "gantry", "--episodes", "2", "--n-envs", "2", "--horizon", "160"]
    args += ["--image", "32", "32", "--occluded-frac", "0.5"]
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "osmesa"))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_demos.py"), *args], check=True, env=env)
    path = convert(demo, tmp_path / "rlds")
    b = tfds.builder_from_directory(str(path))
    assert b.info.splits["train"].num_examples == 2
    ds = b.as_dataset(split="train")
    ep0 = next(iter(tfds.as_numpy(ds)))
    steps = list(ep0["steps"])
    idx = int(Path(ep0["episode_metadata"]["file_path"].decode()).stem.split("_")[1])
    ref = load_episode(demo, idx)
    assert len(steps) == len(ref["action"])
    s0 = steps[0]
    assert s0["observation"]["image"].shape == (32, 32, 3) and s0["observation"]["wrist_image"].dtype == np.uint8
    np.testing.assert_array_equal(s0["observation"]["image"], ref["observation.images.third_person"][0])
    np.testing.assert_allclose(s0["observation"]["state"], ref["observation.state"][0])
    np.testing.assert_allclose([s["action"] for s in steps], libero_action(ref["action"]), atol=1e-6)
    assert s0["is_first"] and steps[-1]["is_last"] and steps[-1]["reward"] == 1.0
    assert s0["language_instruction"].decode().startswith("pick up")
