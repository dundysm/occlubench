"""Convert an OccluBench demo dataset (``occlubench.demos``) to an RLDS/TFDS dataset that
openvla-oft's ``vla-scripts/finetune.py`` can train on.

The output mirrors the modified LIBERO RLDS datasets openvla-oft fine-tunes on
(``libero_spatial_no_noops`` etc.), so the LIBERO dataset config and transform apply unchanged
once the dataset name is registered (``scripts/patch_openvla_oft.py``)::

    steps/observation/image        (H, W, 3) uint8   third-person camera  ("primary")
    steps/observation/wrist_image  (H, W, 3) uint8   wrist camera
    steps/observation/state        (8,) float32      tcp xyz, tcp axis-angle, 2 finger joints
    steps/observation/joint_state  (P,) float32      OccluBench proprio vector
    steps/action                   (7,) float32      dxyz, drpy, gripper in LIBERO's raw convention
                                                     (-1 = open, +1 = close; libero_dataset_transform
                                                     clips to [0, 1] and inverts to 1 = open)
    steps/language_instruction, reward, discount, is_first, is_last, is_terminal
    episode_metadata/file_path, episode_metadata/seed, episode_metadata/condition

Requires ``tensorflow`` and ``tensorflow_datasets`` (``pip install -e ".[rlds]"``). This module
does not import MuJoCo: run it in its own process (``scripts/convert_rlds.py``), because TensorFlow
and MuJoCo crash when loaded into the same process on some builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_episode(demo_dir: str | Path, index: int) -> dict:
    """Same as ``occlubench.demos.load_episode``, without importing MuJoCo."""
    with np.load(Path(demo_dir) / "data" / f"episode_{index:06d}.npz") as z:
        return {k: z[k] for k in z.files}


DEFAULT_NAME = "occlubench_libero"  # "libero" in the name makes openvla-oft pick its LIBERO constants


def libero_action(action: np.ndarray) -> np.ndarray:
    """OccluBench demo action (gripper in [0, 1], 1 = open) -> LIBERO raw action (gripper -1 open, +1 close)."""
    a = np.array(action, np.float32, copy=True)
    a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


def convert(demo_dir: str | Path, out_dir: str | Path, name: str = DEFAULT_NAME, version: str = "1.0.0") -> Path:
    """Build ``out_dir/<name>/<version>`` with TFDS. Returns that directory."""
    import tensorflow_datasets as tfds

    demo_dir = Path(demo_dir)
    info = json.loads((demo_dir / "meta" / "info.json").read_text())
    eps = [json.loads(x) for x in (demo_dir / "meta" / "episodes.jsonl").read_text().splitlines() if x.strip()]
    if not eps:
        raise ValueError(f"no episodes in {demo_dir}")
    first = load_episode(demo_dir, eps[0]["episode_index"])
    h, w = first["observation.images.third_person"].shape[1:3]
    pdim = first["observation.proprio"].shape[1]
    f32 = np.float32

    features = tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "image": tfds.features.Image(shape=(h, w, 3), dtype=np.uint8, encoding_format="png"),
                            "wrist_image": tfds.features.Image(shape=(h, w, 3), dtype=np.uint8, encoding_format="png"),
                            "state": tfds.features.Tensor(shape=(8,), dtype=f32),
                            "joint_state": tfds.features.Tensor(shape=(pdim,), dtype=f32),
                        }
                    ),
                    "action": tfds.features.Tensor(shape=(7,), dtype=f32),
                    "discount": tfds.features.Scalar(dtype=f32),
                    "reward": tfds.features.Scalar(dtype=f32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(),
                    "seed": tfds.features.Scalar(dtype=np.int64),
                    "condition": tfds.features.Text(),
                }
            ),
        }
    )

    def gen():
        for e in eps:
            ep = load_episode(demo_dir, e["episode_index"])
            t = len(ep["action"])
            act = libero_action(ep["action"])
            steps = []
            for k in range(t):
                last = k == t - 1
                steps.append(
                    dict(
                        observation=dict(
                            image=ep["observation.images.third_person"][k],
                            wrist_image=ep["observation.images.wrist"][k],
                            state=ep["observation.state"][k].astype(f32),
                            joint_state=ep["observation.proprio"][k].astype(f32),
                        ),
                        action=act[k],
                        discount=1.0,
                        reward=float(last and e["success"]),
                        is_first=k == 0,
                        is_last=last,
                        is_terminal=bool(last and e["success"]),
                        language_instruction=e["task"],
                    )
                )
            key = f"episode_{e['episode_index']:06d}"
            meta = dict(
                file_path=str(demo_dir / "data" / f"{key}.npz"),
                seed=int(e["seed"]),
                condition=json.dumps(e["condition"]),
            )
            yield key, dict(steps=steps, episode_metadata=meta)

    class _Builder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version(version)

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                features=features,
                description=f"OccluBench scripted-oracle demos ({info.get('robot')}), LIBERO-style RLDS.",
            )

        def _split_generators(self, dl_manager):
            return {"train": gen()}

        def _generate_examples(self):  # pragma: no cover - unused, splits come from gen()
            yield from gen()

    _Builder.name = name
    builder = _Builder(data_dir=str(out_dir))
    builder.download_and_prepare()
    return Path(out_dir) / name / version
