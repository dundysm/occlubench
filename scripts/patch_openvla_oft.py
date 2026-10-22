"""Register an OccluBench RLDS dataset in an openvla-oft checkout (idempotent).

python scripts/patch_openvla_oft.py /path/to/openvla-oft [--name occlubench_libero]

Appends to prismatic/vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py so that <name> reuses
the LIBERO config (image / wrist_image / EEF_state + gripper_state) and libero_dataset_transform.
Keep "libero" in the name: openvla-oft picks its LIBERO constants (8-step chunks, 7-D actions,
8-D proprio) when "libero" appears in the command line.
"""

import argparse
from pathlib import Path

MARK = "# --- OccluBench registration: {name} ---"
SNIPPETS = {
    "configs.py": 'OXE_DATASET_CONFIGS["{name}"] = dict(OXE_DATASET_CONFIGS["libero_spatial_no_noops"])\n',
    "transforms.py": 'OXE_STANDARDIZATION_TRANSFORMS["{name}"] = libero_dataset_transform\n',
    "mixtures.py": 'OXE_NAMED_MIXTURES["{name}"] = [("{name}", 1.0)]\n',
}


def patch(repo: str | Path, name: str = "occlubench_libero") -> list[Path]:
    base = Path(repo) / "prismatic" / "vla" / "datasets" / "rlds" / "oxe"
    changed = []
    for fname, snippet in SNIPPETS.items():
        f = base / fname
        text = f.read_text()
        mark = MARK.format(name=name)
        if mark in text:
            continue
        f.write_text(text.rstrip("\n") + "\n\n\n" + mark + "\n" + snippet.format(name=name))
        changed.append(f)
    return changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--name", default="occlubench_libero")
    a = p.parse_args()
    for f in patch(a.repo, a.name):
        print("patched", f)


if __name__ == "__main__":
    main()
