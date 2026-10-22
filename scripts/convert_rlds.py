"""Convert an OccluBench demo dataset to RLDS/TFDS for openvla-oft fine-tuning.

python scripts/convert_rlds.py --demos data/demos_panda --out datasets/rlds [--name occlubench_libero]
-> datasets/rlds/occlubench_libero/1.0.0  (pass --data_root_dir datasets/rlds --dataset_name occlubench_libero
   to openvla-oft's vla-scripts/finetune.py after scripts/patch_openvla_oft.py)
"""

import argparse

from occlubench.rlds import DEFAULT_NAME, convert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=DEFAULT_NAME)
    a = p.parse_args()
    print(convert(a.demos, a.out, name=a.name))


if __name__ == "__main__":
    main()
