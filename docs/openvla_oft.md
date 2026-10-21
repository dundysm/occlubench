# Running OpenVLA-OFT on OccluBench

`occlubench.policies.OpenVLAOFTPolicy` adapts [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
to OccluBench. Status: the adapter's chunking, unnormalisation and action conversion are tested in CI
with `backend="mock"`. The `hf` and `oft` backends load real weights but **have not yet been run end to
end on a GPU by this project**, and no OpenVLA-OFT numbers are reported yet.

## What the adapter does

| OccluBench | OpenVLA-OFT |
|---|---|
| `rgb["third_person"]` | `full_image` (resized to 224×224) |
| `rgb["wrist"]` | `wrist_image` (resized to 224×224) |
| proprio | 8-D `state`: tcp xyz, 3 zeros (orientation), finger width ×2 |
| absolute tcp target + gripper in [-1, 1] (+1 = close) | chunk of 7-D deltas `(dx, dy, dz, droll, dpitch, dyaw, grip∈[0,1], 1 = open)` |

Per env, the adapter queries the model once per `open_loop_steps` (default = `chunk` = 8), queues the
chunk, and executes one delta per control step: `target = tcp + delta_scale * dxyz`, yaw accumulates
(used by the Panda), gripper is mapped and inverted (`invert_gripper=True`). Bounds unnormalisation
(`[-1, 1] -> [q01, q99]`, masked dims passed through) is `occlubench.policies.unnormalize_actions`.

Backends:

* `mock`: no model; deterministic chunks. For CI and for testing the plumbing.
* `hf`: `transformers.AutoModelForVision2Seq(..., trust_remote_code=True)` + `model.predict_action`.
  Works for checkpoints whose remote code returns actions from `predict_action` alone. Set
  `outputs_normalized=True` if the model returns normalised actions.
* `oft` (recommended for OFT checkpoints): uses the openvla-oft repo's loaders, which add the L1
  regression action head and proprio projector that OFT checkpoints need.

## Brev H100 recipe

```bash
git clone https://github.com/dundysm/occlubench && cd occlubench
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[warp,vla,dev]"
git clone https://github.com/moojink/openvla-oft && pip install -e openvla-oft   # for backend=oft
python -m occlubench.menagerie                                                     # fetch the Panda
export MUJOCO_GL=egl
# smoke test the plumbing first
python scripts/run_grid.py --policy openvla_oft_mock --robot panda --seeds 8 --n-envs 8 --image 224 224
# then a released checkpoint (expect ~0 success zero-shot: LIBERO robot, cameras and action space differ)
OCCLUBENCH_VLA_BACKEND=oft OCCLUBENCH_VLA_MODEL=moojink/openvla-7b-oft-finetuned-libero-spatial \
  python scripts/run_grid.py --policy openvla_oft --robot panda --seeds 64 --n-envs 8 --image 224 224 \
  --physics warp --render warp --out results/oft_zeroshot.jsonl
```

A 7B model in bf16 needs about 16 GB of GPU memory; the adapter queries one env at a time, so
throughput is bounded by model latency, not simulation.

## Before trusting the numbers

1. Fine-tune on OccluBench demonstrations (scripted-oracle rollouts on clean episodes); the
   released checkpoints were trained on LIBERO.
2. Check the clean success rate is high before reading any paired drop.
3. Verify the orientation part of `state` and the gripper convention against your checkpoint's
   training data, and adjust `delta_scale` / `invert_gripper` if needed.
