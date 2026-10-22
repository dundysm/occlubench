#!/usr/bin/env bash
# Fine-tune OpenVLA-OFT (LoRA, L1 action head, 2 cameras + proprio) on OccluBench demos and evaluate it
# on the frozen v1 seed sets. Written for one NVIDIA L40S (48 GB) on Brev; works on A100/H100 too.
#
#   git clone https://github.com/dundysm/occlubench && cd occlubench
#   bash scripts/brev_finetune.sh                # all stages
#   STAGES="eval" CKPT=runs/<run> bash scripts/brev_finetune.sh   # re-run a single stage
#
# Stages (STAGES, space separated): setup demos convert train eval
# Overrides: ROBOT=panda EPISODES=500 OCC_FRAC=0.3 IMAGE=224 N_ENVS=64 MAX_STEPS=20000 BATCH=4 ACCUM=2
#            LR=5e-4 LORA_RANK=32 SAVE_FREQ=5000 EVAL_SEEDS=64 EVAL_ENVS=16 HORIZON=200 PORT=8777
#            OFT_COMMIT=<openvla-oft sha> WANDB_MODE=offline
#
# Memory: 7B in bf16 is ~15 GB; with LoRA r32, two 224x224 images and BATCH=4 training should fit in
# 48 GB (not yet measured on an L40S by this project). If it OOMs, use BATCH=2 ACCUM=4.
# Two venvs on purpose: .venv (OccluBench: MuJoCo + Warp) and .venv-oft (openvla-oft: torch 2.2,
# TensorFlow 2.15, Python 3.10). TensorFlow and MuJoCo crash in one process, so evaluation runs
# the model behind openvla-oft's deploy.py server and OccluBench talks to it over HTTP.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGES="${STAGES:-setup demos convert train eval}"
ROBOT="${ROBOT:-panda}"
EPISODES="${EPISODES:-500}"
OCC_FRAC="${OCC_FRAC:-0.3}"
IMAGE="${IMAGE:-224}"
N_ENVS="${N_ENVS:-64}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-2}"
LR="${LR:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
EVAL_SEEDS="${EVAL_SEEDS:-64}"
EVAL_ENVS="${EVAL_ENVS:-16}"
HORIZON="${HORIZON:-200}"
PORT="${PORT:-8777}"
OFT_COMMIT="${OFT_COMMIT:-e4287e94541f459edc4feabc4e181f537cd569a8}"   # moojink/openvla-oft main, checked 2026-10-22
DATASET="occlubench_libero"   # keep "libero" in the name: openvla-oft then uses its LIBERO constants
DEMOS="data/demos_${ROBOT}"
RLDS="datasets/rlds"
OFT="third_party/openvla-oft"
export WANDB_MODE="${WANDB_MODE:-offline}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/finetune_${TS}"
mkdir -p "$OUT" data datasets runs third_party
log() { echo "== $*" | tee -a "$OUT/log.txt"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has setup; then
  log "setup"
  nvidia-smi | tee -a "$OUT/log.txt"
  if ! dpkg -s libegl1 >/dev/null 2>&1 || ! python3 -m ensurepip --version >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y libegl1 libgl1 libosmesa6 git python3-venv || true
  fi
  # OccluBench venv
  [ -d .venv ] || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e ".[dev,warp]"
  .venv/bin/python -m occlubench.menagerie
  # openvla-oft venv (Python 3.10 via uv)
  python3 -m pip install -q --user uv 2>/dev/null || .venv/bin/pip install -q uv
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"; [ -x "$UV" ] || UV=".venv/bin/uv"
  [ -d .venv-oft ] || "$UV" venv -p 3.10 .venv-oft
  [ -d "$OFT" ] || git clone https://github.com/moojink/openvla-oft "$OFT"
  git -C "$OFT" fetch -q origin && git -C "$OFT" checkout -q "$OFT_COMMIT"
  VIRTUAL_ENV=.venv-oft "$UV" pip install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
  VIRTUAL_ENV=.venv-oft "$UV" pip install -q -e "$OFT"
  VIRTUAL_ENV=.venv-oft "$UV" pip install -q packaging ninja
  .venv-oft/bin/python -m pip --version >/dev/null 2>&1 || VIRTUAL_ENV=.venv-oft "$UV" pip install -q pip
  .venv-oft/bin/python -m pip install -q "flash-attn==2.5.5" --no-build-isolation || log "flash-attn build failed; training still works without it (slower)"
  VIRTUAL_ENV=.venv-oft "$UV" pip install -q --no-deps -e .   # only occlubench.rlds is used here (no MuJoCo)
  .venv/bin/python scripts/patch_openvla_oft.py "$OFT" --name "$DATASET" | tee -a "$OUT/log.txt"
fi

if has demos; then
  log "demos: $EPISODES $ROBOT episodes, occluded fraction $OCC_FRAC"
  rm -rf "$DEMOS"
  MUJOCO_GL=egl .venv/bin/python scripts/generate_demos.py --out "$DEMOS" --robot "$ROBOT" --episodes "$EPISODES" \
    --n-envs "$N_ENVS" --horizon "$HORIZON" --image "$IMAGE" "$IMAGE" --physics warp --render warp \
    --occluded-frac "$OCC_FRAC" | tee -a "$OUT/log.txt"
fi

if has convert; then
  log "convert to RLDS ($RLDS/$DATASET)"
  rm -rf "${RLDS:?}/$DATASET"
  .venv-oft/bin/python scripts/convert_rlds.py --demos "$DEMOS" --out "$RLDS" --name "$DATASET" | tee -a "$OUT/log.txt"
fi

if has train; then
  log "train: LoRA r$LORA_RANK, batch $BATCH x accum $ACCUM, $MAX_STEPS steps"
  (cd "$OFT" && "$ROOT/.venv-oft/bin/torchrun" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir "$ROOT/$RLDS" \
    --dataset_name "$DATASET" \
    --run_root_dir "$ROOT/runs" \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 2 --use_proprio True \
    --batch_size "$BATCH" --grad_accumulation_steps "$ACCUM" \
    --learning_rate "$LR" --num_steps_before_decay $((MAX_STEPS * 2 / 3)) --max_steps "$MAX_STEPS" \
    --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only True --merge_lora_during_training True \
    --image_aug True --lora_rank "$LORA_RANK" --shuffle_buffer_size 20000 \
    --wandb_entity occlubench --wandb_project occlubench \
    --run_id_note "occlubench-${ROBOT}-occ${OCC_FRAC}") 2>&1 | tee "$OUT/train.log"
fi

if has eval; then
  CKPT="${CKPT:-$(ls -td runs/*/ 2>/dev/null | head -1)}"
  [ -n "$CKPT" ] || { echo "no checkpoint under runs/; set CKPT=..."; exit 1; }
  log "eval $CKPT on v1 cells, $EVAL_SEEDS seeds per cell (partial unless EVAL_SEEDS=1024)"
  (cd "$OFT" && "$ROOT/.venv-oft/bin/python" vla-scripts/deploy.py --pretrained_checkpoint "$ROOT/${CKPT%/}" \
    --unnorm_key "$DATASET" --num_images_in_input 2 --use_proprio True --center_crop True \
    --lora_rank "$LORA_RANK" --port "$PORT") > "$OUT/server.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 1 120); do
    grep -q "Uvicorn running" "$OUT/server.log" && break
    kill -0 "$SERVER" 2>/dev/null || { cat "$OUT/server.log"; echo "policy server died"; exit 1; }
    sleep 10
  done
  MUJOCO_GL=egl OCCLUBENCH_VLA_BACKEND=server OCCLUBENCH_VLA_SERVER="http://127.0.0.1:$PORT/act" \
    OCCLUBENCH_VLA_UNNORM_KEY="$DATASET" .venv/bin/python scripts/run_benchmark.py --policy openvla_oft \
    --robot "$ROBOT" --max-seeds "$EVAL_SEEDS" --n-envs "$EVAL_ENVS" --horizon "$HORIZON" --image "$IMAGE" "$IMAGE" \
    --physics warp --render warp --out "$OUT/openvla_oft_ft_${ROBOT}.json" | tee -a "$OUT/log.txt"
  .venv/bin/python scripts/build_leaderboard.py --results results --out "$OUT/leaderboard.md" | tee -a "$OUT/log.txt"
fi
log "done: $OUT"
