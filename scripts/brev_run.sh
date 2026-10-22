#!/usr/bin/env bash
# OccluBench GPU run for a fresh NVIDIA Brev H100 / A100 instance (Ubuntu + NVIDIA driver).
#
#   git clone https://github.com/dundysm/occlubench && cd occlubench
#   bash scripts/brev_run.sh            # writes results/brev_<timestamp>/{results.json,results.md}
#
# Env overrides: ROBOT=gantry|panda (default: both), ENVS="64 128 256 512", IMAGE="128 128",
#                GRID_SEEDS=256, GRID_ENVS=256, PYTHON=python3
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
ENVS="${ENVS:-64 128 256 512}"
IMAGE="${IMAGE:-128 128}"
GRID_SEEDS="${GRID_SEEDS:-256}"
GRID_ENVS="${GRID_ENVS:-256}"
ROBOTS="${ROBOT:-gantry panda}"
OUT="results/brev_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

echo "== system" | tee "$OUT/log.txt"
nvidia-smi | tee -a "$OUT/log.txt" || { echo "nvidia-smi not found: is this a GPU instance?"; exit 1; }

NEED=""
for pkg in libegl1 libgl1 libosmesa6 git; do dpkg -s "$pkg" >/dev/null 2>&1 || NEED="$NEED $pkg"; done
"$PYTHON" -c "import ensurepip" >/dev/null 2>&1 || NEED="$NEED python3-venv $("$PYTHON" -c 'import sys;print(f"python{sys.version_info[0]}.{sys.version_info[1]}-venv")')"
if [ -n "$NEED" ]; then
  sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y $NEED || true
fi

echo "== venv" | tee -a "$OUT/log.txt"
[ -x .venv/bin/python ] || { rm -rf .venv; "$PYTHON" -m venv .venv; }
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev,warp]"
python - <<'PY' | tee -a "$OUT/log.txt"
import mujoco, warp as wp, mujoco_warp
wp.init()
print("mujoco", mujoco.__version__, "| warp", wp.__version__, "| devices", [str(d) for d in wp.get_devices()])
PY
python -m occlubench.menagerie | tee -a "$OUT/log.txt"   # fetch the Panda model once

echo "== smoke tests" | tee -a "$OUT/log.txt"
: > "$OUT/smoke.jsonl"
for robot in $ROBOTS; do
  # shellcheck disable=SC2086
  MUJOCO_GL=egl python scripts/smoke_test.py --robot "$robot" --n-envs $ENVS --physics warp --render warp \
    --image $IMAGE --steps 50 | tee -a "$OUT/smoke.jsonl" || echo "{\"robot\": \"$robot\", \"error\": \"smoke test failed\"}" >> "$OUT/smoke.jsonl"
done

echo "== small grid (ColorCue, gantry)" | tee -a "$OUT/log.txt"
# shellcheck disable=SC2086
MUJOCO_GL=egl python scripts/run_grid.py --policy colorcue --robot gantry --seeds "$GRID_SEEDS" --n-envs "$GRID_ENVS" \
  --physics warp --render warp --image 64 64 --severities 0.5 1.0 --onsets 0.0 0.15 \
  --out "$OUT/grid_records.jsonl" | tee "$OUT/grid.md" || echo "grid failed" | tee -a "$OUT/grid.md"

python - "$OUT" <<'PY'
import json, pathlib, subprocess, sys
out = pathlib.Path(sys.argv[1])
smoke = [json.loads(l) for l in (out / "smoke.jsonl").read_text().splitlines() if l.strip().startswith("{")]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
res = dict(gpu=gpu, smoke=smoke, grid_markdown=(out / "grid.md").read_text())
(out / "results.json").write_text(json.dumps(res, indent=2))
lines = [f"# OccluBench Brev run\n\nGPU: `{gpu}`\n", "| robot | n_envs | env-steps/s | GPU mem used (MiB) | wall (s) |", "|---|---|---|---|---|"]
for s in smoke:
    if "error" in s:
        lines.append(f"| {s.get('robot')} | - | error | - | - |")
        continue
    mem = s["gpu"][0]["used"] if s.get("gpu") else "-"
    lines.append(f"| {s['robot']} | {s['n_envs']} | {s['env_steps_per_s']} | {mem} | {s['wall_s']} |")
lines += ["", "## Grid (ColorCue, gantry)", "", res["grid_markdown"]]
(out / "results.md").write_text("\n".join(lines))
print("\n".join(lines))
PY
echo "== done: paste $OUT/results.md (and results.json) back"
