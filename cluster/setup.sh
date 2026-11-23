#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${MARIO_VENV:-$project_dir/.venv-cluster}"
cuda_wheel="${CUDA_WHEEL:-cu126}"

case "$cuda_wheel" in
    cu118|cu126|cu128) ;;
    *) echo "CUDA_WHEEL must be cu118, cu126, or cu128." >&2; exit 2 ;;
esac

"${PYTHON_BIN:-python3.10}" -m venv "$venv"
python="$venv/bin/python"
"$python" -m pip install 'pip<26' wheel 'setuptools<81'
"$python" -m pip install "torch==2.7.1+${cuda_wheel}" \
    --index-url "https://download.pytorch.org/whl/$cuda_wheel"
"$python" -m pip install -r "$project_dir/cluster/requirements.txt"
"$python" -m pip check
"$python" -c 'import cv2, gym_super_mario_bros, stable_baselines3, torch; print("PyTorch", torch.__version__, "CUDA build", torch.version.cuda)'
