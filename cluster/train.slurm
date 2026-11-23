#!/usr/bin/env bash
#SBATCH --job-name=mario-ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=mario-ppo-%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project root with sbatch}"
python="${MARIO_VENV:-$PWD/.venv-cluster}/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Python environment missing at $python. Run bash cluster/setup.sh first." >&2
    exit 1
fi

run_dir="${MARIO_RUN_ROOT:-runs/cluster}/${SLURM_JOB_ID:?Missing Slurm job ID}"
mkdir -p "$run_dir"
export MPLCONFIGDIR="$run_dir/matplotlib"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Default checkpoint: $run_dir/latest.zip"
exec "$python" -u train_ppo.py train --device cuda --envs 4 --verbose \
    --steps 102400 --checkpoint-every 24576 --output "$run_dir/latest.zip" "$@"
