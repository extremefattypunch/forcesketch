#!/bin/bash
#SBATCH --job-name=fs-ladder
#SBATCH --output=slurm/logs/ladder_%j.out
#SBATCH --error=slurm/logs/ladder_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm/logs
module load Mambaforge/23.11.0-fasrc01
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 FORCESKETCH_MACE_LOSS=normal
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MPLBACKEND=Agg
P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
echo "== $(hostname) ${SLURM_JOB_PARTITION} ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1
$P experiments/j5_3bpa_temperature_ladder.py
