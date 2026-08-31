#!/bin/bash
#SBATCH --job-name=fs-j6-naive
#SBATCH --output=slurm/logs/j6_%j.out
#SBATCH --error=slurm/logs/j6_%j.err
#SBATCH --partition=gpu,gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Not a timing job -- this only produces float64 head-force caches, so
# co-tenancy is irrelevant and no exclusive node is requested. MIG is fine too.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
module load Mambaforge/23.11.0-fasrc01
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 FORCESKETCH_MACE_LOSS=normal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MPLBACKEND=Agg
P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
echo "== $(hostname) ${SLURM_JOB_PARTITION} ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1

$P experiments/j6_naive_caches.py --batch-size 4 --device cuda
