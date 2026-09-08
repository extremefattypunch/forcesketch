#!/bin/bash
#SBATCH --job-name=fs-j8-bench
#SBATCH --output=slurm/logs/j8_%j.out
#SBATCH --error=slurm/logs/j8_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive
# --exclusive plus all GPUs on the node: co-tenants on an SXM board share a power
# budget and host bandwidth, which is enough to move a 30 ms measurement. We use
# GPU 0 only. NEVER use gpu_test here: it is entirely MIG a100_3g.20gb slices.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

module load Mambaforge/23.11.0-fasrc01
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1     # e3nn 0.4.4 bare torch.load at import
export FORCESKETCH_MACE_LOSS=normal           # reach the committee branch (eval only)
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton
export MPLBACKEND=Agg

P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
echo "== node $(hostname) partition ${SLURM_JOB_PARTITION} job ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total,mig.mode.current --format=csv

$P experiments/j0b_batched_vjp_guard.py
$P experiments/j8_lane_bench.py --batch-sizes 1 4 16 64 --lanes 1 2 3 4 8
