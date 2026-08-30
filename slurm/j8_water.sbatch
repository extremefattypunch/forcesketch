#!/bin/bash
#SBATCH --job-name=fs-j8-water
#SBATCH --output=slurm/logs/j8w_%j.out
#SBATCH --error=slurm/logs/j8w_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --exclusive
# Whole node, GPU 0 only. gpu_test is entirely MIG and must never be used here.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
module load Mambaforge/23.11.0-fasrc01
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export FORCESKETCH_MACE_LOSS=normal
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/inductor
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton
export MPLBACKEND=Agg

P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
F=../forcesketch
echo "== node $(hostname) partition ${SLURM_JOB_PARTITION} job ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader | head -1

# Water: 192 atoms, ~17,150 edges/structure (33x 3BPA). The finer batch grid is
# there to locate the launch-bound -> compute-bound transition, which on 3BPA sat
# between B=16 and B=64.
$P experiments/j8_lane_bench.py \
   --ckpt $F/models/zenodo/water/multihead-disjoint/multihead_committee_stagetwo.model \
   --data $F/models/zenodo/water/w64_md_T300K.xyz \
   --batch-sizes 1 2 4 8 16 32 --lanes 1 2 3 4 8 \
   --iters 50 --warmup 25 \
   --out results/records/j8_lane_bench_water_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr " /" "__").jsonl
