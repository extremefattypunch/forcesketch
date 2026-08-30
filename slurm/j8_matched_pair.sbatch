#!/bin/bash
#SBATCH --job-name=fs-j8-pair
#SBATCH --output=slurm/logs/j8pair_%j.out
#SBATCH --error=slurm/logs/j8pair_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --exclusive
# BOTH systems in ONE job on ONE exclusive node. The matched-pair claim is that
# water and 3BPA were measured on the same device under the same conditions;
# running them as separate jobs cannot guarantee that, and a co-tenant on the
# node doubled absolute times once already (23.07 -> 60.75 ms at L=1, SM clock
# down to 780 MHz).
#
# EXPECT A LONG QUEUE WAIT, and budget for it rather than rediscovering it.
# MEASURED on 2026-08-16: submitted 02:26, started 04:45 (A100) and 05:00 (H200)
# -- about 2h20m pending on (Resources) -- then ran 30 min and 17 min.
#
# Why: --exclusive makes SLURM drain a WHOLE node, so it waits for all 4 GPUs to
# free even though this job uses one. Allocation actually granted was 64 CPUs +
# 4 GPUs (A100 node) and 112 CPUs (H200 node), against a job that runs on
# CUDA_VISIBLE_DEVICES=0 with 16 threads. Billing 3411 / 10671.
#
# It is not removable for a SCORED run: protocols/j8_water_prediction*.yaml set
# `require_exclusive_node: true`, so a co-tenanted run is inadmissible by
# pre-registration, and that rule exists because co-tenancy already destroyed one
# grid. For exploratory timing, drop --exclusive and check `sm_clock_mhz` in the
# records instead -- these clean runs sat at 1410 MHz (A100) and 1980 MHz (H200),
# i.e. maximum, which is the signature to look for.
#
# --mem is deliberately omitted: under --exclusive the whole node's memory is
# allocated regardless, so stating a number only misleads.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm/logs
module load Mambaforge/23.11.0-fasrc01
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 FORCESKETCH_MACE_LOSS=normal
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MPLBACKEND=Agg
P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
DEV=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr " /" "__")
echo "== $(hostname) ${SLURM_JOB_PARTITION} ${SLURM_JOB_ID} device=$DEV"
nvidia-smi --query-gpu=name,mig.mode.current,clocks.max.sm --format=csv,noheader | head -1

# 3BPA reference, then water, same process conditions, more iters than the
# first pass so the noisy load-bearing cell that made the H200 run inadmissible
# has a chance to settle.
$P experiments/j8_lane_bench.py --batch-sizes 1 4 16 64 --lanes 1 2 3 4 8 \
   --iters 100 --warmup 50 \
   --out results/records/j8_lane_bench_${DEV}_pair.jsonl

$P experiments/j8_lane_bench.py \
   --ckpt ../forcesketch/models/zenodo/water/multihead-disjoint/multihead_committee_stagetwo.model \
   --data ../forcesketch/models/zenodo/water/w64_md_T300K.xyz \
   --batch-sizes 1 2 4 8 16 32 --lanes 1 2 3 4 8 \
   --iters 100 --warmup 50 \
   --out results/records/j8_lane_bench_water_${DEV}_pair.jsonl
