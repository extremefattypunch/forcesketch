#!/bin/bash -l
# G1: regenerate every head-force cache the journal analyses read (3BPA ladder,
# water, naive committees). Correctness only -- MIG is fine, timing is not measured.
#SBATCH -J fs-g1-caches
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-03:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
$P experiments/j5_3bpa_temperature_ladder.py --batch-size 8
$P experiments/j5_water_caches.py --batch-size 4
$P experiments/j6_naive_caches.py --batch-size 4 --device cuda
ls -la "$FORCESKETCH_CACHES" "$FORCESKETCH_CACHES/naive"
echo "== done $(date -Is)"
