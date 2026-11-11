#!/bin/bash -l
# G5: rMD17 caches with reference energies (a few minutes).
#SBATCH -J fs-g5-rmd17
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:40
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
$P experiments/j5_rmd17_caches.py
touch "$FORCESKETCH_SCRATCH/done/fs-g5-rmd17"
echo "== done $(date -Is)"
