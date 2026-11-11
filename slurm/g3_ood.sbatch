#!/bin/bash -l
# G3: out-of-distribution scan of OMat24 validation subsets with all four
# foundation committees (protocols/j10_ood.yaml). Smoke run first.
#SBATCH -J fs-g3-ood
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-10:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
$P experiments/j10_ood_scan.py --smoke --batch-size 16
$P experiments/j10_ood_scan.py --batch-size 16 --chunk 2000
touch "$FORCESKETCH_SCRATCH/done/fs-g3-ood"
echo "== done $(date -Is)"
