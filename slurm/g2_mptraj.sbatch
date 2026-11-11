#!/bin/bash -l
# G2: rescan the 136,923-structure MPtraj pool with the QbC-8000 foundation
# committee, now recording reference energies and force magnitudes per structure.
# A 32-structure smoke run precedes the full scan so a schema bug fails in a
# minute rather than at the end of the job.
#SBATCH -J fs-g2-mptraj
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
$P experiments/j6b_foundation_scan.py --first-n 64 --n-sample 32 --batch-size 16 --chunk 16 \
   --out "$FORCESKETCH_CACHES/j6b_smoke.pt"
rm -f results/records/j6b_smoke_perstructure.npz
$P experiments/j6b_foundation_scan.py --batch-size 16 --chunk 4000
touch "$FORCESKETCH_SCRATCH/done/fs-g2-mptraj"
echo "== done $(date -Is)"
