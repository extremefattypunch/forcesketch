#!/bin/bash -l
# G4 seed 1: train one single-head MACE per selection rule on 100 + 50 frames
# (protocols/j11_active_learning.yaml) and evaluate on the common held-out set.
#SBATCH -J fs-g4-al-s1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
source slurm/common.sh
test -f "$FORCESKETCH_SCRATCH/al/selection.json" || $P experiments/j11_active_learning.py --phase select
for sel in none global maxcomp maxatom force_norm energy_std random; do
  $P experiments/j11_active_learning.py --phase train --selectors $sel --seeds 1
  $P experiments/j11_active_learning.py --phase evaluate --selectors $sel --seeds 1
done
touch "$FORCESKETCH_SCRATCH/done/fs-g4-al-s1"
echo "== done $(date -Is)"
