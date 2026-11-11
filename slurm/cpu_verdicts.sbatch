#!/bin/bash -l
# Re-score every pre-registered protocol whose inputs were regenerated, and the
# new J10/J11/J12 analyses, on a compute node (the bootstraps exceed the
# login-node memory limit). Idempotent: skips what has no input yet.
#SBATCH -J fs-verdicts
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-04:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/n/home05/ianpoon/forcesketch}"
export OMP_NUM_THREADS=8 MPLBACKEND=Agg TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export FORCESKETCH_SCRATCH=/n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch
export FORCESKETCH_CACHES=$FORCESKETCH_SCRATCH/caches
P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
run() { echo "== $* $(date -Is)"; "$@" 2>&1 | grep -v "UserWarning\|_Jd, _W3j" ; echo "   rc=${PIPESTATUS[0]}"; }
SCAN=$FORCESKETCH_CACHES/j6b_foundation_scan.pt
if [ -f "$SCAN" ]; then
  run $P experiments/j6b_foundation_verdict.py --scan "$SCAN"
  run $P experiments/j6c_size_normalisation.py --scan "$SCAN"
  run $P experiments/j6d_complementarity.py --scan "$SCAN"
fi
run $P experiments/j10_ood_verdict.py
run $P experiments/j12_accuracy.py
run $P experiments/j12_calibration_data.py
ls results/records/j11/*.json >/dev/null 2>&1 && run $P experiments/j11_verdict.py
run $P tools/make_tables.py
run $P tools/make_figures.py
run $P tools/paper_numbers.py
echo "== done $(date -Is)"
