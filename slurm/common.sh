# Sourced by every sbatch here. Hekstra-lab Cannon conventions: software on
# holylabs, bulk output on netscratch, per-job temp on node-local /scratch.
source /etc/profile.d/lmod.sh 2>/dev/null || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 FORCESKETCH_MACE_LOSS=normal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MPLBACKEND=Agg
export FORCESKETCH_SCRATCH=${FORCESKETCH_SCRATCH:-/n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch}
export FORCESKETCH_CACHES=${FORCESKETCH_CACHES:-$FORCESKETCH_SCRATCH/caches}
P=/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-gpu/bin/python
TMP=/scratch/${USER}_${SLURM_JOB_ID}; mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT; export TMPDIR="$TMP"
mkdir -p "$FORCESKETCH_CACHES" "$FORCESKETCH_SCRATCH/logs"
cd "${SLURM_SUBMIT_DIR:-/n/home05/ianpoon/forcesketch}"
echo "== $(hostname) ${SLURM_JOB_PARTITION:-?} ${SLURM_JOB_ID:-?} $(date -Is)"
nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader 2>/dev/null | head -1 || true
