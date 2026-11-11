#!/bin/bash -l
# Offline reproduction tier on a compute node: the paired block bootstrap needs
# more memory than a login-node process is allowed.
#SBATCH -J fs-reproduce
#SBATCH -p test
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/n/home05/ianpoon/forcesketch}"
export OMP_NUM_THREADS=4
/n/holylabs/hekstra_lab/Everyone/ianpoon/envs/fs-cpu/bin/python tools/reproduce.py --tier offline
