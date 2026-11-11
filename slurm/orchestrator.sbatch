#!/bin/bash -l
# Keeps the free gpu_test slots busy without anyone watching.
#
# gpu_test's QOS allows 2 jobs per user and rejects --dependency chains, so a
# queue of more than two jobs cannot simply be submitted. This 1-core job on the
# free `test` partition polls every 2 minutes, and whenever fewer than 2 of our
# jobs are on gpu_test it submits the next line of slurm/jobs.txt whose done
# marker ($FORCESKETCH_SCRATCH/done/<job-name>) does not exist yet. Each worker
# sbatch touches its own marker as its last command, so a job that died is
# resubmitted on the next pass (at most MAX_RETRY times). `test` caps jobs at
# 12 h, so the orchestrator resubmits itself shortly before that.
#SBATCH -J fs-orch
#SBATCH -p test
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -t 12:00:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.out
#SBATCH -e /n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch/logs/%x-%j.err
set -uo pipefail
REPO=/n/home05/ianpoon/forcesketch
S=${FORCESKETCH_SCRATCH:-/n/netscratch/hekstra_lab/Lab/ian_poon/forcesketch}
JOBS=${JOBS:-$REPO/slurm/jobs.txt}; MAX_RETRY=${MAX_RETRY:-2}; MAX_TEST=${MAX_TEST:-2}
mkdir -p "$S/done" "$S/logs"
cd "$REPO"
START=$(date +%s); LIMIT=$((11*3600+40*60))
while true; do
  if [ $(( $(date +%s) - START )) -gt $LIMIT ]; then
    echo "$(date -Is) resubmitting orchestrator"
    until jid=$(sbatch --parsable "$REPO/slurm/orchestrator.sbatch"); do sleep 60; done
    echo "   orchestrator job $jid"; exit 0
  fi
  if ! jobs_now=$(squeue -u "$USER" -h -o "%j %P"); then
    echo "$(date -Is) squeue failed; skipping this pass"; sleep 120; continue
  fi
  on_test=$(awk '$1 ~ /^fs-/ && $2 ~ /(^|,)gpu_test(,|$)/' <<<"$jobs_now" | wc -l)
  pending=0; submitted=0
  while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    script=$(awk '{print $1}' <<<"$line")
    name=$(grep -m1 -oE '^#SBATCH -J [^ ]+' "$REPO/$script" | awk '{print $3}')
    [ -z "$name" ] && { echo "no -J in $script"; continue; }
    [ -f "$S/done/$name" ] && continue
    pending=$((pending+1))
    awk '{print $1}' <<<"$jobs_now" | grep -qx "$name" && continue
    tries=$(cat "$S/done/$name.tries" 2>/dev/null || echo 0)
    [ "$tries" -ge "$MAX_RETRY" ] && { echo "$(date -Is) $name failed $tries times; giving up"; continue; }
    [ "$on_test" -ge "$MAX_TEST" ] && continue
    jid=$(sbatch --parsable "$REPO/$script") && echo "$(date -Is) submit $name -> $jid" || { echo "sbatch FAILED for $script"; continue; }
    echo $((tries+1)) > "$S/done/$name.tries"
    on_test=$((on_test+1)); submitted=$((submitted+1))
  done < "$JOBS"
  [ "$pending" -eq 0 ] && { echo "$(date -Is) all jobs in $JOBS complete"; exit 0; }
  [ "$submitted" -gt 0 ] && sleep 180 || sleep 120
done
