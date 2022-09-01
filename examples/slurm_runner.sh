set -e -x

slurm_script="$1"
results_dir="$2"
model_name="$3"

sbatch_command="sbatch --nv-meta ml-model.${model_name} ${slurm_script}"

output=$(eval ${sbatch_command})
job_id=$(echo ${output} | grep -oP "\d+")
echo "Current job id: ${job_id}"

sleep 5

while [ ! -z "$(squeue | grep "${job_id}")" ]; do
  sleep 600
done

path_to_errors="$(find "${results_dir}" -name "error-${job_id}-0.out")"

while [ ! -z "$(grep 'DUE TO TIME LIMIT ***' "${path_to_errors}")" ]; do
  echo RESTARTING
  output=$(eval "${sbatch_command}")
  job_id=$(echo ${output} | grep -oP "\d+")
  echo "Current job id: ${job_id}"

  sleep 5

  while [ ! -z "$(squeue | grep "${job_id}")" ]; do
    sleep 600
  done

  path_to_errors="$(find "${results_dir}" -name "error-${job_id}-0.out")"
done

set +e +x