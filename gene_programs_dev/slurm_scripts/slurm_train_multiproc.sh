#!/bin/bash
#SBATCH --job-name=QA_dataset_msigdb
#SBATCH --output /home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/train_model/slurm_%j.log
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.zhang.ddz5@yale.edu                   # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gres=gpu:1                               # start with 1
#SBATCH --constraint=h100|a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=512gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname

module load miniconda
conda activate /gpfs/radev/home/sr2464/.conda/envs/llamp/

nvidia-smi

# Helpers
calculate_memory_usage() {
    local gpu_index=$1
    local memory_used=$(nvidia-smi --id=$gpu_index --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    echo $memory_used
}

# Function to calculate memory usage for dataset
calculate_memory_required() {
    local metadata_file="$1/metadata.txt"
    if [ -f "$metadata_file" ]; then
        local num_cells=$(grep "num_cells" "$metadata_file" | awk '{print $2}')
        # heuristic: 0.5GB/ 10k cells
        local memory_required_gb=$(awk "BEGIN {print ($num_cells / 10000) * 0.5}")
        echo "$memory_required_gb"
    else
        echo "$0"
    fi
}

# Function to check if all datasets are done
all_datasets_done() {
    for dataset in $(ls -d $DATASET_DIR*/); do
        if [ ! -f "$dataset/training_done.flag" ]; then
            return 1  # At least one dataset is not done
        fi
    done
    return 0  # All datasets are done
}


# Testing proper gpu indexing

export GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print $1}')
# GPU_MEMORY_USED=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')

# Detect GPU type and set memory threshold
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

# Change if not A/H100 (80k was causing OOM - probably due to intermediate tensors)
GPU_MEMORY_THRESHOLD=60000
MAX_CONCURRENT_CHILDS=3

echo "Detected GPU: $GPU_TYPE at index $GPU_ID with memory threshold: $GPU_MEMORY_THRESHOLD MiB"

# test - change path
# /home/ddz5/scratch/test_cell2gsea_qa
DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/"
SLEEP_INTERVAL=1800  # Interval to check for processed datasets (in seconds)

declare -A job_memory_usage # associate memory usage with PID
declare -A job_datasets # track the dataset each job is processing

# Main loop
current_memory=$(calculate_memory_usage $GPU_INDEX)
running_jobs=()

trap "echo 'Script terminated. Cleaning up...'; exit 1" SIGINT SIGTERM
while true; do
    all_datasets_done && { echo "All datasets have been trained. Exiting."; break; }

    echo "$(date): Current memory used: ${current_memory} MiB"

    # Check if any jobs have finished
    for job_pid in "${running_jobs[@]}"; do
        # Skip blank entries - shouldn't happen
        if [[ -z "$job_pid" ]]; then
            echo "$(date): Skipping empty job PID in running_jobs"
            continue
        fi
        if ! kill -0 "$job_pid" 2>/dev/null; then
            wait "$job_pid"
            exit_status=$?
            if [[ $exit_status -eq 0 ]]; then
                echo "$(date): Job $job_pid completed successfully."
            else
                echo "$(date): Job $job_pid terminated with errors. Exit status: $exit_status"

            # Free memory and update job tracking
            if [[ -n "${job_memory_usage[$job_pid]}" ]]; then
                memory_freed=${job_memory_usage[$job_pid]}
                current_memory=$((current_memory - memory_freed))
                echo "$(date): Freed ${memory_freed} MiB from job $job_pid. Updated memory: ${current_memory} MiB"
            else
                echo "$(date): Warning: job_memory_usage[$job_pid] not found. Skipping memory update for this job."
            fi
            # Remove job from tracking
            running_jobs=($(for pid in "${running_jobs[@]}"; do [[ "$pid" != "$job_pid" && -n "$pid" ]] && echo "$pid"; done))
            unset job_memory_usage[$job_pid]
            unset job_datasets[$job_pid]
        fi
    done

    # check if we are at max concurrent jobs
    if (( ${#running_jobs[@]} >= MAX_CONCURRENT_CHILDS )); then
        echo "$(date): Maximum concurrent jobs reached. ${#running_jobs[@]} / ${MAX_CONCURRENT_CHILDS}"
    else
        # baseline memory used
        for dataset in $(ls -d $DATASET_DIR*/); do
            if [ -f "$dataset/training_inputs.pickle" ] && [ ! -f "$dataset/training_done.flag" ]; then
                # Check if dataset is already being processed
                dataset_being_processed=0
                for job in "${running_jobs[@]}"; do
                    if [[ "${job_datasets[$job]}" == "$dataset" ]]; then
                        dataset_being_processed=1
                        break
                    fi
                done

                if [ $dataset_being_processed -eq 0 ]; then
                    # this is a heuristic calculation as we do not know how much memory will be used until loading
                    memory_required_gb=$(calculate_memory_required "$dataset")
                    memory_required_mib=$(awk "BEGIN {print int($memory_required_gb * 1024)}")

                    echo "$(date): Dataset ${dataset} requires ${memory_required_mib} MiB of memory"

                    if (( current_memory + memory_required_mib < GPU_MEMORY_THRESHOLD )) && (( ${#running_jobs[@]} < MAX_CONCURRENT_CHILDS )); then
                        echo "$(date): Starting training for $dataset on GPU"
                        /gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_training.py \
                            --input_path "${dataset}/training_inputs.pickle" \
                            --output_prefix "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/" \
                            --dataset_name "$(basename $dataset)" &

                        job_pid=$!
                        running_jobs+=($job_pid)
                        job_memory_usage[$job_pid]=$memory_required_mib
                        job_datasets[$job_pid]=$dataset

                        # Preemptively update memory with heuristic
                        current_memory=$((current_memory + memory_required_mib))
                        echo "$(date): Memory updated preemptively: ${current_memory} MiB"
                    else
                        echo "$(date): Not enough GPU memory for $dataset. Current: ${current_memory} MiB, Required: ${memory_required_mib} MiB"
                        continue
                    fi
                fi
            fi
        done
    fi
    echo "$(date): Current memory usage: ${current_memory} MiB"
    echo "$(date): Memory threshold: ${GPU_MEMORY_THRESHOLD} MiB"
    echo "$(date): Currently running jobs: ${#running_jobs[@]}"
    for job in "${running_jobs[@]}"; do
        echo "$(date): Job ${job}: ${job_memory_usage[$job]} MiB"
    done
    
    sleep $SLEEP_INTERVAL
done
