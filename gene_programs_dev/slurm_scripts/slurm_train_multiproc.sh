#!/bin/bash
#SBATCH --job-name=set_34_QA_dataset_msigdb
#SBATCH --output /home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/train_model/set_34_%j.log
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.zhang.ddz5@yale.edu                   # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gres=gpu:1                               # start with 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb                                 # Job memory request
#SBATCH --time=1-12:00:00                          # Time limit hrs:min:sec
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
        echo "0"
    fi
}

# Function to get total number of datasets in training directory
count_total_datasets() {
    local count=0
    for dir in "${DATASET_DIR}"*/; do
        if [ -f "$dir/training_inputs.pickle" ]; then
            count=$((count+1))
        fi
    done
    echo $count
}

# Function to check if all datasets are either processed / error'd
all_datasets_done() {
    local total_datasets=$(count_total_datasets)
    local processed_count=${#processed_datasets[@]}
    echo "$(date): Processed $processed_count out of $total_datasets datasets"
    if [ $processed_count -ge $total_datasets ]; then
        return 0
    else
        return 1
    fi
}


# Testing proper gpu indexing
export GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print $1}')
# GPU_MEMORY_USED=$(nvidia-smi --id=$GPU_ID --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')

# Detect GPU type and set memory threshold
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

# Change if not A/H100 (80k was causing OOM - probably due to intermediate tensors)
GPU_MEMORY_THRESHOLD=60000
MAX_CONCURRENT_CHILDS=3
MAX_RETRY_ATTEMPTS=1

echo "Detected GPU: $GPU_TYPE at index $GPU_ID with memory threshold: $GPU_MEMORY_THRESHOLD MiB"

# test - change path
# /home/ddz5/scratch/test_cell2gsea_qa
# DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/eighth_set/"
# DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/fourth_set/"
DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models_bk/Cell2GSEA_QA_dataset_models/set_34/"
SLEEP_INTERVAL=1800  # Interval to check for processed datasets (in seconds)

# declare -A job_memory_usage # associate memory usage with PID
# declare -A job_datasets # track the dataset each job is processing

# Arrays and maps for tracking
declare -A job_memory_usage    # Associate memory usage with PID
declare -A job_datasets        # Track the dataset each job is processing
declare -A dataset_attempts    # Track number of attempts for each dataset
declare -A processed_datasets  # Track datasets that are completed or errored out
declare -A dataset_status      # Track status (success/error) for each dataset

# Main loop
current_memory=$(calculate_memory_usage $GPU_ID)
running_jobs=()

# Error log file
ERROR_LOG="${DATASET_DIR}/error_datasets.log"
OVERSIZED_LOG="${DATASET_DIR}/oversized_datasets.log"
touch "$ERROR_LOG"  # Create or clear the log file
touch "$OVERSIZED_LOG" # Create or clear oversized dataset log file

# Initial check for oversized datasets
echo "$(date): Performing initial check for oversized datasets..."
for dataset in $(ls -d $DATASET_DIR*/); do
    if [ -f "$dataset/training_inputs.pickle" ]; then
        memory_required_gb=$(calculate_memory_required "$dataset")
        memory_required_mib=$(awk "BEGIN {print int($memory_required_gb * 1024)}")

        if (( memory_required_mib > GPU_MEMORY_THRESHOLD )); then
            echo "$(date): WARNING - Dataset $dataset requires ${memory_required_mib} MiB which exceeds GPU memory threshold (${GPU_MEMORY_THRESHOLD} MiB)"
            echo "$(date): Marking dataset as 'oversized' and skipping training"
            processed_datasets["$dataset"]="oversized"
            dataset_status["$dataset"]="oversized"
            echo "[$(date)]: Dataset $dataset requires ${memory_required_mib} MiB which exceeds GPU memory threshold" >> "$OVERSIZED_LOG"
        fi
    fi
done

trap "echo 'Script terminated. Cleaning up...'; for job in \"\${running_jobs[@]}\"; do kill -9 \$job; done; exit 1" SIGINT SIGTERM
while true; do
    all_datasets_done && { 
        echo "All datasets have been processed. Exiting."
        echo "Summary of dataset statuses:"
        for dataset in "${!dataset_status[@]}"; do
            echo "  - $dataset: ${dataset_status[$dataset]}"
        done
        break
    }

    echo "$(date): Current memory used: ${current_memory} MiB"

    # Check if any jobs have finished
    for job_pid in "${running_jobs[@]}"; do
        # Skip blank entries
        if [[ -z "$job_pid" ]]; then
            echo "$(date): Skipping empty job PID in running_jobs"
            continue
        fi
        
        if ! kill -0 "$job_pid" 2>/dev/null; then
            dataset="${job_datasets[$job_pid]}"
            echo "$(date): Job $job_pid for dataset $dataset terminated. Checking status..."
            
            # Check if the job completed successfully (training_done.flag exists)
            if [ -f "${dataset}/training_done.flag" ]; then
                echo "$(date): Job $job_pid completed successfully. Dataset $dataset is done."
                processed_datasets["$dataset"]="success"
                dataset_status["$dataset"]="success"
            else
                # Job failed
                current_attempts=${dataset_attempts["$dataset"]}
                current_attempts=$((current_attempts + 1))
                dataset_attempts["$dataset"]=$current_attempts
                
                echo "$(date): Job $job_pid failed. Dataset $dataset has failed $current_attempts times."
                
                # Check if max retry attempts reached
                if [ $current_attempts -ge $MAX_RETRY_ATTEMPTS ]; then
                    echo "$(date): Dataset $dataset has reached maximum retry attempts ($MAX_RETRY_ATTEMPTS). Marking as errored."
                    processed_datasets["$dataset"]="error"
                    dataset_status["$dataset"]="error"
                    echo "[$(date)]: Dataset $dataset failed after $current_attempts attempts" >> "$ERROR_LOG"
                else
                    echo "$(date): Will retry dataset $dataset later."
                fi
            fi
            
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

    # Check if we are at max concurrent jobs
    if (( ${#running_jobs[@]} >= MAX_CONCURRENT_CHILDS )); then
        echo "$(date): Maximum concurrent jobs reached. ${#running_jobs[@]} / ${MAX_CONCURRENT_CHILDS}"
    else
        # Process eligible datasets
        for dataset in $(ls -d $DATASET_DIR*/); do
            # Skip already processed datasets
            if [[ -n "${processed_datasets[$dataset]}" ]]; then
                continue
            fi
            
            # Skip datasets without training inputs
            if [ ! -f "$dataset/training_inputs.pickle" ]; then
                continue
            fi
            
            # Skip datasets that are already done
            if [ -f "$dataset/training_done.flag" ]; then
                echo "$(date): Dataset $dataset already has training_done.flag. Marking as processed."
                processed_datasets["$dataset"]="success"
                dataset_status["$dataset"]="success"
                continue
            fi
            
            # Check if dataset is already being processed
            dataset_being_processed=0
            for job in "${running_jobs[@]}"; do
                if [[ "${job_datasets[$job]}" == "$dataset" ]]; then
                    dataset_being_processed=1
                    break
                fi
            done

            if [ $dataset_being_processed -eq 0 ]; then
                # Check again for max concurrent jobs (safety check)
                if (( ${#running_jobs[@]} >= MAX_CONCURRENT_CHILDS )); then
                    echo "$(date): Maximum concurrent jobs reached inside dataset loop. Breaking."
                    break
                fi
                
                # Calculate memory requirement
                memory_required_gb=$(calculate_memory_required "$dataset")
                memory_required_mib=$(awk "BEGIN {print int($memory_required_gb * 1024)}")

                echo "$(date): Dataset ${dataset} requires ${memory_required_mib} MiB of memory"
                # Check if oversized
                if (( memory_required_mib > GPU_MEMORY_THRESHOLD )); then
                    echo "$(date): Dataset $dataset requires ${memory_required_mib} MiB which exceeds GPU memory threshold (${GPU_MEMORY_THRESHOLD} MiB)"
                    echo "$(date): Marking dataset as 'oversized' and skipping training"
                    processed_datasets["$dataset"]="oversized"
                    dataset_status["$dataset"]="oversized"
                    echo "[$(date)]: Dataset $dataset requires ${memory_required_mib} MiB which exceeds GPU memory threshold" >> "$OVERSIZED_LOG"
                    continue
                fi

                if (( current_memory + memory_required_mib < GPU_MEMORY_THRESHOLD )); then
                    # Initialize attempt counter if first try
                    if [[ -z "${dataset_attempts[$dataset]}" ]]; then
                        dataset_attempts["$dataset"]=0
                    fi
                    
                    echo "$(date): Starting training for $dataset on GPU (attempt ${dataset_attempts[$dataset]})"
                    /gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_training.py \
                        --input_path "${dataset}/training_inputs.pickle" \
                        --output_prefix "${DATASET_DIR}" \
                        --dataset_name "$(basename $dataset)" \
                        --seed 13 &

                    job_pid=$!
                    running_jobs+=($job_pid)
                    job_memory_usage[$job_pid]=$memory_required_mib
                    job_datasets[$job_pid]=$dataset

                    # Preemptively update memory with heuristic
                    current_memory=$((current_memory + memory_required_mib))
                    echo "$(date): Memory updated preemptively: ${current_memory} MiB"
                    
                    # Check if we're at the limit after adding this job
                    if (( ${#running_jobs[@]} >= MAX_CONCURRENT_CHILDS )); then
                        echo "$(date): Maximum concurrent jobs reached after adding job. Breaking dataset loop."
                        break
                    fi
                else
                    echo "$(date): Not enough GPU memory for $dataset. Current: ${current_memory} MiB, Required: ${memory_required_mib} MiB"

                    continue
                fi
            fi
        done
    fi
    
    # Print current status
    echo "$(date): Current memory usage: ${current_memory} MiB"
    echo "$(date): Memory threshold: ${GPU_MEMORY_THRESHOLD} MiB"
    echo "$(date): Currently running jobs: ${#running_jobs[@]}"
    for job in "${running_jobs[@]}"; do
        echo "$(date): Job ${job} processing ${job_datasets[$job]}: ${job_memory_usage[$job]} MiB"
    done
    
    echo "$(date): Processed datasets: ${#processed_datasets[@]}"
    echo "$(date): Dataset statuses:"
    for ds in "${!dataset_status[@]}"; do
        echo "  - $(basename $ds): ${dataset_status[$ds]}"
    done
    
    echo "$(date): Sleeping for $SLEEP_INTERVAL seconds..."
    sleep $SLEEP_INTERVAL
done

# Final report
echo "=== Final Processing Report ==="
echo "Total datasets with training_inputs.pickle: $(count_total_datasets)"
echo "Total datasets processed: ${#processed_datasets[@]}"
echo "Successful datasets: $(grep -c "success" <<< "$(printf '%s\n' "${dataset_status[@]}")")"
echo "Failed datasets: $(grep -c "error" <<< "$(printf '%s\n' "${dataset_status[@]}")")"
echo "Oversized datasets: $(grep -c "oversized" <<< "$(printf '%s\n' "${dataset_status[@]}")")"
echo "See $ERROR_LOG for details on failed datasets"
echo "See $OVERSIZED_LOG for details on oversized datasets"