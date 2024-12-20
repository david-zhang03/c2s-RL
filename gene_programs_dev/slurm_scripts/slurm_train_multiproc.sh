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
# declare -A GPU_MEMORY_THRESHOLDS
# GPU_MEMORY_THRESHOLDS["A100"]=80000
# GPU_MEMORY_THRESHOLDS["H100"]=80000
# GPU_MEMORY_THRESHOLDS["A40"]=48000

# GPU memory thresholds per model
# include memory requirements for dataset
# Detect GPU type and set memory threshold
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
# GPU_MEMORY_THRESHOLD=${GPU_MEMORY_THRESHOLDS[$GPU_TYPE]}
# if [ -z "$GPU_MEMORY_THRESHOLD" ]; then
#     echo "Unknown GPU type: $GPU_TYPE"
#     exit 1
# fi
# Change if not A/H100
GPU_MEMORY_THRESHOLD=80000

echo "Detected GPU: $GPU_TYPE with memory threshold: $GPU_MEMORY_THRESHOLD MiB"

# Function to check GPU memory usage - NOT USED
# calculate_baseline_memory_usage() {
#     local gpu_index=$1
#     local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu_index + 1))p")
#     echo $memory_used
# }
check_memory_usage() {
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
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

DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/"
SLEEP_INTERVAL=1800  # Interval to check for processed datasets (in seconds)

# Function to check if all datasets are done
all_datasets_done() {
    for dataset in $(ls -d $DATASET_DIR/*/); do
        if [ ! -f "$dataset/training_done.flag" ]; then
            return 1  # At least one dataset is not done
        fi
    done
    return 0  # All datasets are done
}

# Main loop
# specify the index of the gpu on node
# gpu_index=3
running_jobs=()
while true; do
    all_datasets_done
    if [ $? -eq 0 ]; then
        echo "All datasets have been trained. Exiting."
        break
    fi

    # baseline memory used
    current_memory=87
    for dataset in $(ls -d $DATASET_DIR/*/); do
        if [ -f "$dataset/training_inputs.pickle" ] && [ ! -f "$dataset/training_done.flag" ]; then
            # Check GPU memory usage
            # current_memory=$(check_memory_usage)

            memory_required_gb=$(calculate_memory_required "$dataset")
            memory_required_mib=$(awk "BEGIN {print int($memory_required_gb * 1024)}")

            if (( current_memory < GPU_MEMORY_THRESHOLD )); then
                echo "Starting training for $dataset on GPU"
                /gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_training.py \
                    --input_path "${dataset}/training_inputs.pickle" \
                    --output_prefix "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/" \
                    --dataset_name "$(basename $dataset)" &

                running_jobs+=($!)
                touch "$dataset/training_done.flag"
                
                current_memory=$((current_memory + memory_required_mib))
            else
                echo "Not enough GPU memory for $dataset. Reset current memory. Waiting..."
                break
            fi
        fi
    done

    # Wait for some jobs to finish if GPU memory is full
    for job in "${running_jobs[@]}"; do
        if ! kill -0 $job 2>/dev/null; then
            running_jobs=("${running_jobs[@]/$job}")
        fi
    done
    sleep $SLEEP_INTERVAL
done

