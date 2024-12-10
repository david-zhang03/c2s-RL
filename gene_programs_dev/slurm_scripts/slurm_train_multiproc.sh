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
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb                                 # Job memory request
#SBATCH --time=1-00:00:00                          # Time limit hrs:min:sec
date;hostname

source /home/ddz5/miniconda3/bin/activate activate  /home/ddz5/.conda/envs/c2gsea 

nvidia-smi
declare -A GPU_MEMORY_THRESHOLDS
GPU_MEMORY_THRESHOLDS["A100"]=20000  # These are just placeholders, actual benchmarking IP
GPU_MEMORY_THRESHOLDS["H100"]=16000  
GPU_MEMORY_THRESHOLDS["A40"]=16000

# GPU memory thresholds per model
# include memory requirements for dataset
# Detect GPU type and set memory threshold
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_MEMORY_THRESHOLD=${GPU_MEMORY_THRESHOLDS[$GPU_TYPE]}
if [ -z "$GPU_MEMORY_THRESHOLD" ]; then
    echo "Unknown GPU type: $GPU_TYPE"
    exit 1
fi

echo "Detected GPU: $GPU_TYPE with memory threshold: $GPU_MEMORY_THRESHOLD MiB"

# Function to check GPU memory usage
check_memory_usage() {
    local memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    echo $memory_used
}

DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/"
SLEEP_INTERVAL=30  # Interval to check for processed datasets (in seconds)

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
while true; do
    all_datasets_done
    if [ $? -eq 0 ]; then
        echo "All datasets have been trained. Exiting."
        break
    fi

    for dataset in $(ls -d $DATASET_DIR/*/); do
        if [ -f "$dataset/training_inputs.pickle" ] && [ ! -f "$dataset/training_done.flag" ]; then
            # Check GPU memory usage
            current_memory=$(check_memory_usage)
            if (( current_memory + GPU_MEMORY_THRESHOLD < 80000 )); then
                echo "Starting training for $dataset on GPU"
                python /home/ddz5/Desktop/gene_programs_dev/scripts/run_training.py \
                    --input_path "${dataset}/training_inputs.pickle" \
                    --output_prefix "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/" \
                    --dataset_name "$(basename $dataset)" &&
                touch "$dataset/training_done.flag"
            else
                echo "Not enough GPU memory for $dataset. Waiting..."
            fi
        fi
    done
    sleep $SLEEP_INTERVAL
done

