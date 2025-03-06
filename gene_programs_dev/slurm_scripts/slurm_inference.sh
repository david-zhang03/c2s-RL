#!/bin/bash
# Script to run inference on trained models across multiple datasets

# Base directory containing all datasets
DATASET_DIR="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models_bk/Cell2GSEA_QA_dataset_models"
# Directory where inference results will be saved
RESULTS_DIR="/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"
# Path to your inference script
INFERENCE_SCRIPT="/home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_inference.py"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Log file for the script
LOG_FILE="/home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/inference/inference_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting inference on trained models"
log "Scanning directory: $DATASET_DIR"

# Count for successful and failed inferences
successful_count=0
failed_count=0

# Iterate through each set directory (set_*)
for set_dir in "$DATASET_DIR"/set_*/; do
    if [ -d "$set_dir" ]; then
        set_name=$(basename "$set_dir")
        log "Processing set: $set_name"
        
        # Create a directory for this set's results
        set_results_dir="$RESULTS_DIR/$set_name"
        mkdir -p "$set_results_dir"
        
        # Iterate through each dataset directory within this set
        for dataset_dir in "$set_dir"/*/; do
            if [ -d "$dataset_dir" ]; then
                dataset_name=$(basename "$dataset_dir")
                log "Checking dataset: $dataset_name"
                
                # Initialize variable to track if we found a complete training
                found_complete_training=false
                
                # Check if this dataset has been trained
                if [ -f "$dataset_dir/training_done.flag" ] && [ -f "$dataset_dir/training_output.pickle" ]; then
                    # Check if dataset has been trained successfully (exists a UMAP at epoch 249)
                    for training_run_dir in "$dataset_dir"/*; do
                        if [ -d "$training_run_dir" ]; then
                            if ls "$training_run_dir"/umap_validation_epoch_249.png 1> /dev/null 2>&1; then
                                log "Found completed training with UMAP 249 in: $training_run_dir"

                                model_path="$training_run_dir/best_model_checkpoint.pt"

                                # Make sure the model file exists
                                if [ ! -f "$model_path" ]; then
                                    log "ERROR: Could not find model checkpoint in $training_run_dir"
                                    continue
                                fi

                                inputs_path="$dataset_dir/training_inputs.pickle"
                                inference_csv_results_dir="$set_results_dir/${dataset_name}_inference"
                                
                                mkdir -p "$inference_csv_results_dir"

                                log "Running inference on: $dataset_name"
                                log "Model path: $model_path"
                                log "Inputs path: $inputs_path"
                                log "Outputs will be saved to: $dataset_dir"
                                log "Output CSV will be saved to: $inference_csv_results_dir"

                                /gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_inference.py \
                                    --model_path "${model_path}" \
                                    --prepared_train_inputs "${inputs_path}" \
                                    --output_saved_path "${dataset_dir}" \
                                    --output_csv_save_path "${inference_csv_results_dir}"
                                
                                # Check if inference was successful
                                if [ $? -eq 0 ]; then
                                    log "Inference completed successfully for: $dataset_name"
                                    successful_count=$((successful_count + 1))
                                else
                                    log "ERROR: Inference failed for: $dataset_name"
                                    failed_count=$((failed_count + 1))
                                fi
                                
                                found_complete_training=true
                                break
                            fi
                        fi
                    done
                fi
                
                if [ "$found_complete_training" = false ]; then
                    log "No complete training found for dataset: $dataset_name"
                fi
            fi
        done
    fi
done

log "Inference run completed"
log "Summary:"
log "  Successful inferences: $successful_count"
log "  Failed inferences: $failed_count"
log "  Total sets processed: $((successful_count + failed_count))"
log "Results saved to: $RESULTS_DIR"