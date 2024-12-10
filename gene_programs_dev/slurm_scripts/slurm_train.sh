#!/bin/bash
#SBATCH --job-name=QA_dataset_on_msigdb
#SBATCH --output /home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/train_model/slurm_%j.log
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.zhang.ddz5@yale.edu                   # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256gb                                 # Job memory request
#SBATCH --time=1-00:00:00                          # Time limit hrs:min:sec
date;hostname

# module load miniconda
source /home/ddz5/miniconda3/bin/activate     /home/ddz5/.conda/envs/cgsea 

nvidia-smi

DATASET_NAME='local(23)'

python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_training.py \
    --input_path /home/ddz5/scratch/Cell2GSEA_QA_dataset_models/${DATASET_NAME}/training_inputs.pickle \
    --output_prefix /home/ddz5/scratch/Cell2GSEA_QA_dataset_models/ \
    --dataset_name $DATASET_NAME \