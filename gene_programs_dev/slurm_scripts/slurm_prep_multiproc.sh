#!/bin/bash
#SBATCH --job-name=QA_dataset_msigdb
#SBATCH --output /home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/prep_input/slurm_%j.log
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.zhang.ddz5@yale.edu                   # Where to send mail
#SBATCH --partition bigmem
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=32GB                          # Job memory request
#SBATCH --time=1-00:00:00                          # Time limit hrs:min:sec
date;hostname

# Script to multiprocess training inputs for GNN gene set annotation

# module load miniconda
source /home/ddz5/miniconda3/bin/activate activate    /home/ddz5/.conda/envs/c2gsea 

python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/multiprocess_datasets.py

# python /home/ddz5/work/gene_programs_dev/scripts/prepare_training_input.py \
#    --h5ad_path /home/sz568/scratch/C2S_RL/all_datasets/hca_cellxgene_cleaned_h5ad/local(23)_cleaned.h5ad \
#    --genesets_path /home/ddz5/Desktop/c2s-RL/gene_programs_dev/data/msigdb.tsv \
#    --dataset_name local_23 \
#    --output_prefix /home/ddz5/scratch/Cell2GSEA_QA_dataset_models/ \


# python /home/sg2597/work/gene_programs_dev/scripts/run_training_from_prepared_inputs_val.py \
#     --input_path /vast/palmer/scratch/dijk/sg2597/gene_programs_data/experiments/the_immune_dataset_on_msigdb/training_inputs.pickle \
#     --output_prefix /vast/palmer/scratch/dijk/sg2597/gene_programs_data/experiments/the_immune_dataset_on_msigdb \
#     --n_epochs 1000

