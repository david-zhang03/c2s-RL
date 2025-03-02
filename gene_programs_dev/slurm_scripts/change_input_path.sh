#!/bin/bash
#SBATCH --job-name=change_data_input_path
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

/gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/change_input_path.py \
                                                    --new_path "/home/sr2464/scratch/C2S_Files/Cell2Sentence_Datasets/hca_cellxgene_cleaned_h5ad"