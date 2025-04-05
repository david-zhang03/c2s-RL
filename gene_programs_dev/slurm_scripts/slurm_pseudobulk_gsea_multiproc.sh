#!/bin/bash
#SBATCH --job-name=QA_dataset_msigdb
#SBATCH --output /home/ddz5/Desktop/c2s-RL/gene_programs_dev/logs/QA_dataset_msigdb/pseudobulk_gsea/slurm_pseudobulk_gsea_%j.log
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

# THIS SCRIPT IS TO MULTIPROCESS PSEUDO-BULK RNA-seq GSEA 

/gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/multiprocess_gsea_datasets.py
