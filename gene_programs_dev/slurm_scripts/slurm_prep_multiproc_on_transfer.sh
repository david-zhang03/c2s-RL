#!/bin/bash
# script to run on local node without slurm
# for use when running script directly on transfer partition
date;hostname
/gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/multiprocess_datasets.py

