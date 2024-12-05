#!/bin/bash
#SBATCH -J 'transformer_training_1'
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
#SBATCH -o outLog_1 
#SBATCH -e errLog_1 
#SBATCH -t 48:00:00 
#SBATCH --mail-user=j_teng@ucsb.edu 
#SBATCH --mail-type ALL

module purge all
module load cuda/12.1 

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate transformer

cd $SLURM_SUBMIT_DIR
python train.py
