#!/bin/bash
#SBATCH -J 'transformer_training_1'
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8  
#SBATCH -p gpu 
#SBATCH --gres=gpu:3
#SBATCH -o outLog 
#SBATCH -e errLog 
#SBATCH -t 02:00:00 
#SBATCH --mail-user=j_teng@ucsb.edu 
#SBATCH --mail-type ALL

module purge all
module load cuda/12.1 

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate transformer

cd $SLURM_SUBMIT_DIR
python train.py
