#!/bin/bash
#SBATCH -J 'transformer_training_1'
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G  
#SBATCH -p gpu 
#SBATCH --gres=gpu:4
#SBATCH -o outLog 
#SBATCH -e errLog 
#SBATCH -t 24:00:00 
#SBATCH --mail-user=j_teng@ucsb.edu 
#SBATCH --mail-type ALL

module purge all
module load cuda/12.1 

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate transformer

cd $SLURM_SUBMIT_DIR
python train.py
