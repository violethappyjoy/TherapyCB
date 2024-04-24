#! /bin/bash  

#SBATCH --job-name=test  

#SBATCH --output=%preprocess.out 

#SBATCH --error=%preprocess.err

#SBATCH --nodes=1  

#SBATCH --ntasks-per-node=1  

#SBATCH --gres=gpu:1

#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

module load anaconda3

# cd Work/test

python preprocess.py