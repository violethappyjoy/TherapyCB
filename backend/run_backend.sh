#! /bin/bash  

#SBATCH --job-name=chatbot_backend 

#SBATCH --output=%backend.out 

#SBATCH --error=%backend.err

#SBATCH --nodes=1  

#SBATCH --ntasks-per-node=10  

#SBATCH --gres=gpu:1

#SBATCH --time=2-00:00:00

cd $SLURM_SUBMIT_DIR

source ~/.bashrc

module load anaconda3

conda activate /home/21bce026/Work/venv

# python --version
# cd Work/test

# python backend.py

jupyter-lab --no-browser --ip $(hostname -f)
