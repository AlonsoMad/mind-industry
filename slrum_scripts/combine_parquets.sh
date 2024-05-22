#!/bin/bash

#SBATCH --job-name=rosie_translation
#SBATCH --output=/fs/clip-scratch/lcalvo/rosie/out/rosie_translation_merge.out
#SBATCH --error=/fs/clip-scratch/lcalvo/rosie/error/rosie_translation_merge.error
#SBATCH --time=00:15:00
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mail-user=lcalvo@pa.uc3m.es
#SBATCH --qos=scavenger

# Load environment
module add Python3/3.9.12
source /fs/nexus-scratch/lcalvo/rosie/.venv/bin/activate   
python3 --version
which python3

# Run the Python script with the provided partition and other default arguments
srun python combine_parquets.py