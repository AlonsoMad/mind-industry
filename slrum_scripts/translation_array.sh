#!/bin/bash

#SBATCH --job-name=rosie_translation
#SBATCH --output=/fs/clip-scratch/lcalvo/rosie/out/rosie_translation_%A_%a.out
#SBATCH --error=/fs/clip-scratch/lcalvo/rosie/error/rosie_translation_%A_%a.error
#SBATCH --time=03:00:00
#SBATCH --array=75,81
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:3
#SBATCH --mail-user=lcalvo@pa.uc3m.es
#SBATCH --qos=scavenger

# Load environment
module add Python3/3.9.12
source /fs/nexus-scratch/lcalvo/rosie/.venv/bin/activate   
python3 --version
which python3

# Assign SLURM_ARRAY_TASK_ID to the partition variable
PARTITION=$SLURM_ARRAY_TASK_ID

# Set the default values for other variables
SOURCE_FILE="/fs/nexus-scratch/lcalvo/rosie/data/en/corpus_strict_v3.0_en_compiled_passages_lang.parquet"
LANG="en"
TARGET="es"
BATCH_SIZE=1024
TRANSLATION_COLUMN="passage"

# Run the Python script with the provided partition and other default arguments
srun python ../src/corpus_building/translation/translate.py "$SOURCE_FILE" "$LANG" "$TARGET" "$BATCH_SIZE" "$TRANSLATION_COLUMN" "$PARTITION"