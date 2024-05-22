#!/bin/bash

#SBATCH --job-name=rosie_translation
#SBATCH --time=01:30:00
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtx6000ada:3
#SBATCH --mail-user=lcalvo@pa.uc3m.es
#SBATCH --qos=scavenger

# Load environment
module add Python3/3.9.12
source /fs/nexus-scratch/lcalvo/rosie/.venv/bin/activate
python3 --version
which python3

# Set the default values for other variables
SOURCE_FILE="/fs/nexus-scratch/lcalvo/rosie/data/en/corpus_strict_v3.0_en_compiled_passages_lang.parquet"
LANG="en"
TARGET="es"
TRANSLATION_COLUMN="passage"
BATCH_SIZE=1024

# Function to submit a job array
submit_array() {
    OFFSET=$1
    echo "Submitting job array with partitions from $((OFFSET + 1)) to $((OFFSET + 500))"
    sbatch --job-name=rosie_translation \
           --output=/fs/clip-scratch/lcalvo/rosie/out/rosie_translation_${OFFSET}_%A_%a.out \
           --error=/fs/clip-scratch/lcalvo/rosie/error/rosie_translation_${OFFSET}_%A_%a.error \
           --array=1-500 \
           --wrap="OFFSET=$OFFSET; echo Running partition \$((SLURM_ARRAY_TASK_ID + OFFSET)); srun python src/corpus_building/translation/translate.py $SOURCE_FILE $LANG $TARGET $BATCH_SIZE $TRANSLATION_COLUMN \$((SLURM_ARRAY_TASK_ID + OFFSET))"
}

# Submit multiple job arrays to cover the range 12 to 1441
submit_array 11
submit_array 511
submit_array 1011