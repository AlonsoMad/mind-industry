#!/usr/bin/env bash
set -euo pipefail

PATH_OUT=data/ablations/retrieval/v2/BAAI/bge-m3
MODELS=("gpt-4o-2024-08-06" "llama3.3:70b" "qwen:32b") #"llama3.3:70b" "qwen:32b"
TOPICS=(11)

for TOPIC in "${TOPICS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Generating table for topic ${TOPIC} and model ${MODEL}"
    python3 ablation/retrieval/generate_table_eval.py \
      --model_eval "${MODEL}" \
      --path_gold_relevant "${PATH_OUT}/topic_${TOPIC}/relevant_${MODEL}.parquet" \
      --paths_found_relevant "${PATH_OUT}/topic_${TOPIC}"
    echo "Done for topic ${TOPIC} and model ${MODEL}"

  done
done