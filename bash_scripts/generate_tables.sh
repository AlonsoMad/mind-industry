#!/usr/bin/env bash
set -euo pipefail

PATH_OUT=data/ablations/retrieval/v2/BAAI/bge-m3

MODELS=("gpt-4o-2024-08-06" "llama3.3:70b" "qwen:32b")
TOPICS=(11)

for TOPIC in "${TOPICS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Generating table for topic ${TOPIC} and model ${MODEL}"
    python3 ablation/retrieval/generate_table_eval.py \
      --model_eval "${MODEL}" \
      --path_gold_relevant "${PATH_OUT}/topic_${TOPIC}/questions_topic_${TOPIC}_${MODEL}_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet" \
      --paths_found_relevant "${PATH_OUT}/topic_${TOPIC}/relevant_${MODEL}.parquet" \
    echo "Done for topic ${TOPIC} and model ${MODEL}"
  done
done