#!/usr/bin/env bash
set -euo pipefail

PATH_SOURCE="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet"
METHOD_EVAL="results_3_weighted"
SETTING="thr__dynamic.parquet"
CONFIG_PATH="config/config.yaml"

TOPICS=(11 15)

for TOPIC in "${TOPICS[@]}"; do
  PATH_RELEVANT="/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/retrieval/v2/BAAI/bge-m3/topic_${TOPIC}"
  PATH_SAVE="/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/qa/v2/outs_good_model_tpc${TOPIC}/answers"

  mkdir -p "$PATH_SAVE"

  CMD="python3 /export/usuarios_ml4ds/lbartolome/Repos/umd/mind/ablation/qa/generate_answers.py \
    --topic $TOPIC \
    --method_eval $METHOD_EVAL \
    --top_k 5 \
    --path_source $PATH_SOURCE \
    --path_relevant $PATH_RELEVANT \
    --setting $SETTING \
    --path_save $PATH_SAVE \
    --config_path $CONFIG_PATH"

  echo "$CMD"
  eval "$CMD"
done