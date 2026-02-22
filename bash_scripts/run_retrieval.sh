#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="BAAI/bge-m3"
PATH_SOURCE="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet"
PATH_MODEL_DIR="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/28_jan/poly_rosie_1_30"

PATH_SAVE_INDICES="/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/retrieval/v3/${MODEL_NAME}"
PATH_OUT="/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/ablations/retrieval/v3/${MODEL_NAME}"

mkdir -p "$PATH_SAVE_INDICES" "$PATH_OUT"

TOPICS=(15)

for TOPIC in "${TOPICS[@]}"; do
  mkdir -p "$PATH_SAVE_INDICES/topic_${TOPIC}" "$PATH_OUT/topic_${TOPIC}"
done

for TOPIC in "${TOPICS[@]}"; do
  echo "Running for topic ${TOPIC}"
  PATH_QUERIES_DIR="/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/data/mind_runs/rosie/v1/outs_good_model_tpc${TOPIC}/questions_queries"

  echo "Getting relevant passages for topic ${TOPIC}"
  echo "runnin python3 /export/usuarios_ml4ds/lbartolome/Repos/umd/mind/ablation/retrieval/get_relevant_passages.py --model_name $MODEL_NAME --path_source $PATH_SOURCE --path_queries_dir $PATH_QUERIES_DIR --path_model_dir $PATH_MODEL_DIR --path_save_indices $PATH_SAVE_INDICES/topic_${TOPIC} --out_dir $PATH_OUT/topic_${TOPIC}"
  python3 "/export/usuarios_ml4ds/lbartolome/Repos/umd/mind/ablation/retrieval/get_relevant_passages.py" \
    --model_name "$MODEL_NAME" \
    --path_source "$PATH_SOURCE" \
    --path_queries_dir "$PATH_QUERIES_DIR" \
    --path_model_dir "$PATH_MODEL_DIR" \
    --path_save_indices "$PATH_SAVE_INDICES/topic_${TOPIC}" \
    --out_dir "$PATH_OUT/topic_${TOPIC}"

  echo "Done for topic ${TOPIC}"
done
