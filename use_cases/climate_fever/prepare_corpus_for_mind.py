import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.corpus_building.rosie_corpus import RosieCorpus
from src.topic_modeling.lda_tm import LDATM


def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

df_evidence = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/evidence_candidates.json")


# path_df_chunks = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/evidence_candidates"
# df_evidence.to_parquet(path_df_chunks)
# path_save = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/preprocessed_claims"
# sample=1
# rosie_corpus = RosieCorpus(
#     paths_data={"EN": path_df_chunks}, 
#     multilingual=False
# )
# path_save = rosie_corpus.generate_tm_tr_corpus(path_save, level="none", sample=sample, column_preproc="claim", column_id="claim_id")

preprocessed_claims = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/preprocessed_claims_1.parquet")

merged_df = pd.merge(
    df_evidence,
    preprocessed_claims[['doc_id', 'lemmas']],
    left_on='claim_id',
    right_on='doc_id',
    how='left'
)
merged_df = merged_df.drop(columns='doc_id')

nr_items = np.inf #1000
df_correspondence = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/correspondence_claim_article.json", lines=True)
df_fever_original = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/climate-fever-dataset-r1.jsonl", lines=True)
#df_transformed = pd.read_json("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/questions_transformed.json", lines=True)
df_transformed = pd.read_json(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/questions_transformed_{nr_items}.json", lines=True)
#df_transformed['claim_group_index'] = df_transformed.groupby('claim_id').cumcount()
#df_transformed['claim_evidence_id'] = df_transformed['claim_id'].astype(str) + '-' + df_transformed['claim_group_index'].astype(str)
# save df_transformed back to json
df_transformed = df_transformed.drop(columns=['claim_group_index'])
#df_transformed.to_json(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/questions_transformed_{nr_items}.json", orient="records", lines=True)

final_df = pd.merge(
    df_transformed,
    merged_df[['claim_id', 'top_sentences', 'lemmas']],
    on='claim_id',
    how='left'
)

final_df = final_df.rename(columns={
    'question': 'questions',
    'answer': 'answers'
})


final_df = final_df.drop(columns=["top_sentences"])

models_dir = Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/models")
folders = [f for f in os.listdir(models_dir) if os.path.isdir(models_dir / f)]

docs = final_df["lemmas"].tolist()

for folder in folders:

    num_topics = int(folder.split("_")[-1]) 


    model_path = models_dir / folder

    print(f"Loading model from {model_path}...")

    model = LDATM.load_model(
        model_folder=model_path,
        langs=["EN"],
        num_topics=num_topics,
        load_existing=True,
        mallet_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/topic_modeling/Mallet-202108/bin/mallet",
    )

    print(f"Inferring model with {num_topics} topics...")
    thetas = model.infer(docs, lang="EN")

    final_df[f"theta_{num_topics}_top_tpcs"] = list(map(get_doc_top_tpcs, thetas))
# remove "top_sentences" column
final_df.to_parquet(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/climate_fever/final_fever_for_mind_{len(df_transformed)}.parquet")

import pdb; pdb.set_trace()