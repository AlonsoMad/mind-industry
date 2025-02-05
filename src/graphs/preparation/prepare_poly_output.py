import pathlib
import pandas as pd
import numpy as np
from scipy import sparse
from qa_metrics.prompt_llm import CloseLLM
import os
from dotenv import load_dotenv
import time
from gensim import corpora
import argparse
from kneed import KneeLocator


def load_environment_variables():
    path_env = pathlib.Path(os.getcwd()).parent.parent.parent / '.env'
    print(path_env)
    load_dotenv(path_env)
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key

def initialize_gpt_model(api_key):
    gpt_model = CloseLLM()
    gpt_model.set_openai_api_key(api_key)
    return gpt_model

def load_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    return file_contents

def get_doc_top_tpcs(doc_distr, topn=2):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    top_weight = [(k, doc_distr[k]) for k in top]
    return top_weight

def get_doc_main_topc(doc_distr):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:1][0]
    return top

def get_most_representative_per_tpc(mat, topn=10, thr=None):
    # Find the most representative document for each topic based on a matrix mat
    top_docs_per_topic = []
    
    mat_ = mat.copy()
    
    if thr is None:
        # Flatten matrix values and sort them
        values = np.sort(mat_.flatten())[::-1]  # Sort descending
        
        # Find the elbow (where diminishing returns occur)
        kneedle = KneeLocator(range(len(values)), values, curve="convex", direction="decreasing")
        thr = values[kneedle.elbow] if kneedle.elbow is not None else np.percentile(values, 90)  # Fallback to 90th percentil
    
    if thr:
        mat_[mat_ > thr] = 0

    print(f"Threshold: {thr}")
    for doc_distr in mat_.T:
        sorted_docs_indices = np.argsort(doc_distr)[::-1]
        top = sorted_docs_indices[:topn].tolist()
        top_docs_per_topic.append(top)
    return top_docs_per_topic

def main(model_path, source_path, orig_en_path, orig_es_path, path_template):
    api_key = load_environment_variables()
    gpt_model = initialize_gpt_model(api_key)
    prompt_template = load_prompt_template(path_template)
    
    # Read raw corpus
    df = pd.read_parquet(source_path)
    df["len"] = df['lemmas'].apply(lambda x: len(x.split()))
    
    # Load thetas
    thetas_en = sparse.load_npz(model_path / "mallet_output" / "thetas_EN.npz").toarray()
    s3_en = sparse.load_npz(model_path / "mallet_output" / "s3_EN.npz").toarray()
    thetas_es = sparse.load_npz(model_path / "mallet_output" / "thetas_ES.npz").toarray()
    s3_es = sparse.load_npz(model_path / "mallet_output" / "s3_ES.npz").toarray()
    
    raw_en = df[df.doc_id.str.contains("EN")].copy()
    raw_en["thetas"] = list(thetas_en)
    raw_en["top_k"] = raw_en["thetas"].apply(get_doc_top_tpcs)
    raw_en["main_topic"] = raw_en["thetas"].apply(get_doc_main_topc)
    
    # get most representative documents per topic based on S3
    top_docs_per_topic_en = get_most_representative_per_tpc(thetas_en, topn=5, thr=0)
    top_docs_per_topic_es = get_most_representative_per_tpc(s3_es, topn=10, thr=0)
    
    raw_es = df[df.doc_id.str.contains("ES")].copy()
    raw_es["thetas"] = list(thetas_es)
    raw_es["top_k"] = raw_es["thetas"].apply(get_doc_top_tpcs)
    raw_es["main_topic"] = raw_es["thetas"].apply(get_doc_main_topc)
   
    # Get topic keys (English)
    with open(model_path / "mallet_output" / "keys_EN.txt", 'r') as file: lines = file.readlines()
    topic_keys = [line.strip() for line in lines]
        
    tpc_labels = []
    for id_tpc, tpc in enumerate(topic_keys):
        this_tpc_prompt = prompt_template.format(keywords=tpc, docs="\n"+"\n".join(raw_en.iloc[top_docs_per_topic_en[id_tpc]].text.values.tolist()))
        print(f"Topic: {tpc}")
        llm_response = gpt_model.prompt_gpt(prompt=this_tpc_prompt, model_engine='gpt-3.5-turbo', temperature=0, max_tokens=10)# 2-5 words
        time.sleep(1)
        #import pdb; pdb.set_trace()
        tpc_labels.append(llm_response)
        print(f"Label: {llm_response}")
    
    # save labels
    with open(model_path / "mallet_output" / "tpc_labels.txt", 'w') as file:
        file.write("\n".join(tpc_labels))
        
    raw_en.loc[:, "label"] = raw_en["main_topic"].apply(lambda x: tpc_labels[x])
    raw_es.loc[:, "label"] = raw_es["main_topic"].apply(lambda x: tpc_labels[x])
    
    # Keep non-zero theta values and convert to string
    def stringfy_thetas(thetas):
        thetas_non = [(i,float(theta)) for i,theta in enumerate(thetas) if float(theta) != 0.0]
        return str(thetas_non)
    raw_en["thetas"] = raw_en["thetas"].apply(stringfy_thetas)
    raw_es["thetas"] = raw_es["thetas"].apply(stringfy_thetas)
    
    raw_en.to_parquet(model_path  / "df_graph_en.parquet")
    raw_es.to_parquet(model_path / "df_graph_es.parquet")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='Path to the Polylingual TM directory',
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30"
    )
    parser.add_argument(
        '--source_path',
        type=str,
        required=False,
        help='Path to the source file with which the TM was trained.',
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet"
    )
    parser.add_argument(
        '--orig_en_path',
        type=str,
        required=False,
        help='Path to the original English corpus.',
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet"
    )
    parser.add_argument(
        '--orig_es_path',
        type=str,
        required=False,
        help='Path to the original Spanish corpus.',
        default="/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet"
    )
    parser.add_argument(
        '--path_template',
        type=str,
        required=False,
        help='Path to the template to use for labelling tool.',
        default="prompt_labels.txt"
    )

    args = parser.parse_args()
    main(
        pathlib.Path(args.model_path),
        pathlib.Path(args.source_path),
        pathlib.Path(args.orig_en_path),
        pathlib.Path(args.orig_es_path),
        args.path_template)
