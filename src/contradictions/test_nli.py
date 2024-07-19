import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune
import os
from dotenv import load_dotenv
import pathlib
import re
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import dsp
import numpy as np
from scipy import sparse
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import faiss
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def create_faiss_index(df, text_column, id_column, model_name="all-mpnet-base-v2", index_file="faiss_index.index"):
    """
    Create a FAISS index from a DataFrame containing text data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    text_column (str): The name of the column containing text data.
    id_column (str): The name of the column containing unique identifiers for the texts.
    model_name (str): The name of the SentenceTransformer model to use for embeddings.
    index_file (str): The file path to save the FAISS index.

    Returns:
    index: The FAISS index object.
    model: The SentenceTransformer model used for embeddings.
    ids: List of document IDs.
    texts: List of document texts.
    """
    texts = df[text_column].tolist()
    ids = df[id_column].tolist()

    model = SentenceTransformer(model_name, device="cuda")

    # Calculate embeddings for the texts
    embeddings = model.encode(texts, show_progress_bar=False)

    # Create a FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  

    # Normalize embeddings to unit length and add to index
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save the index to a file
    faiss.write_index(index, index_file)

    return index, model, ids, texts

def retrieve_similar_documents(query_text, model, index, ids, texts, k=5):
    """
    Retrieve the k most similar documents to the query text.

    Parameters:
    query_text (str): The query text.
    model: The SentenceTransformer model used for embeddings.
    index: The FAISS index object.
    ids (list): List of document IDs.
    texts (list): List of document texts.
    k (int): The number of nearest neighbors to retrieve.

    Returns:
    list: A list of dictionaries containing document IDs, distances, and texts of the k most similar documents.
    """
    # Encode the query text
    query_embedding = model.encode([query_text], show_progress_bar=False)
    faiss.normalize_L2(query_embedding)
    
    # Search the index for the k nearest neighbors
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the corresponding texts and ids
    results = []
    for i in range(k):
        result = {
            "document_id": ids[indices[0][i]],
            "distance": distances[0][i],
            "text": texts[indices[0][i]]
        }
        results.append(result)
    
    return results

############
# DATA #####
############
path_orig_en = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_en_tr.parquet")
path_orig_es = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/corpus_pass_es_tr.parquet")
path_source = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/translated_stops_filtered_by_al/df_1.parquet")

path_model = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/LDA_FILTERED_AL/rosie_1_20")
path_corpus_en = path_model / "train_data" / "corpus_EN.txt"
path_corpus_es = path_model / "train_data" / "corpus_ES.txt"

persist_directory = (path_model / 'db_contr_mono').as_posix()

raw = pd.read_parquet(path_source)
with path_corpus_en.open("r", encoding="utf-8") as f:
    lines = [line for line in f.readlines()]
corpus_en = [line.rsplit(" 0 ")[1].strip().split() for line in lines]

ids = [line.split(" 0 ")[0] for line in lines]
df_en = pd.DataFrame({"lemmas": [" ".join(doc) for doc in corpus_en]})
df_en["doc_id"] = ids
df_en["len"] = df_en['lemmas'].apply(lambda x: len(x.split()))
df_en["id_top"] = range(len(df_en))
df_en_raw = df_en.merge(raw, how="inner", on="doc_id")[["doc_id", "id_top", "id_preproc", "lemmas_x", "text", "len"]]

# Read thetas 
thetas = sparse.load_npz(path_model.joinpath(f"mallet_output/{'EN'}/thetas.npz")).toarray()
betas = np.load((path_model.joinpath(f"mallet_output/{'EN'}/betas.npy")))
def get_thetas_str(row,thetas):
    return " ".join([f"{id_}|{round(el, 4)}" for id_,el in enumerate(thetas[row]) if el!=0.0])

def get_most_repr_tpc(row,thetas):
    return np.argmax(thetas[row])

# Save thetas in dataframe and "assigned topic"
df_en_raw["thetas"] = df_en_raw.apply(lambda row: get_thetas_str(row['id_top'], thetas), axis=1)
df_en_raw["id_tpc"] = df_en_raw.apply(lambda row: get_most_repr_tpc(row['id_top'], thetas), axis=1)
tpc = 1
df_tpc = df_en_raw[df_en_raw.id_tpc == tpc]

print(f"-- -- Generating index...")
index_en, model_en, ids_en, texts_en = create_faiss_index(df_tpc, text_column='text', id_column='doc_id', index_file='faiss_index_en.index')

import pdb; pdb.set_trace()