import pathlib
import pandas as pd
from scipy import sparse
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

import pathlib
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
import numpy as np
import pickle
from itertools import islice
import gzip
import sys
import os
import pathlib
import numpy as np
import json
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.ndimage import uniform_filter1d
from scipy import sparse
from sentence_transformers import SentenceTransformer
import torch

# Load the SentenceTransformer model
MDOEL = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    #"BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})


def get_doc_top_tpcs(doc_distr, topn=2):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    top_weight = [(k, doc_distr[k]) for k in top]
    return top_weight

def get_doc_main_topc(doc_distr):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:1][0]
    return top

def thrFig(
        thetas32,
        topics=None,
        max_docs=1000,
        poly_degree=3,
        smoothing_window=5,
        do_knee=True,
        n_steps=1000,
        figsize=(10, 6),
        fontsize=12,
        output_fpath=None,
    ):
    significant_docs = {}
    all_elbows = []
    
    # use colorbrewer Set2 colors
    colors = plt.cm.Dark2(np.linspace(0, 1, thetas32.shape[1]))
    n_docs = thetas32.shape[0]
    print(max_docs)
    max_docs = n_docs
    plt.figure(figsize=figsize)

    lines = []
    for k in range(len(thetas32.T)):
        theta_k = np.sort(thetas32[:, k])
        theta_over_th = theta_k[-max_docs:]
        step = max(1, int(np.round(len(theta_over_th) / n_steps)))
        y_values = theta_over_th[::step]
        x_values = np.arange(n_docs-max_docs, n_docs)[::step]

        # Apply smoothing
        x_values_smooth = uniform_filter1d(x_values, size=smoothing_window)

        label = None
        if topics is not None:
            label = topics[k]
        line, = plt.plot(x_values_smooth, y_values, color=colors[k], label=label)
        lines.append(line)
        
        if do_knee:
            # Using KneeLocator to find the elbow point
            allvalues = np.sort(thetas32[:, k].flatten())
            step = int(np.round(len(allvalues) / 1000))
            theta_values = allvalues[::step]
            idx_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]
            
            # Apply smoothing
            idx_values_smooth = uniform_filter1d(idx_values, size=smoothing_window)

            kneedle = KneeLocator(theta_values, idx_values_smooth, curve='concave', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
            elbow = kneedle.elbow
            if elbow is not None:
                all_elbows.append(elbow)

                # Filter document indices based on the elbow point (keeping values above the elbow)
                significant_docs[k] = np.where(thetas32[:, k] >= elbow)[0]

        if elbow:
            # plot elbow in same color, smaller linewidth
            plt.plot([n_docs - max_docs, n_docs], [elbow, elbow], color=colors[k], linestyle='--', linewidth=1)

    # add legend where this series is named with the kth topic, do not assign to the 
    # elbow line
    if topics is not None:
        plt.legend(handles=lines, loc='upper left', fontsize=fontsize-1)

    # Add axis labels
    plt.xlabel('Document Index', fontsize=fontsize)
    plt.ylabel('Theta — P(k | d)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if output_fpath:
        # make bounding box extremely tight
        plt.savefig(output_fpath, bbox_inches='tight', pad_inches=0)

    plt.show()

    return significant_docs, all_elbows

# Configuration
#LLM_MODEL_EMBEDDINGS = 'mxbai-embed-large'
BATCH_SIZE = 8
#EMBEDDING_URL = "http://kumo01.tsc.uc3m.es:11434/api/embeddings"

# Initialize embedding function
#embedding_function = OllamaEmbeddingFunction(
#    url=EMBEDDING_URL,
##    model_name=LLM_MODEL_EMBEDDINGS,
#)

def get_doc_top_topics(doc_distr, topn=2):
    """Extract top topics and their weights."""
    sorted_indices = np.argsort(doc_distr)[::-1]
    return [(idx, doc_distr[idx]) for idx in sorted_indices[:topn]]

def get_doc_main_topic(doc_distr):
    """Extract the main topic index."""
    return np.argmax(doc_distr)

def process_batch(df_batch, collection, model):
    """Process a batch of rows and add to ChromaDB collection."""
    try:
        # Extract metadata
        metadata = df_batch.apply(
            lambda row: row[["id_preproc", "document_id", "common_id", "full_doc"]].to_dict(),
            axis=1
        ).tolist()
        ids = df_batch["id_top"].astype(str).tolist()
        texts = df_batch["text"].tolist()

        # Compute embeddings using the SentenceTransformer model
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

        # Add data to the ChromaDB collection
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)
    except Exception as e:
        print(f"Error processing batch: {e}")

def create_index(df, collection_name, model):
    """Create an index for a specific collection."""
    client = chromadb.PersistentClient(path="indices_multi")
    collection = client.create_collection(name=collection_name)

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        print(f"Processing rows {start}:{end} / {len(df)}")
        process_batch(df.iloc[start:end], collection, model)

def main():
    # Paths
    PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/29_dec/all/df_1.parquet")
    PATH_MODEL = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/all/poly_rosie_1_10")

    print("-- Loading data...")
    raw = pd.read_parquet(PATH_SOURCE)
    thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_EN.npz").toarray()
    S3 = sparse.load_npz(PATH_MODEL / "mallet_output" / "s3_EN.npz").toarray()

    significant_docs, elbows = thrFig(
        S3,#[:, topics],
        #topics=kept_topics,
        max_docs=1000,
        do_knee=True,
        n_steps=5_000,
        figsize=(3.1, 2.5),
        fontsize=7,
        #output_fpath="../figures/thetas_mallet.pdf",
    )

    raw_en = raw[raw.doc_id.str.contains("EN")].copy()  
    raw_en["thetas"] = list(thetas)
    raw_en["s3"] = list(S3)
    raw_en["top_k"] = raw_en["thetas"].apply(get_doc_top_tpcs)
    raw_en["main_topic"] = raw_en["thetas"].apply(get_doc_main_topc)
    raw_en["s3_main_topic"] = raw_en.apply(lambda x: x["s3"][x["main_topic"]], axis=1)
    raw_en["s3_keep_doc"] = raw_en.apply(lambda x: x["s3_main_topic"] > 0.1* elbows[x["main_topic"]], axis=1)
    
    # same for ES
    thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_ES.npz").toarray()
    S3 = sparse.load_npz(PATH_MODEL / "mallet_output" / "s3_ES.npz").toarray()
    significant_docs, elbows = thrFig(
        S3,#[:, topics],
        #topics=kept_topics,
        max_docs=1000,
        do_knee=True,
        n_steps=5_000,
        figsize=(3.1, 2.5),
        fontsize=7,
        #output_fpath="../figures/thetas_mallet.pdf",
    )
    raw_es = raw[raw.doc_id.str.contains("ES")].copy()
    raw_es["thetas"] = list(thetas)
    raw_es["s3"] = list(S3)
    raw_es["top_k"] = raw_es["thetas"].apply(get_doc_top_tpcs)
    raw_es["main_topic"] = raw_es["thetas"].apply(get_doc_main_topc)
    raw_es["s3_main_topic"] = raw_es.apply(lambda x: x["s3"][x["main_topic"]], axis=1)
    raw_es["s3_keep_doc"] = raw_es.apply(lambda x: x["s3_main_topic"] > 0.1* elbows[x["main_topic"]], axis=1)
    
    df_en = raw_en[raw_en.s3_keep_doc].copy()
    df_es = raw_es[raw_es.s3_keep_doc].copy()
    
    print(f"-- -- Keeping {len(df_en) / len(raw_en)} English documents")
    print(f"-- -- Keeping {len(df_es) / len(raw_es)} Spanish documents")
    
    # save df_en
    df_en.to_parquet("df_en.parquet")
    
    with open(PATH_MODEL / "mallet_output" / "keys_ES.txt", 'r') as file:
        lines = file.readlines()
        topic_keys = [line.strip() for line in lines]
        
    # Create index for all Spanish documents
    print(f"-- Creating index for all Spanish documents...")
    create_index(df_es, "docs_all_es", MDOEL)

    # Create indices for each topic
    print(f"-- Creating indices for each topic...")
    for tpc_idx, topic_key in enumerate(topic_keys[:1]):
        print(f"-- Processing topic {tpc_idx}: {topic_key} --")
        df_es_tpc = df_es[df_es.main_topic == tpc_idx]
        create_index(df_es_tpc, f"docs_{tpc_idx}_es", MDOEL)

if __name__ == "__main__":
    main()