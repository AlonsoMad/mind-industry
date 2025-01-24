import pathlib
import pandas as pd
import numpy as np
import faiss
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
import os

# Configuration
model_name = "quora-distilbert-multilingual"
embedding_size = 768  # Dimension of sentence embeddings
min_clusters = 8  # Minimum number of clusters for small topic sizes
top_k_hits = 10  # Number of nearest neighbors to retrieve
BATCH_SIZE = 32
THR_TOPIC_ASSIGNMENT = 0.05

# Load SentenceTransformer model
model = SentenceTransformer(model_name)

# Paths
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/29_dec/all/df_1.parquet")
PATH_MODEL = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/29_dec/all/poly_rosie_1_50")
FAISS_SAVE_DIR = pathlib.Path("faiss_indices_50")  # Directory to save FAISS indices

# Ensure save directory exists
os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

# Load data
print("-- Loading data...")
raw = pd.read_parquet(PATH_SOURCE)
thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_ES.npz").toarray()

# Filter Spanish documents
raw_es = raw[raw.doc_id.str.contains("ES")].copy()
raw_es["thetas"] = list(thetas)

# Compute top topics with weights
def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

raw_es["top_k"] = raw_es["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))

print("-- Checking existing indices...")
topic_indices = {}
all_indices_exist = True 

for topic in range(thetas.shape[1]):
    index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
    doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

    if index_path.exists() and doc_ids_path.exists():
        # Load the FAISS index and document IDs
        print(f"Loading indices for topic {topic}...")
        index = faiss.read_index(str(index_path))
        doc_ids = np.load(doc_ids_path, allow_pickle=True)
        topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
    else:
        # If any index is missing, set flag to False
        print(f"Missing indices for topic {topic}")
        all_indices_exist = False
        break

# We only generate embeddings and create new indices if any of the existing indices are missing
if not all_indices_exist:
    print("-- Generating embeddings...")
    corpus_embeddings = model.encode(
        raw_es["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE
    )
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Create FAISS indices for each topic
    for topic in range(thetas.shape[1]):
        index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
        doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

        if index_path.exists() and doc_ids_path.exists():
            continue

        print(f"-- Creating index for topic {topic}...")
        topic_embeddings = []
        doc_ids = []

        for i, top_k in enumerate(raw_es["top_k"]):
            for t, weight in top_k:
                if t == topic and weight > THR_TOPIC_ASSIGNMENT:  # Relevance threshold for topic assignment
                    topic_embeddings.append(corpus_embeddings[i])
                    doc_ids.append(raw_es.iloc[i].doc_id)

        if topic_embeddings:
            topic_embeddings = np.array(topic_embeddings)
            N = len(topic_embeddings)
            n_clusters = max(int(4 * np.sqrt(N)), min_clusters)

            print(f"-- Tópico {topic}: {N} documentos, {n_clusters} clusters")

            # Train IVF index
            quantizer = faiss.IndexFlatIP(embedding_size)
            index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
            index.train(topic_embeddings)
            index.add(topic_embeddings)

            # Save the index and document IDs
            faiss.write_index(index, str(index_path))
            np.save(doc_ids_path, np.array(doc_ids))
            topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
else:
    print("-- All indices are loaded.")

# Query processing
def query_faiss(question, theta_query, top_k=10):
    """
    Perform a weighted topic search using FAISS indices.
    """
    question_embedding = model.encode([question], normalize_embeddings=True)[0]
    results = []

    for topic, weight in theta_query:
        index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
        doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

        if index_path.exists() and doc_ids_path.exists():
            # Load the FAISS index and document IDs
            index = faiss.read_index(str(index_path))
            doc_ids = np.load(doc_ids_path, allow_pickle=True)

            # Perform the search
            distances, indices = index.search(np.expand_dims(question_embedding, axis=0), top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    results.append({"topic": topic, "doc_id": doc_ids[idx], "score": dist }) #* weight

    # Sort results by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# Query Example: Use an English document to search the Spanish corpus
print("-- Testing query from English corpus...")

# Filter English documents
thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_EN.npz").toarray()
raw_en = raw[raw.doc_id.str.contains("EN")].copy()
raw_en["thetas"] = list(thetas_en)
raw_en["top_k"] = raw_en["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))
raw_en["main_topic"] = raw_en["thetas"].apply(lambda x: np.argmax(x))  # Main topic assignment

# Select an English document from topic 0
doc_from_topic_0 = raw_en[raw_en["main_topic"] == 0].iloc[0]  # First document from topic 0
query_text = doc_from_topic_0["text"]
query_theta = get_doc_top_tpcs(doc_from_topic_0["thetas"], topn=10)

# Perform the search
results = query_faiss(query_text, query_theta, top_k=10)

# Display results
print(f"Query document (English): {query_text}")
print("\nSimilar Spanish documents:")
for result in results:
    doc_id = result["doc_id"]
    score = result["score"]
    similar_doc_text = raw_es[raw_es["doc_id"] == doc_id]["text"].values[0]
    print(f"Doc ID: {doc_id}, Score: {score:.3f}, Text: {similar_doc_text[:200]}...")

# Compute recall: Compare topic-based ANN vs exact search
print("\n-- Evaluating Recall of Topic-Based ANN Search...")

# Exact search over the entire corpus (baseline for recall)
corpus_embeddings_es = model.encode(raw_es["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE)
corpus_embeddings_es = corpus_embeddings_es / np.linalg.norm(corpus_embeddings_es, axis=1, keepdims=True)  # Normalize
# Compute recall: Compare topic-based ANN vs exact search
print("\n-- Evaluating Recall of Topic-Based ANN Search...")

# Exact search over the entire corpus (baseline for recall)
query_embedding = model.encode([query_text], normalize_embeddings=True)  # Encode and normalize the query
exact_hits = util.semantic_search(
    query_embedding,  # Query embedding
    corpus_embeddings_es,  # Entire Spanish corpus embeddings
    top_k=10
)[0]

# Map exact hits corpus indices to document IDs
exact_hit_ids = set()
for hit in exact_hits:
    corpus_index = hit["corpus_id"]  # Index in the Spanish corpus
    if corpus_index < len(raw_es):  # Ensure within bounds
        exact_hit_ids.add(raw_es.iloc[corpus_index]["doc_id"])

# Topic-based ANN results (already retrieved)
ann_hit_ids = set(result["doc_id"] for result in results)

# Compute recall
recall = len(ann_hit_ids.intersection(exact_hit_ids)) / len(exact_hit_ids) if exact_hit_ids else 0
print(f"Recall@10 of Topic-Based ANN vs Exact Search: {recall * 100:.2f}%")

# Print missing results if recall < 1
if recall < 1:
    print("Missing results (present in exact search but not in ANN):")
    for hit in exact_hits:
        corpus_index = hit["corpus_id"]
        if corpus_index < len(raw_es):  # Ensure within bounds
            doc_id = raw_es.iloc[corpus_index]["doc_id"]
            if doc_id not in ann_hit_ids:
                missing_doc_text = raw_es.iloc[corpus_index]["text"]
                missing_doc_main_thetas = raw_es.iloc[corpus_index]["top_k"]
                print(f"\tScore: {hit['score']:.3f}, Doc ID: {doc_id}, Text: {missing_doc_text[:200]}...", f"top_k: {missing_doc_main_thetas}")


import pdb; pdb.set_trace()