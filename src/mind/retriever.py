import logging
import time
from typing import Union
from sentence_transformers import SentenceTransformer, util # type: ignore
from pathlib import Path
import pandas as pd # type: ignore
from scipy import sparse
import faiss # type: ignore
import numpy as np # type: ignore
from src.mind.utils import get_doc_top_tpcs
from tqdm import tqdm # type: ignore
from kneed import KneeLocator # type: ignore
from scipy.ndimage import uniform_filter1d
from src.utils.utils import init_logger

class IndexRetriever:
    def __init__(
        self, 
        model: SentenceTransformer, 
        top_k: int = 10,
        batch_size: int = 32,
        min_clusters: int = 8,
        do_weighting: bool = True,
        n_clusters_ann: int = 100,
        nprobe: int = 10,
        logger: logging.Logger = None,
        config_path: Path = Path("config/config.yaml")
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
    
        self.model = model
        self.batch_size = batch_size
        self.top_k = top_k
        self.min_clusters = min_clusters
        self.do_weighting = do_weighting
        self.n_clusters_ann = n_clusters_ann
        self.nprobe = nprobe
        
        self.faiss_index = None
        self.doc_ids = None
        self.topic_indices = None  # only for TB-ENN and TB-ANN
        self.index_method = None
        self.save_path = None
        
    def dynamic_thresholds(
        self,
        mat_, 
        poly_degree=3, 
        smoothing_window=5
    ):
        '''
        Computes the threshold dynamically to obtain significant
        topics in the indexing phase.
        '''
        thrs = []
        for k in range(len(mat_.T)):
            allvalues = np.sort(mat_[:, k].flatten())
            step = int(np.round(len(allvalues) / 1000))
            x_values = allvalues[::step]
            x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
            y_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]
            y_values_smooth = uniform_filter1d(y_values, size=smoothing_window)
            kneedle = KneeLocator(x_values, y_values_smooth, curve='concave', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
            thrs.append(kneedle.elbow)
        return thrs
    
    def load_indices(self, method: str, source_path: str, model_path: str, save_path_parent: str):
        self.index_method = method
        self.save_path = (
            Path(save_path_parent)
            / Path(source_path).stem
            / Path(model_path).stem
            / method
        )

        if method in ["ENN", "ANN"]:
            index_path = self.save_path / "faiss_index.index"
            doc_ids_path = self.save_path / "doc_ids.npy"
            if index_path.exists() and doc_ids_path.exists():
                self._logger.info(f"Loading FAISS index and doc_ids for method {method}...")
                self.faiss_index = faiss.read_index(str(index_path))
                if method == "ANN":
                    self.faiss_index.nprobe = self.nprobe
                self.doc_ids = np.load(doc_ids_path, allow_pickle=True)
            else:
                raise FileNotFoundError(f"Index or doc_ids not found for method {method}.")
        
        elif method in ["TB-ENN", "TB-ANN"]:
            self.topic_indices = {}
            for index_file in self.save_path.glob("faiss_index_topic_*.index"):
                topic = int(index_file.stem.split("_")[-1])
                doc_ids_path = self.save_path / f"doc_ids_topic_{topic}.npy"
                if doc_ids_path.exists():
                    index = faiss.read_index(str(index_file))
                    if method == "TB-ANN":
                        index.nprobe = self.nprobe
                    doc_ids = np.load(doc_ids_path, allow_pickle=True)
                    self.topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
            if not self.topic_indices:
                raise FileNotFoundError(f"No topic-based indices found in {self.save_path}.")
    
    def index(
        self,
        source_path:str,
        save_path_parent:str,
        model_path:str=None,
        col_to_index:str="chunk_text",
        col_id:str="chunk_id",
        lang:str=None, # if lang is given, it will be used to filter the dataframe
        thr_assignment: Union[float, str] = "var",
        method:str = "TB-ENN"
    ):  
        save_path = (
            Path(save_path_parent)
            / Path(source_path).stem
            / Path(model_path).stem
            / method
        )
        save_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_parquet(source_path)
        if lang:
            df = df[df[col_id].str.contains(lang)].copy()
        corpus_embeddings = self.model.encode(df[col_to_index].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=self.batch_size)
        
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        embedding_size = corpus_embeddings.shape[1]

        if method in ["ENN", "ANN"]:            
            self._logger.info(f"Creating {method} indices")
            quantizer = faiss.IndexFlatIP(embedding_size)
            
            index = (
                faiss.IndexIVFFlat(quantizer, embedding_size, self.n_clusters_ann, faiss.METRIC_INNER_PRODUCT)
                if method == 'ANN' else quantizer
            )
            
            if method == 'ANN':
                index.train(corpus_embeddings)
                index.nprobe = self.nprobe
        
            index.add(corpus_embeddings)
            
            faiss.write_index(index, (save_path / "faiss_index.index").as_posix())
            # save doc_ids
            np.save((save_path / "doc_ids.npy").as_posix(), df[col_id].values)
            

        elif method in ["TB-ENN", "TB-ANN"]:
        
            thetas = sparse.load_npz(Path(model_path) / "mallet_output" / f"thetas_{lang}.npz").toarray()
            
            df["thetas"] = list(thetas)
            df["top_k"] = df["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=self.top_k))
            
            # determine threshold for topic assignment
            if thr_assignment == "var":
                thrs = self.dynamic_thresholds(thetas)
            else:
                thrs = thr_assignment * np.ones(thetas.shape[1])
            
            topic_indices = {}
            for topic in tqdm(range(thetas.shape[1]), desc="Creating index"):   
                this_tpc_thr = thrs[topic]
                index_path = save_path / f"faiss_index_topic_{topic}.index"
                doc_ids_path = save_path / f"doc_ids_topic_{topic}.npy"

                if index_path.exists() and doc_ids_path.exists():
                    continue

                self._logger.info(f"-- Creating index for topic {topic}...")
                topic_embeddings = []
                doc_ids = []

                for i, top_k in enumerate(df["top_k"]):
                    for t, weight in top_k:
                        if t == topic and weight > this_tpc_thr:
                            topic_embeddings.append(corpus_embeddings[i])
                            doc_ids.append(df.iloc[i][col_id])
                if topic_embeddings:
                    topic_embeddings = np.array(topic_embeddings).astype("float32")
                    
                    N = len(topic_embeddings)
                    n_clusters = max(int(4 * np.sqrt(N)), self.min_clusters)
                    
                    self._logger.info(f"-- TOPIC {topic}: {N} documents, {n_clusters} clusters")
                    
                    # Train IVF index
                    embedding_size = topic_embeddings.shape[1]
                    quantizer = faiss.IndexFlatIP(embedding_size)

                    index = (
                        faiss.IndexIVFFlat(quantizer, embedding_size,n_clusters, faiss.METRIC_INNER_PRODUCT)
                        if method == 'TB_ANN' else quantizer
                    )
                    
                    if method == 'TB_ANN':
                        index.train(topic_embeddings)
                        index.nprobe = self.nprobe
                    
                    # Add the topic embeddings to the index
                    index.add(topic_embeddings)
                    
                    # Save the index and document IDs
                    faiss.write_index(index, str(index_path))
                    np.save(doc_ids_path, np.array(doc_ids))
                    topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
                else:
                    self._logger.info(f"-- No documents found for topic {topic} with threshold {this_tpc_thr}.")
                    topic_indices[topic] = {"index": None, "doc_ids": None}
                    
        self._logger.info(f"Indices created and saved in {save_path}")
        
        self.load_indices(
            method=method,
            source_path=source_path,
            model_path=model_path,
            save_path_parent=save_path_parent
        )
            
        return
    
    def build_or_load_index(
        self,
        source_path: str,
        model_path: str,
        save_path_parent: str,
        method: str = "TB-ENN",
        lang: str = "EN",
        **index_kwargs
    ):
        save_path = (
            Path(save_path_parent)
            / Path(source_path).stem
            / Path(model_path).stem
            / method
        )

        if method in ["ENN", "ANN"]:
            index_path = save_path / "faiss_index.index"
            doc_ids_path = save_path / "doc_ids.npy"
            if index_path.exists() and doc_ids_path.exists():
                self._logger.info(f"Existing index found for {method}. Loading...")
                return self.load_indices(method, source_path, model_path, save_path_parent)

        elif method in ["TB-ENN", "TB-ANN"]:
            any_index = list(save_path.glob("faiss_index_topic_*.index"))
            if any_index:
                self._logger.info(f"Existing topic-based indices found for {method}. Loading...")
                return self.load_indices(method, source_path, model_path, save_path_parent)

        self._logger.info(f"No existing index found for {method}. Building index...")
        self.index(
            source_path=source_path,
            model_path=model_path,
            save_path_parent=save_path_parent,
            method=method,
            lang=lang,
            **index_kwargs
        )

    
    def retrieve_enn_ann(self, query: str, top_k: int = 10):
        if self.faiss_index is None or self.doc_ids is None:
            raise ValueError("FAISS index or doc_ids not loaded. Make sure to call load_indices first.")

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]  # shape: (dim,)
        distances, indices = self.faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)

        return [
            {"doc_id": self.doc_ids[idx], "score": distances[0][i]}
            for i, idx in enumerate(indices[0]) if idx != -1
        ]
        
    def retrieve_topic_faiss(self, query, theta_query, top_k=10, thrs=None, do_weighting=True):
        if self.topic_indices is None:
            raise ValueError("Topic-based indices not loaded.")

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        results = []

        for topic, weight in theta_query:
            thr = thrs[topic] if thrs is not None else 0
            if weight > thr:
                topic_data = self.topic_indices.get(topic)
                if topic_data and topic_data["index"] is not None:
                    index = topic_data["index"]
                    doc_ids = topic_data["doc_ids"]
                    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx != -1:
                            score = dist * weight if do_weighting else dist
                            results.append({"topic": topic, "doc_id": doc_ids[idx], "score": score})

        # Remove duplicates, keeping the highest score
        unique_results = {}
        for result in results:
            doc_id = result["doc_id"]
            if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
                unique_results[doc_id] = result

        return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    def retrieve(self, query: str, theta_query: list = None):
        time_start = time.time()
        if self.index_method in ["ENN", "ANN"]:
            results =  self.retrieve_enn_ann(query, self.top_k)
        elif self.index_method in ["TB-ENN", "TB-ANN"]:
            if theta_query is None:
                raise ValueError("theta_query must be provided for topic-based retrieval.")
            
            results =  self.retrieve_topic_faiss(
                query=query,
                theta_query=theta_query,
                top_k=self.top_k,
                do_weighting=self.do_weighting,
            )
        else:
            raise ValueError(f"Unknown index method: {self.index_method}")
        
        time_end = time.time()

        return results, time_end - time_start

if __name__ == "__main__":
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create IndexRetriever instance
    retriever = IndexRetriever(
        model=model,
        top_k=10,
        batch_size=32,
        min_clusters=8,
    )

    source_path = "data/climate_fever/outs/1000/wikipedia_chunks_1234_1000_partial_1470.parquet"
    model_path = "data/climate_fever/models/1_10"
    save_path_parent = "data/climate_fever/index"
    lang = "EN"

    df = pd.read_parquet(source_path)
    thetas = sparse.load_npz(Path(model_path) / "mallet_output" / lang / "thetas.npz").toarray()
    theta_query = get_doc_top_tpcs(thetas[0], topn=10) 
    import pdb; pdb.set_trace()
    
    query = "The Nakba is the ethnic cleansing of Palestinian Arabs"


    methods = ["ENN", "ANN", "TB-ENN", "TB-ANN"]
    query_text = df.iloc[0]["chunk_text"]

    for method in methods:
        print(f"Testing method: {method}")
        retriever.index(
            source_path=source_path,
            save_path_parent=save_path_parent,
            model_path=model_path,
            method=method,
            lang=lang,
        )

        if method in ["TB-ENN", "TB-ANN"]:
            results, elapsed = retriever.retrieve(query=query_text, theta_query=theta_query, top_k=5)
        else:
            results, elapsed = retriever.retrieve(query=query_text, top_k=5)

        print(f"Time taken: {elapsed:.4f} seconds")
        print(" Top results:")
        for r in results:
            info = f" (topic {r['topic']})" if 'topic' in r else ""
            print(f"- doc_id: {r['doc_id']}, score: {r['score']:.4f}{info}, text: {df[df['chunk_id'] == r['doc_id']]['chunk_text'].values[0][:50]}...")