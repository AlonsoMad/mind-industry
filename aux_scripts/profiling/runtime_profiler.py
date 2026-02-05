"""
MIND Profiling - Runtime Profiler

Measures execution time for key MIND operations:
- Segmentation throughput
- Data preparation (NLPipe/spaCy)
- Retrieval operations
- Embedding generation
"""

import gc
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List
import statistics


class RuntimeProfiler:
    """Profiles runtime performance of MIND components."""
    
    def __init__(self, results_dir: Path, warmup_runs: int = 1, benchmark_runs: int = 3):
        self.results_dir = results_dir
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.project_root / "src"))
    
    @contextmanager
    def _timer(self):
        """Context manager for timing code blocks."""
        gc.collect()
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        return elapsed
    
    def _benchmark(self, func, *args, **kwargs) -> Dict:
        """
        Run function multiple times and collect statistics.
        Returns dict with mean, std, min, max, and all times.
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
            gc.collect()
        
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            gc.collect()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                return {"error": str(e), "runtime_seconds": 0}
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return {
            "runtime_seconds": statistics.mean(times),
            "std_seconds": statistics.stdev(times) if len(times) > 1 else 0,
            "min_seconds": min(times),
            "max_seconds": max(times),
            "runs": len(times),
        }
    
    def profile_segmenter(self, corpus_path: Path) -> Dict:
        """
        Profile runtime of Segmenter.segment()
        
        Measures: Documents per second throughput.
        """
        import pandas as pd
        from mind.corpus_building.segmenter import Segmenter
        import tempfile
        
        df = pd.read_parquet(corpus_path)
        num_docs = len(df)
        
        def run_segmentation():
            segmenter = Segmenter()
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                segmenter.segment(
                    path_df=corpus_path,
                    path_save=tmp_path,
                    text_col="text",
                    min_length=50
                )
                result_df = pd.read_parquet(tmp_path)
                return len(result_df)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        
        print("  Profiling: Segmenter runtime")
        result = self._benchmark(run_segmentation)
        result["test"] = "segmenter"
        result["num_input_docs"] = num_docs
        result["docs_per_second"] = num_docs / result["runtime_seconds"] if result["runtime_seconds"] > 0 else 0
        return result
    
    def profile_data_preparer(self, corpus_path: Path) -> Dict:
        """
        Profile runtime of text lemmatization approaches.
        
        Compares: subprocess NLPipe vs in-process spaCy.
        """
        import pandas as pd
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(200, len(df))
        texts = df["text"].head(sample_size).tolist()
        
        results = {}
        
        # Test: spaCy in-process (proposed improvement)
        def lemmatize_spacy():
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                lemmatized = []
                for doc in nlp.pipe(texts, batch_size=50):
                    lemmas = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
                    lemmatized.append(lemmas)
                return len(lemmatized)
            except OSError:
                return -1  # spaCy model not installed
        
        print("  Profiling: spaCy lemmatization (in-process)")
        results["spacy_inprocess"] = self._benchmark(lemmatize_spacy)
        results["spacy_inprocess"]["sample_size"] = sample_size
        
        # Calculate throughput
        rt = results["spacy_inprocess"]["runtime_seconds"]
        if rt > 0:
            results["spacy_inprocess"]["docs_per_second"] = sample_size / rt
        
        return {
            "test": "data_preparer",
            "strategies": results,
            "sample_size": sample_size,
        }
    
    def profile_retriever(self, corpus_path: Path) -> Dict:
        """
        Profile runtime of retrieval operations.
        
        Tests: Single query vs batched query encoding.
        """
        import pandas as pd
        import numpy as np
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(1000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        # Generate test queries
        num_queries = 50
        test_queries = [
            f"What is the main topic about {doc[:50]}?"
            for doc in documents[:num_queries]
        ]
        
        results = {}
        
        # Setup: Create index
        from sentence_transformers import SentenceTransformer
        import faiss
        
        print("    Setting up FAISS index...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        doc_embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
        doc_embeddings = doc_embeddings.astype(np.float32)
        faiss.normalize_L2(doc_embeddings)
        
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings)
        
        # Test: Single query at a time (current approach)
        def search_single_queries():
            all_results = []
            for query in test_queries:
                q_emb = model.encode([query], show_progress_bar=False)
                q_emb = q_emb.astype(np.float32)
                faiss.normalize_L2(q_emb)
                D, I = index.search(q_emb, 10)
                all_results.append(I[0].tolist())
            return len(all_results)
        
        print("  Profiling: Retriever (single query)")
        results["single_query"] = self._benchmark(search_single_queries)
        results["single_query"]["num_queries"] = num_queries
        results["single_query"]["queries_per_second"] = (
            num_queries / results["single_query"]["runtime_seconds"]
            if results["single_query"]["runtime_seconds"] > 0 else 0
        )
        
        gc.collect()
        
        # Test: Batched query encoding (proposed improvement)
        def search_batched_queries():
            q_emb = model.encode(test_queries, batch_size=32, show_progress_bar=False)
            q_emb = q_emb.astype(np.float32)
            faiss.normalize_L2(q_emb)
            D, I = index.search(q_emb, 10)
            return I.shape[0]
        
        print("  Profiling: Retriever (batched query)")
        results["batched_query"] = self._benchmark(search_batched_queries)
        results["batched_query"]["num_queries"] = num_queries
        results["batched_query"]["queries_per_second"] = (
            num_queries / results["batched_query"]["runtime_seconds"]
            if results["batched_query"]["runtime_seconds"] > 0 else 0
        )
        
        return {
            "test": "retriever",
            "num_documents": sample_size,
            "num_queries": num_queries,
            "strategies": results,
        }
    
    def profile_embedding(self, corpus_path: Path) -> Dict:
        """
        Profile embedding generation with different batch sizes.
        
        Identify optimal batch size for the embedding model.
        """
        import pandas as pd
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(2000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        batch_sizes = [1, 8, 16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            def encode_documents():
                return model.encode(
                    documents,
                    batch_size=batch_size,
                    show_progress_bar=False
                ).shape
            
            print(f"  Profiling: Embedding (batch_size={batch_size})")
            result = self._benchmark(encode_documents)
            result["batch_size"] = batch_size
            result["docs_per_second"] = (
                sample_size / result["runtime_seconds"]
                if result["runtime_seconds"] > 0 else 0
            )
            results[f"batch_{batch_size}"] = result
            gc.collect()
        
        # Find optimal batch size
        optimal = max(results.items(), key=lambda x: x[1].get("docs_per_second", 0))
        
        return {
            "test": "embedding_batch_sizes",
            "num_documents": sample_size,
            "batch_results": results,
            "optimal_batch_size": optimal[1].get("batch_size"),
            "optimal_throughput": optimal[1].get("docs_per_second"),
        }
    
    def profile_question_filtering(self) -> Dict:
        """
        Profile the question filtering regex operations.
        
        The _filter_bad_questions method uses complex regex patterns.
        """
        import re
        
        # Sample questions to filter
        sample_questions = [
            "Is this a valid yes/no question?",
            "What is the capital of France?",
            "Does the study include participants from Europe?",
            "Are there any side effects mentioned in the document?",
            "And what about the results?",
            "Has the report been published?",
            "According to the study, is the treatment effective?",
        ] * 100  # Repeat for more stable timing
        
        # Current regex patterns from pipeline.py
        _aux = (
            "is", "are", "am", "was", "were",
            "do", "does", "did",
            "has", "have", "had",
            "can", "could",
            "will", "would",
            "shall", "should",
            "may", "might",
            "must"
        )
        _aux_re = re.compile(rf"^\W*(?:{'|'.join(_aux)})\b", re.IGNORECASE)
        
        _DOC = r"(?:study|survey|report|document)"
        _study_like_re = re.compile(rf"\b{_DOC}\b", re.IGNORECASE)
        
        def filter_questions():
            kept = []
            for q in sample_questions:
                if _aux_re.match(q) and not _study_like_re.search(q):
                    kept.append(q)
            return len(kept)
        
        print("  Profiling: Question filtering regex")
        result = self._benchmark(filter_questions)
        result["test"] = "question_filtering"
        result["num_questions"] = len(sample_questions)
        result["questions_per_second"] = (
            len(sample_questions) / result["runtime_seconds"]
            if result["runtime_seconds"] > 0 else 0
        )
        
        return result
