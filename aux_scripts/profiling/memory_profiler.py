"""
MIND Profiling - Memory Profiler

Measures memory usage for key MIND operations:
- Segmenter document processing
- Corpus loading strategies
- Retriever indexing
- Translator preprocessing
"""

import gc
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, Optional
import json

# Handle optional dependencies gracefully
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")


class MemoryProfiler:
    """Profiles memory usage of MIND components."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.project_root / "src"))
    
    def _get_process_memory_mb(self) -> float:
        """Get current process memory in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _run_with_memory_tracking(self, func, *args, **kwargs) -> Dict:
        """
        Run a function while tracking memory usage.
        Returns dict with peak_memory_mb, allocated_mb, and result.
        """
        gc.collect()
        
        mem_before = self._get_process_memory_mb()
        tracemalloc.start()
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            tracemalloc.stop()
            return {
                "error": str(e),
                "peak_memory_mb": 0,
                "allocated_mb": 0,
            }
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        mem_after = self._get_process_memory_mb()
        
        return {
            "peak_memory_mb": peak / (1024 * 1024),
            "allocated_mb": current / (1024 * 1024),
            "rss_delta_mb": mem_after - mem_before,
            "rss_before_mb": mem_before,
            "rss_after_mb": mem_after,
        }
    
    def profile_segmenter(self, corpus_path: Path) -> Dict:
        """
        Profile memory usage of Segmenter.segment()
        
        This tests the current row-by-row iteration approach.
        Key metric: Peak memory relative to input size.
        """
        from mind.corpus_building.segmenter import Segmenter
        import tempfile
        
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
                return tmp_path
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        
        print("  Profiling: Segmenter.segment()")
        result = self._run_with_memory_tracking(run_segmentation)
        result["test"] = "segmenter"
        result["input_file_mb"] = corpus_path.stat().st_size / (1024 * 1024)
        return result
    
    def profile_corpus_loading(self, corpus_path: Path) -> Dict:
        """
        Profile memory usage of Corpus loading.
        
        Tests: Full DataFrame load vs lazy loading approaches.
        """
        import pandas as pd
        import pyarrow.parquet as pq
        
        results = {}
        
        # Test 1: Full pandas load (current approach)
        def load_full_pandas():
            df = pd.read_parquet(corpus_path)
            return len(df)
        
        print("  Profiling: Corpus loading (full pandas)")
        results["full_pandas"] = self._run_with_memory_tracking(load_full_pandas)
        
        gc.collect()
        
        # Test 2: PyArrow with self_destruct
        def load_pyarrow_destruct():
            table = pq.read_table(corpus_path)
            df = table.to_pandas(self_destruct=True, ignore_metadata=True)
            return len(df)
        
        print("  Profiling: Corpus loading (pyarrow self_destruct)")
        results["pyarrow_destruct"] = self._run_with_memory_tracking(load_pyarrow_destruct)
        
        gc.collect()
        
        # Test 3: Chunked loading (proposed improvement)
        def load_chunked():
            parquet_file = pq.ParquetFile(corpus_path)
            total_rows = 0
            for batch in parquet_file.iter_batches(batch_size=5000):
                df_chunk = batch.to_pandas()
                total_rows += len(df_chunk)
                del df_chunk
            return total_rows
        
        print("  Profiling: Corpus loading (chunked iteration)")
        results["chunked"] = self._run_with_memory_tracking(load_chunked)
        
        return {
            "test": "corpus_loading",
            "strategies": results,
            "input_file_mb": corpus_path.stat().st_size / (1024 * 1024),
        }
    
    def profile_retriever_indexing(self, corpus_path: Path) -> Dict:
        """
        Profile memory usage of IndexRetriever indexing.
        
        Tests embedding generation and FAISS index construction.
        """
        import pandas as pd
        
        # Load a sample of documents for indexing
        df = pd.read_parquet(corpus_path)
        sample_size = min(1000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        results = {}
        
        # Test: Embedding generation memory
        def generate_embeddings():
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
            return embeddings.shape
        
        print("  Profiling: Embedding generation")
        results["embedding_generation"] = self._run_with_memory_tracking(generate_embeddings)
        results["embedding_generation"]["num_documents"] = sample_size
        
        gc.collect()
        
        # Test: FAISS index construction
        def build_faiss_index():
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
            
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            return index.ntotal
        
        print("  Profiling: FAISS index construction")
        results["faiss_construction"] = self._run_with_memory_tracking(build_faiss_index)
        
        return {
            "test": "retriever_indexing",
            "sample_size": sample_size,
            "strategies": results,
        }
    
    def profile_translator(self, corpus_path: Path) -> Dict:
        """
        Profile memory usage of Translator sentence splitting.
        
        The current implementation creates new rows for each sentence,
        which can significantly increase memory usage.
        """
        import pandas as pd
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(500, len(df))
        sample_df = df.head(sample_size).copy()
        
        results = {}
        
        # Test: Row-by-row splitting (current approach)
        def split_row_by_row():
            rows = []
            for _, row in sample_df.iterrows():
                sentences = str(row["text"]).split(". ")
                for i, s in enumerate(sentences):
                    if len(s) > 10:
                        new_row = row.to_dict()
                        new_row["text"] = s
                        rows.append(new_row)
            return pd.DataFrame(rows)
        
        print("  Profiling: Translator split (row-by-row)")
        results["row_by_row"] = self._run_with_memory_tracking(split_row_by_row)
        
        gc.collect()
        
        # Test: Vectorized splitting (proposed improvement)
        def split_vectorized():
            temp_df = sample_df.copy()
            temp_df["sentences"] = temp_df["text"].str.split(r"\.\s+", regex=True)
            exploded = temp_df.explode("sentences")
            filtered = exploded[exploded["sentences"].str.len() > 10]
            return filtered
        
        print("  Profiling: Translator split (vectorized)")
        results["vectorized"] = self._run_with_memory_tracking(split_vectorized)
        
        return {
            "test": "translator_splitting",
            "sample_size": sample_size,
            "strategies": results,
        }
    
    def profile_topic_model_state(self, state_gz_path: Optional[Path] = None) -> Dict:
        """
        Profile memory usage of topic model state file parsing.
        
        This tests the output-state.gz parsing which can be 500MB+.
        """
        if state_gz_path is None or not state_gz_path.exists():
            return {"test": "topic_model_state", "skipped": True, "reason": "No state file provided"}
        
        import gzip
        import pandas as pd
        
        results = {}
        
        # Test: Full DataFrame load (current approach)
        def load_full_dataframe():
            with gzip.open(state_gz_path) as fin:
                df = pd.read_csv(
                    fin, delim_whitespace=True,
                    names=['docid', 'lang', 'wd_docid', 'wd_vocabid', 'wd', 'tpc'],
                    header=None, skiprows=1
                )
            return len(df)
        
        print("  Profiling: Topic state loading (full DataFrame)")
        results["full_dataframe"] = self._run_with_memory_tracking(load_full_dataframe)
        
        gc.collect()
        
        # Test: Streaming parse (proposed improvement)
        def load_streaming():
            counts = {}
            with gzip.open(state_gz_path, 'rt') as fin:
                next(fin)  # Skip header
                for line in fin:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        tpc = int(parts[5])
                        counts[tpc] = counts.get(tpc, 0) + 1
            return sum(counts.values())
        
        print("  Profiling: Topic state loading (streaming)")
        results["streaming"] = self._run_with_memory_tracking(load_streaming)
        
        return {
            "test": "topic_model_state",
            "input_file_mb": state_gz_path.stat().st_size / (1024 * 1024),
            "strategies": results,
        }
