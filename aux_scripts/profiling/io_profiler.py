"""
MIND Profiling - I/O Profiler

Measures disk I/O performance for key MIND operations:
- Parquet read performance (full vs chunked)
- Parquet write with different compression
- Checkpoint write patterns
- FAISS index I/O
"""

import gc
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict


class IOProfiler:
    """Profiles disk I/O performance of MIND components."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.project_root / "src"))
    
    def _time_operation(self, func, *args, **kwargs) -> Dict:
        """Time a single operation and return metrics."""
        gc.collect()
        
        # Sync filesystem before timing
        if hasattr(os, 'sync'):
            os.sync()
        
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            return {"error": str(e), "runtime_seconds": 0}
        elapsed = time.perf_counter() - start
        
        return {
            "runtime_seconds": elapsed,
            "result": result if isinstance(result, (int, float, str)) else None,
        }
    
    def profile_parquet_read(self, corpus_path: Path) -> Dict:
        """
        Profile Parquet read performance with different strategies.
        
        Tests: Full read, columns subset, chunked iteration.
        """
        import pandas as pd
        import pyarrow.parquet as pq
        
        file_size_mb = corpus_path.stat().st_size / (1024 * 1024)
        results = {}
        
        # Test 1: Full pandas read
        def read_full_pandas():
            df = pd.read_parquet(corpus_path)
            return len(df)
        
        print("  Profiling: Parquet read (full pandas)")
        result = self._time_operation(read_full_pandas)
        result["method"] = "full_pandas"
        result["throughput_mb_per_sec"] = (
            file_size_mb / result["runtime_seconds"]
            if result["runtime_seconds"] > 0 else 0
        )
        results["full_pandas"] = result
        gc.collect()
        
        # Test 2: Column subset read
        def read_columns_only():
            df = pd.read_parquet(corpus_path, columns=["text", "id_preproc"])
            return len(df)
        
        print("  Profiling: Parquet read (columns subset)")
        result = self._time_operation(read_columns_only)
        result["method"] = "columns_subset"
        result["throughput_mb_per_sec"] = (
            file_size_mb / result["runtime_seconds"]
            if result["runtime_seconds"] > 0 else 0
        )
        results["columns_subset"] = result
        gc.collect()
        
        # Test 3: Chunked iteration
        def read_chunked():
            parquet_file = pq.ParquetFile(corpus_path)
            total_rows = 0
            for batch in parquet_file.iter_batches(batch_size=10000):
                total_rows += len(batch)
            return total_rows
        
        print("  Profiling: Parquet read (chunked)")
        result = self._time_operation(read_chunked)
        result["method"] = "chunked"
        result["throughput_mb_per_sec"] = (
            file_size_mb / result["runtime_seconds"]
            if result["runtime_seconds"] > 0 else 0
        )
        results["chunked"] = result
        
        return {
            "test": "parquet_read",
            "file_size_mb": file_size_mb,
            "strategies": results,
        }
    
    def profile_parquet_write(self, corpus_path: Path) -> Dict:
        """
        Profile Parquet write performance with different compression.
        
        Tests: gzip, zstd, snappy, none.
        """
        import pandas as pd
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(5000, len(df))
        sample_df = df.head(sample_size).copy()
        
        compression_options = [
            ("gzip", {"compression": "gzip"}),
            ("zstd", {"compression": "zstd"}),
            ("snappy", {"compression": "snappy"}),
            ("none", {"compression": None}),
        ]
        
        results = {}
        
        for name, options in compression_options:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            def write_parquet():
                sample_df.to_parquet(tmp_path, **options, index=False)
                return tmp_path.stat().st_size
            
            print(f"  Profiling: Parquet write ({name})")
            result = self._time_operation(write_parquet)
            
            if tmp_path.exists():
                result["file_size_bytes"] = tmp_path.stat().st_size
                result["file_size_mb"] = result["file_size_bytes"] / (1024 * 1024)
                result["rows_per_second"] = (
                    sample_size / result["runtime_seconds"]
                    if result["runtime_seconds"] > 0 else 0
                )
                tmp_path.unlink()
            
            result["compression"] = name
            results[name] = result
            gc.collect()
        
        return {
            "test": "parquet_write",
            "num_rows": sample_size,
            "compression_results": results,
        }
    
    def profile_checkpoint_write(self, corpus_path: Path) -> Dict:
        """
        Profile checkpoint writing patterns.
        
        Tests: Sync write, async write (simulated), batched writes.
        """
        import pandas as pd
        import threading
        from queue import Queue
        
        df = pd.read_parquet(corpus_path)
        
        # Simulate checkpoint data (200 rows per checkpoint)
        checkpoint_size = 200
        num_checkpoints = 10
        checkpoints = [
            df.head(checkpoint_size * (i + 1)).copy()
            for i in range(num_checkpoints)
        ]
        
        results = {}
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test 1: Synchronous writes (current approach)
            def write_sync():
                for i, ckpt in enumerate(checkpoints):
                    path = temp_dir / f"sync_{i}.parquet"
                    ckpt.to_parquet(path, index=False)
                return num_checkpoints
            
            print("  Profiling: Checkpoint write (synchronous)")
            results["synchronous"] = self._time_operation(write_sync)
            results["synchronous"]["num_checkpoints"] = num_checkpoints
            
            # Cleanup
            for f in temp_dir.glob("sync_*.parquet"):
                f.unlink()
            gc.collect()
            
            # Test 2: Async writes (proposed improvement)
            write_queue = Queue()
            write_complete = threading.Event()
            
            def async_writer():
                while not write_complete.is_set() or not write_queue.empty():
                    try:
                        df_ckpt, path = write_queue.get(timeout=0.1)
                        df_ckpt.to_parquet(path, index=False)
                        write_queue.task_done()
                    except Exception:
                        pass
            
            def write_async():
                writer_thread = threading.Thread(target=async_writer, daemon=True)
                writer_thread.start()
                
                for i, ckpt in enumerate(checkpoints):
                    path = temp_dir / f"async_{i}.parquet"
                    write_queue.put((ckpt.copy(), path))
                
                write_queue.join()
                write_complete.set()
                return num_checkpoints
            
            print("  Profiling: Checkpoint write (asynchronous)")
            results["asynchronous"] = self._time_operation(write_async)
            results["asynchronous"]["num_checkpoints"] = num_checkpoints
            
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "test": "checkpoint_write",
            "checkpoint_size": checkpoint_size,
            "num_checkpoints": num_checkpoints,
            "strategies": results,
        }
    
    def profile_faiss_io(self, corpus_path: Path) -> Dict:
        """
        Profile FAISS index I/O operations.
        
        Tests: Index save/load times, mmap loading.
        """
        import pandas as pd
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load documents and create embeddings
        df = pd.read_parquet(corpus_path)
        sample_size = min(5000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        print("    Generating embeddings for FAISS benchmark...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        results = {}
        
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Test: Index write
            def write_index():
                faiss.write_index(index, str(tmp_path))
                return tmp_path.stat().st_size
            
            print("  Profiling: FAISS index write")
            result = self._time_operation(write_index)
            result["file_size_mb"] = tmp_path.stat().st_size / (1024 * 1024)
            result["vectors_per_second"] = (
                sample_size / result["runtime_seconds"]
                if result["runtime_seconds"] > 0 else 0
            )
            results["write"] = result
            gc.collect()
            
            # Test: Index read (standard)
            def read_index():
                loaded = faiss.read_index(str(tmp_path))
                return loaded.ntotal
            
            print("  Profiling: FAISS index read (standard)")
            results["read_standard"] = self._time_operation(read_index)
            gc.collect()
            
            # Test: Index read with mmap (if supported)
            def read_index_mmap():
                try:
                    loaded = faiss.read_index(str(tmp_path), faiss.IO_FLAG_MMAP)
                    return loaded.ntotal
                except Exception as e:
                    return f"mmap not supported: {e}"
            
            print("  Profiling: FAISS index read (mmap)")
            results["read_mmap"] = self._time_operation(read_index_mmap)
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        
        return {
            "test": "faiss_io",
            "num_vectors": sample_size,
            "dimension": dim,
            "operations": results,
        }
