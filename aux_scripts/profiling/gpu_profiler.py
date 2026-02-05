"""
MIND Profiling - GPU Profiler

Measures GPU utilization and performance for:
- Embedding generation batch size optimization
- FAISS GPU vs CPU comparison
- NLI model batch processing
"""

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class GPUProfiler:
    """Profiles GPU performance of MIND components."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.project_root / "src"))
        
        self.gpu_available = self._check_gpu()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else {}
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "cuda_version": torch.version.cuda,
                }
        except Exception:
            pass
        return {}
    
    def _get_gpu_memory_usage(self) -> Dict:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                    "cached_mb": torch.cuda.memory_reserved() / (1024**2),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
                }
        except Exception:
            pass
        return {}
    
    def _clear_gpu_cache(self):
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        gc.collect()
    
    def profile_embedding_batch_sizes(self, corpus_path: Path) -> Dict:
        """
        Profile embedding generation with different batch sizes on GPU.
        
        Identifies optimal batch size for GPU utilization.
        """
        if not self.gpu_available:
            return {"error": "GPU not available", "test": "embedding_batch_sizes"}
        
        import pandas as pd
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(3000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        
        batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]
        results = {}
        
        for batch_size in batch_sizes:
            self._clear_gpu_cache()
            mem_before = self._get_gpu_memory_usage()
            
            start = time.perf_counter()
            try:
                embeddings = model.encode(
                    documents,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device="cuda"
                )
                # Force synchronization
                import torch
                torch.cuda.synchronize()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f"batch_{batch_size}"] = {
                        "error": "OOM",
                        "batch_size": batch_size,
                    }
                    self._clear_gpu_cache()
                    continue
                raise
            
            elapsed = time.perf_counter() - start
            mem_after = self._get_gpu_memory_usage()
            
            print(f"  Profiling: Embedding GPU (batch_size={batch_size}): {elapsed:.2f}s")
            
            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "runtime_seconds": elapsed,
                "docs_per_second": sample_size / elapsed if elapsed > 0 else 0,
                "gpu_memory_used_mb": mem_after.get("max_allocated_mb", 0),
            }
            
            del embeddings
            self._clear_gpu_cache()
        
        # Find optimal batch size
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            optimal = max(valid_results.items(), key=lambda x: x[1].get("docs_per_second", 0))
            optimal_info = {
                "optimal_batch_size": optimal[1].get("batch_size"),
                "optimal_throughput": optimal[1].get("docs_per_second"),
            }
        else:
            optimal_info = {}
        
        return {
            "test": "embedding_batch_sizes",
            "gpu_info": self.gpu_info,
            "num_documents": sample_size,
            "batch_results": results,
            **optimal_info,
        }
    
    def profile_faiss_gpu_vs_cpu(self, corpus_path: Path) -> Dict:
        """
        Compare FAISS GPU vs CPU search performance.
        """
        import pandas as pd
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load documents and create embeddings
        df = pd.read_parquet(corpus_path)
        sample_size = min(5000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        print("    Generating embeddings for FAISS GPU benchmark...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(documents, batch_size=64, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Generate test queries
        num_queries = 100
        query_embeddings = embeddings[:num_queries].copy()
        
        dim = embeddings.shape[1]
        results = {}
        
        # Test 1: CPU FAISS search
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeddings)
        
        def search_cpu():
            D, I = cpu_index.search(query_embeddings, 10)
            return I.shape[0]
        
        print("  Profiling: FAISS search (CPU)")
        start = time.perf_counter()
        search_cpu()
        elapsed = time.perf_counter() - start
        
        results["cpu"] = {
            "runtime_seconds": elapsed,
            "queries_per_second": num_queries / elapsed if elapsed > 0 else 0,
        }
        gc.collect()
        
        # Test 2: GPU FAISS search (if available)
        if self.gpu_available:
            try:
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                
                def search_gpu():
                    D, I = gpu_index.search(query_embeddings, 10)
                    return I.shape[0]
                
                # Warmup
                search_gpu()
                
                print("  Profiling: FAISS search (GPU)")
                start = time.perf_counter()
                search_gpu()
                elapsed = time.perf_counter() - start
                
                results["gpu"] = {
                    "runtime_seconds": elapsed,
                    "queries_per_second": num_queries / elapsed if elapsed > 0 else 0,
                }
                
                # Calculate speedup
                cpu_time = results["cpu"]["runtime_seconds"]
                gpu_time = results["gpu"]["runtime_seconds"]
                results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
                
            except Exception as e:
                results["gpu"] = {"error": str(e)}
        else:
            results["gpu"] = {"error": "GPU FAISS not available"}
        
        return {
            "test": "faiss_gpu_vs_cpu",
            "num_vectors": sample_size,
            "num_queries": num_queries,
            "comparison": results,
        }
    
    def profile_nli_batch_sizes(self) -> Dict:
        """
        Profile NLI model batch processing on GPU.
        """
        if not self.gpu_available:
            return {"error": "GPU not available", "test": "nli_batch_sizes"}
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            return {"error": "transformers not installed", "test": "nli_batch_sizes"}
        
        # Load NLI model
        model_name = "potsawee/deberta-v3-large-mnli"
        print(f"    Loading NLI model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = model.to("cuda")
            model.eval()
        except Exception as e:
            return {"error": f"Failed to load model: {e}", "test": "nli_batch_sizes"}
        
        # Generate test pairs
        test_pairs = [
            ("The sky is blue.", "The sky has a blue color."),
            ("Cats are animals.", "Dogs are not cats."),
            ("The treatment is effective.", "The treatment works well."),
        ] * 50  # 150 pairs
        
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            self._clear_gpu_cache()
            
            start = time.perf_counter()
            try:
                all_probs = []
                for i in range(0, len(test_pairs), batch_size):
                    batch = test_pairs[i:i + batch_size]
                    inputs = tokenizer.batch_encode_plus(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True
                    )
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        all_probs.extend(probs.cpu().tolist())
                
                torch.cuda.synchronize()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f"batch_{batch_size}"] = {"error": "OOM", "batch_size": batch_size}
                    self._clear_gpu_cache()
                    continue
                raise
            
            elapsed = time.perf_counter() - start
            mem = self._get_gpu_memory_usage()
            
            print(f"  Profiling: NLI GPU (batch_size={batch_size}): {elapsed:.2f}s")
            
            results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "runtime_seconds": elapsed,
                "pairs_per_second": len(test_pairs) / elapsed if elapsed > 0 else 0,
                "gpu_memory_used_mb": mem.get("max_allocated_mb", 0),
            }
            
            self._clear_gpu_cache()
        
        # Cleanup
        del model
        del tokenizer
        self._clear_gpu_cache()
        
        # Find optimal
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            optimal = max(valid_results.items(), key=lambda x: x[1].get("pairs_per_second", 0))
            optimal_info = {
                "optimal_batch_size": optimal[1].get("batch_size"),
                "optimal_throughput": optimal[1].get("pairs_per_second"),
            }
        else:
            optimal_info = {}
        
        return {
            "test": "nli_batch_sizes",
            "model": model_name,
            "num_pairs": len(test_pairs),
            "batch_results": results,
            **optimal_info,
        }
    
    def profile_mixed_precision(self, corpus_path: Path) -> Dict:
        """
        Profile FP16 vs FP32 embedding generation.
        """
        if not self.gpu_available:
            return {"error": "GPU not available", "test": "mixed_precision"}
        
        import pandas as pd
        import torch
        from sentence_transformers import SentenceTransformer
        
        df = pd.read_parquet(corpus_path)
        sample_size = min(2000, len(df))
        documents = df["text"].head(sample_size).tolist()
        
        results = {}
        
        # Test FP32
        self._clear_gpu_cache()
        model_fp32 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        
        start = time.perf_counter()
        emb_fp32 = model_fp32.encode(documents, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        torch.cuda.synchronize()
        elapsed_fp32 = time.perf_counter() - start
        mem_fp32 = self._get_gpu_memory_usage()
        
        print(f"  Profiling: Embedding FP32: {elapsed_fp32:.2f}s")
        results["fp32"] = {
            "runtime_seconds": elapsed_fp32,
            "docs_per_second": sample_size / elapsed_fp32 if elapsed_fp32 > 0 else 0,
            "gpu_memory_mb": mem_fp32.get("max_allocated_mb", 0),
        }
        
        del model_fp32, emb_fp32
        self._clear_gpu_cache()
        
        # Test FP16
        model_fp16 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        model_fp16.half()  # Convert to FP16
        
        start = time.perf_counter()
        emb_fp16 = model_fp16.encode(documents, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        torch.cuda.synchronize()
        elapsed_fp16 = time.perf_counter() - start
        mem_fp16 = self._get_gpu_memory_usage()
        
        print(f"  Profiling: Embedding FP16: {elapsed_fp16:.2f}s")
        results["fp16"] = {
            "runtime_seconds": elapsed_fp16,
            "docs_per_second": sample_size / elapsed_fp16 if elapsed_fp16 > 0 else 0,
            "gpu_memory_mb": mem_fp16.get("max_allocated_mb", 0),
        }
        
        # Calculate improvements
        results["speedup"] = elapsed_fp32 / elapsed_fp16 if elapsed_fp16 > 0 else 0
        results["memory_reduction"] = 1 - (
            mem_fp16.get("max_allocated_mb", 0) / mem_fp32.get("max_allocated_mb", 1)
        )
        
        del model_fp16, emb_fp16
        self._clear_gpu_cache()
        
        return {
            "test": "mixed_precision",
            "num_documents": sample_size,
            "precision_results": results,
        }
