# MIND Project Optimization Guide

> **Document Version:** 1.0  
> **Last Updated:** 2026-02-01  
> **Target Audience:** Human Developers, Project Maintainers

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Optimization Roadmap](#3-optimization-roadmap)
4. [Memory (RAM) Optimizations](#4-memory-ram-optimizations)
5. [Disk I/O Optimizations](#5-disk-io-optimizations)
6. [GPU Optimizations](#6-gpu-optimizations)
7. [Runtime Optimizations](#7-runtime-optimizations)
8. [Architecture Refactoring](#8-architecture-refactoring)
9. [Priority Matrix](#9-priority-matrix)
10. [Implementation Strategy](#10-implementation-strategy)

---

## 1. Executive Summary

This document provides a comprehensive analysis of optimization opportunities within the MIND project's `src/mind/` codebase. The analysis focuses on four core metrics:

| Metric | Current State | Optimization Potential |
|--------|---------------|------------------------|
| **RAM Usage** | High (full DataFrame loading) | 40-60% reduction possible |
| **Disk I/O** | Moderate (Parquet with gzip) | 20-30% improvement possible |
| **GPU Usage** | Suboptimal (sequential embedding) | 50-70% better utilization |
| **Runtime** | Sequential processing | 30-50% speedup achievable |

### Key Bottlenecks Identified

1. **Memory**: Full corpus DataFrames loaded into memory; no chunked processing
2. **I/O**: Subprocess-based NLPipe creates temporary files; checkpoint writes are synchronous
3. **GPU**: Embeddings computed one-at-a-time in retrieval; no batching during pipeline execution
4. **Runtime**: Sequential topic/chunk processing; no parallelization of independent LLM calls

---

## 1.1 Profiling Baseline Results (2026-02-04)

The profiling suite was executed on 2026-02-04 to establish a baseline for optimization efforts. The results are stored in `aux_scripts/profiling/profiling/results/profiling_results_baseline.json` and can be visualized using the Jupyter notebook at `aux_scripts/profiling/analysis/profiling_visualization.ipynb`.

### Memory Profiling Highlights

| Strategy | Peak Memory (10k docs) | RSS Increase |
|----------|------------------------|--------------|
| `full_pandas` | 41.7 MB | 143.3 MB |
| `pyarrow_destruct` | 41.7 MB | 91.6 MB |
| **`chunked`** | **20.9 MB** | **27.4 MB** |

**Recommendation:** Use `chunked` loading for 50% memory reduction.

### Runtime Profiling Highlights

| Operation | Strategy | Performance |
|-----------|----------|-------------|
| Retrieval | `single_query` | ~60 queries/sec |
| Retrieval | **`batched_query`** | **~160 queries/sec (2.7x faster)** |

| Embedding Batch Size | Throughput (docs/sec) |
|---------------------|------------------------|
| 1 | 12.9 |
| **8** | **15.0 (optimal)** |
| 16 | 14.1 |
| 32 | 13.5 |
| 128 | 13.0 |

**Recommendation:** Use batch size of 8 for embeddings; always use batched queries.

### I/O Profiling Highlights (Parquet Write @ 5000 rows)

| Compression | Write Speed | File Size |
|-------------|-------------|-----------|
| `gzip` | 17,088 rows/sec | 1.10 MB |
| **`zstd`** | **58,148 rows/sec** | **1.59 MB** |
| `snappy` | 66,529 rows/sec | 2.44 MB |
| `none` | 68,061 rows/sec | 19.75 MB |

**Recommendation:** Use `zstd` for the best balance of speed (3.4x faster than gzip) and compression (35% smaller than snappy).

### GPU Profiling

GPU profiling returned "No GPU available" for this baseline run. GPU-specific optimizations (FAISS GPU, mixed precision) are outlined in Section 6.



## 2. Current State Analysis

### 2.1 Module Overview

| Module | Lines | Primary Function | Key Issues |
|--------|-------|------------------|------------|
| `pipeline/pipeline.py` | 683 | Main MIND pipeline | Sequential processing, no batching |
| `pipeline/retriever.py` | 420 | FAISS-based retrieval | Single-query embeddings, recomputes for each subquery |
| `pipeline/corpus.py` | 226 | Corpus management | Full DataFrame in memory, row-by-row iteration |
| `prompter/prompter.py` | 314 | LLM interface | Good caching, but synchronous calls only |
| `corpus_building/translator.py` | 262 | NMT translation | HuggingFace Dataset batching (good), but sentence-level splitting is slow |
| `corpus_building/data_preparer.py` | 412 | NLPipe preprocessing | Subprocess overhead, temporary file creation |
| `corpus_building/segmenter.py` | 103 | Document segmentation | Row-by-row iteration (slow for large corpora) |
| `topic_modeling/polylingual_tm.py` | 579 | Mallet PLTM wrapper | Large state parsing in memory, subprocess I/O |

### 2.2 Data Flow Bottlenecks

```
Input Corpus
    │
    ├── [BOTTLENECK 1] Segmenter iterates row-by-row with tqdm
    │
    ├── [BOTTLENECK 2] Translator splits sentences sequentially
    │
    ├── [BOTTLENECK 3] DataPreparer spawns subprocess per language
    │
    ├── [BOTTLENECK 4] PolylingualTM parses large gzip state file in memory
    │
    ├── [BOTTLENECK 5] IndexRetriever computes one embedding per query
    │
    └── [BOTTLENECK 6] MIND pipeline processes chunks sequentially
```

---

## 3. Optimization Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- Enable GPU batching in embeddings
- Add async LLM calls where possible
- Implement chunked DataFrame reading
- Replace row iteration with vectorized operations

### Phase 2: Medium Effort (2-4 weeks)
- Implement parallel topic processing
- Add memory-mapped FAISS indices
- Batch subquery embeddings
- Optimize checkpoint I/O

### Phase 3: Architecture Changes (4-8 weeks)
- Replace subprocess NLPipe with in-process spaCy
- Implement streaming pipeline architecture
- Add distributed processing support (Ray/Dask)
- GPU-accelerated topic modeling alternative

---

## 4. Memory (RAM) Optimizations

### 4.1 Chunked DataFrame Loading

**Current**: Full corpus loaded with `pd.read_parquet(path)` in multiple modules.

**Problem**: For corpora with 100K+ documents, this consumes several GB of RAM.

**Solution**: Use PyArrow incremental reading:

```python
# Instead of:
df = pd.read_parquet(path)

# Use:
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile(path)
for batch in parquet_file.iter_batches(batch_size=10000):
    df_chunk = batch.to_pandas()
    # Process chunk
```

**Affected Files**:
- `corpus_building/segmenter.py` (line 40)
- `corpus_building/translator.py` (line 196)
- `corpus_building/data_preparer.py` (lines 254-255)
- `topic_modeling/polylingual_tm.py` (line 159)
- `pipeline/corpus.py` (lines 99-100)

**Estimated Impact**: 40-60% RAM reduction for large corpora.

---

### 4.2 Generator-Based Chunk Iteration

**Current**: `chunks_with_topic()` in `corpus.py` yields chunks but the DataFrame is still fully loaded.

**Problem**: The underlying DataFrame remains in memory even when processing streamed chunks.

**Solution**: Implement lazy chunk loading with row-group filtering:

```python
def chunks_with_topic(self, topic_id, sample_size=None):
    # Filter only required columns from Parquet
    columns = ["doc_id", "text", "full_doc", self.row_top_k, "main_topic_thetas"]
    
    parquet_file = pq.ParquetFile(self.path)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=1000):
        df_batch = batch.to_pandas()
        topic_rows = df_batch[df_batch.main_topic_thetas == topic_id]
        for _, row in topic_rows.iterrows():
            yield self._make_chunk(row)
```

**Affected Files**:
- `pipeline/corpus.py` (lines 144-201)

---

### 4.3 Sparse Matrix Optimization for Thetas

**Current**: Thetas are stored as sparse matrices but converted to dense arrays for processing.

**Problem**: `thetas.toarray()` converts sparse to dense, negating memory savings.

**Solution**: Keep operations in sparse format:

```python
# Instead of:
thetas = sparse.load_npz(path).toarray()
df["thetas"] = list(thetas)

# Use sparse row access:
thetas_sparse = sparse.load_npz(path)
# Access individual rows as sparse:
for i in range(thetas_sparse.shape[0]):
    theta_row = thetas_sparse.getrow(i)
```

**Affected Files**:
- `pipeline/retriever.py` (lines 150-152, 217-222)
- `pipeline/corpus.py` (line 110)

---

### 4.4 State File Streaming for Topic Models

**Current**: `save_model_info()` in `polylingual_tm.py` loads the entire `output-state.gz` (often 500MB+) into a DataFrame.

**Problem**: Memory spike during post-processing.

**Solution**: Stream-parse the gzip file:

```python
import gzip

betas = np.zeros((num_topics, vocab_size))
with gzip.open(topic_state_model, 'rt') as fin:
    next(fin)  # Skip header
    for line in fin:
        parts = line.split()
        tpc = int(parts[5])
        vocab_id = int(parts[3])
        betas[tpc, vocab_id] += 1
```

**Affected Files**:
- `topic_modeling/polylingual_tm.py` (lines 387-392)

**Estimated Impact**: 50-70% peak memory reduction during topic model training.

---

## 5. Disk I/O Optimizations

### 5.1 Replace Subprocess NLPipe with In-Process spaCy

**Current**: `DataPreparer._preprocess_df()` calls NLPipe as a subprocess, writing/reading temp Parquet files.

**Problem**: 
- Subprocess spawn overhead (~100ms per call)
- Temp file I/O (write + read)
- Process startup loads spaCy models each time

**Solution**: Direct spaCy integration:

```python
import spacy

class DataPreparer:
    def __init__(self, ...):
        # Load spaCy models once
        self._nlp_models = {}
    
    def _get_nlp(self, lang: str):
        if lang not in self._nlp_models:
            model_name = self._spacy_model_for(lang)
            self._nlp_models[lang] = spacy.load(model_name, disable=["ner", "parser"])
        return self._nlp_models[lang]
    
    def _lemmatize_batch(self, texts: List[str], lang: str) -> List[str]:
        nlp = self._get_nlp(lang)
        docs = nlp.pipe(texts, batch_size=1000, n_process=4)
        return [" ".join(tok.lemma_ for tok in doc) for doc in docs]
```

**Affected Files**:
- `corpus_building/data_preparer.py` (lines 143-219)

**Estimated Impact**: 
- 10x faster preprocessing for small corpora
- Elimination of temp file I/O

---

### 5.2 Async Checkpoint Writes

**Current**: Results checkpointed every 200 entries with synchronous `to_parquet()`.

**Problem**: I/O blocks pipeline execution.

**Solution**: Background thread for checkpointing:

```python
import threading
from queue import Queue

class AsyncCheckpointer:
    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()
    
    def _writer(self):
        while True:
            df, path = self.queue.get()
            df.to_parquet(path, index=False)
            self.queue.task_done()
    
    def save(self, df, path):
        self.queue.put((df.copy(), path))
```

**Affected Files**:
- `pipeline/pipeline.py` (lines 441-464)

---

### 5.3 Memory-Mapped FAISS Indices

**Current**: FAISS indices loaded fully into RAM with `faiss.read_index()`.

**Problem**: Large indices (500MB+) consume significant RAM.

**Solution**: Use FAISS IO_FLAG for memory mapping:

```python
# Load with memory mapping
index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
```

**Affected Files**:
- `pipeline/retriever.py` (lines 127, 141)

**Note**: Requires FAISS compiled with `NO_MMAP` disabled.

---

### 5.4 Optimized Parquet Compression

**Current**: Uses gzip compression throughout.

**Solution**: Switch to zstd for better compression ratio and speed:

```python
df.to_parquet(path, compression="zstd", compression_level=3)
```

**Trade-off**: Slightly larger files but 2-3x faster read/write.

---

## 6. GPU Optimizations

### 6.1 Batched Query Embeddings

**Current**: Single embedding computed per retrieval query in `retrieve_topic_faiss()` and `retrieve_enn_ann()`.

**Problem**: GPU underutilized (tiny batches).

**Solution**: Collect subqueries and batch encode:

```python
# In _process_question, collect all subqueries first:
all_subqueries = self._generate_subqueries(question, chunk)
query_embeddings = self.embedding_model.encode(
    all_subqueries, 
    batch_size=32, 
    show_progress_bar=False,
    convert_to_numpy=True
)
# Then retrieve for each
for subquery, embedding in zip(all_subqueries, query_embeddings):
    results = self.retriever.retrieve_with_embedding(embedding)
```

**Affected Files**:
- `pipeline/pipeline.py` (lines 264-271)
- `pipeline/retriever.py` (add `retrieve_with_embedding()` method)

**Estimated Impact**: 5-10x faster retrieval when processing multiple subqueries.

---

### 6.2 Mixed Precision Embeddings

**Current**: Full FP32 embeddings throughout.

**Solution**: Use FP16 for GPU-resident operations:

```python
model = SentenceTransformer(model_name)
model.half()  # Convert to FP16

# Or use encode with precision setting
embeddings = model.encode(texts, convert_to_tensor=True)
embeddings = embeddings.half()
```

**Affected Files**:
- `pipeline/retriever.py` (line 185)
- `pipeline/pipeline.py` (line 165)

**Trade-off**: Marginal accuracy loss, significant memory savings.

---

### 6.3 GPU FAISS Index

**Current**: CPU FAISS indices by default.

**Solution**: Use GPU-accelerated indices:

```python
import faiss

# For indexing
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# For search
distances, indices = gpu_index.search(query_embeddings, top_k)
```

**Affected Files**:
- `pipeline/retriever.py` (entire `index()` and `retrieve*` methods)

**Requirements**: FAISS with GPU support (`faiss-gpu` package).

---

### 6.4 NLI Model Batching

**Current**: NLI entailment check processes one pair at a time.

**Solution**: Batch NLI checks:

```python
def _batch_check_entailment(self, pairs: List[Tuple[str, str]], threshold=0.5):
    inputs = self._nli_tokenizer.batch_encode_plus(
        pairs,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        logits = self._nli_model(**inputs.to("cuda")).logits
        probs = torch.softmax(logits, dim=-1)
    return [(p[0].item(), p[1].item(), p[0].item() >= threshold) for p in probs]
```

**Affected Files**:
- `pipeline/pipeline.py` (lines 640-670)

---

## 7. Runtime Optimizations

### 7.1 Parallel Topic Processing

**Current**: Topics processed sequentially in `run_pipeline()`.

**Solution**: Use multiprocessing or concurrent futures:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_pipeline(self, topics, sample_size=None, ...):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(self._process_topic, topic, path_save, ...): topic
            for topic in topics
        }
        for future in as_completed(futures):
            topic = futures[future]
            try:
                future.result()
            except Exception as e:
                self._logger.error(f"Topic {topic} failed: {e}")
```

**Affected Files**:
- `pipeline/pipeline.py` (lines 191-198)

**Considerations**: 
- Need to handle shared state (results list, seen_triplets)
- May need to use multiprocessing Manager for shared data

---

### 7.2 Async LLM Calls

**Current**: LLM calls are synchronous via requests/OpenAI client.

**Solution**: Use async clients:

```python
import asyncio
from openai import AsyncOpenAI

class AsyncPrompter:
    def __init__(self, ...):
        self.async_client = AsyncOpenAI()
    
    async def prompt_async(self, question: str, ...):
        response = await self.async_client.chat.completions.create(
            model=self.model_type,
            messages=messages,
            **params
        )
        return response.choices[0].message.content

# Usage in pipeline:
async def _process_questions_async(self, questions, chunk, topic):
    tasks = [self._prompter.prompt_async(q, ...) for q in questions]
    return await asyncio.gather(*tasks)
```

**Affected Files**:
- `prompter/prompter.py` (add async methods)
- `pipeline/pipeline.py` (convert to async where beneficial)

**Note**: Ollama client also supports async operations.

---

### 7.3 Vectorized Segmentation

**Current**: Row-by-row iteration with tqdm in `Segmenter.segment()`.

**Solution**: Pandas vectorized operations:

```python
def segment(self, path_df, path_save, text_col="text", ...):
    df = pd.read_parquet(path_df)
    
    # Explode paragraphs into rows
    df["paragraphs"] = df[text_col].str.split(sep)
    df = df.explode("paragraphs")
    
    # Filter by length
    df = df[df["paragraphs"].str.len() > min_length]
    
    # Reset IDs
    df["id_preproc"] = df.groupby(level=0).cumcount().astype(str)
    df["id_preproc"] = df["id_preproc_orig"] + "_" + df["id_preproc"]
    
    df.to_parquet(path_save, compression="gzip")
```

**Affected Files**:
- `corpus_building/segmenter.py` (lines 52-63)

**Estimated Impact**: 10-50x speedup for large corpora.

---

### 7.4 Cached Embedding Model Loading

**Current**: `SentenceTransformer(model_name)` called in multiple places.

**Problem**: Model loaded multiple times across pipeline components.

**Solution**: Singleton pattern for embedding models:

```python
class EmbeddingModelCache:
    _models = {}
    
    @classmethod
    def get(cls, model_name: str, device: str = "cuda"):
        if model_name not in cls._models:
            cls._models[model_name] = SentenceTransformer(model_name, device=device)
        return cls._models[model_name]
```

**Affected Files**:
- `pipeline/pipeline.py` (line 165)
- `pipeline/retriever.py` (constructor)

---

### 7.5 Translation Optimization

**Current**: Sentence splitting before translation is sequential.

**Solution**: Parallel sentence splitting with multiprocessing:

```python
from multiprocessing import Pool

def _split_parallel(self, df, ...):
    with Pool(processes=4) as pool:
        results = pool.starmap(
            self._split_row,
            [(row, tokenizer, max_tokens) for _, row in df.iterrows()]
        )
    return pd.concat(results, ignore_index=True)
```

**Affected Files**:
- `corpus_building/translator.py` (lines 39-82)

---

## 8. Architecture Refactoring

### 8.1 Streaming Pipeline Architecture

Replace batch-oriented processing with a streaming architecture:

```
Input Stream → Segmenter → Translator → DataPreparer → TopicModeler → MIND Pipeline → Output Stream
     │              │            │             │              │               │
     └──────────────┴────────────┴─────────────┴──────────────┴───────────────┘
                                    Backpressure Control
```

**Benefits**:
- Constant memory usage regardless of corpus size
- Enables real-time processing
- Natural parallelization points

**Implementation**: Consider using Ray Data or Dask for distributed streaming.

---

### 8.2 Index Persistence Improvements

**Current**: FAISS indices rebuilt if not found; stored per topic.

**Improvements**:
1. Add index versioning based on corpus hash
2. Implement incremental index updates
3. Use `faiss.index_factory()` for better index type selection

---

### 8.3 Configuration-Driven Optimization Profiles

Add optimization profiles to `config.yaml`:

```yaml
optimization:
  profile: balanced  # or: memory_optimized, speed_optimized, gpu_heavy
  
  memory_optimized:
    chunk_size: 5000
    faiss_mmap: true
    sparse_thetas: true
    
  speed_optimized:
    parallel_topics: 4
    async_llm: true
    gpu_faiss: true
    batch_embeddings: 64
```

---

## 9. Priority Matrix

| Optimization | Impact | Effort | Priority | Risk |
|--------------|--------|--------|----------|------|
| Batched embeddings | High | Low | 1 | Low |
| Vectorized segmentation | High | Low | 1 | Low |
| Chunked DataFrame loading | High | Medium | 2 | Low |
| In-process spaCy | High | Medium | 2 | Medium |
| Async checkpoints | Medium | Low | 3 | Low |
| GPU FAISS | High | Medium | 3 | Medium |
| Async LLM calls | Medium | Medium | 4 | Medium |
| Parallel topics | High | High | 4 | High |
| Streaming architecture | Very High | Very High | 5 | High |

**Priority Legend**:
- 1 = Implement immediately
- 5 = Long-term refactoring

---

## 10. Implementation Strategy

### 10.1 Testing Considerations

Before implementing optimizations:

1. **Establish baselines**: Measure current RAM, disk I/O, GPU utilization, and runtime
2. **Create regression tests**: Ensure output consistency after changes
3. **Profile incrementally**: Use `py-spy`, `memory_profiler`, and `nvtop`

### 10.2 Rollout Order

1. **Week 1-2**: 
   - Implement vectorized segmentation
   - Add batched embeddings in retriever
   - Add async checkpointing

2. **Week 3-4**:
   - Replace NLPipe subprocess with in-process spaCy
   - Implement chunked DataFrame reading
   - Add GPU FAISS support

3. **Week 5-6**:
   - Async LLM calls
   - Parallel topic processing
   - Memory-mapped FAISS

4. **Week 7+**:
   - Streaming architecture investigation
   - Distributed processing prototype

### 10.3 Compatibility Notes

- All optimizations must maintain Python 3.12 compatibility
- Preserve Parquet file format for interoperability
- Keep Mallet integration (no pure-Python PLTM replacement yet)
- Maintain OpenAI/Ollama/vLLM backend flexibility

---

## Appendix A: Profiling Commands

```bash
# Memory profiling
python -m memory_profiler script.py

# CPU profiling
py-spy record -o profile.svg -- python script.py

# GPU monitoring
watch -n 0.5 nvidia-smi

# Disk I/O monitoring
iotop -aoP
```

---

## Appendix B: Benchmark Setup

Create a standardized benchmark corpus:

```python
# Generate synthetic benchmark data
benchmark_sizes = [1000, 10000, 100000]
for size in benchmark_sizes:
    create_benchmark_corpus(size, f"benchmark_{size}.parquet")
```

Measure each optimization against these benchmarks.

---

**End of Optimization Guide**
