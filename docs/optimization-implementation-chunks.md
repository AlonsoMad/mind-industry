# MIND Project Optimization Implementation Chunks

> **Purpose**: AI-digestible optimization specifications for agentic coders  
> **Format**: Each chunk is a self-contained optimization task with context, current code, target code, and verification steps

---

## Chunk Index

| ID | Module | Optimization Type | Effort | Dependencies |
|----|--------|-------------------|--------|--------------|
| [OPT-001](#opt-001) | segmenter.py | Runtime | Low | None |
| [OPT-002](#opt-002) | retriever.py | GPU/Runtime | Low | None |
| [OPT-003](#opt-003) | corpus.py | Memory | Medium | OPT-001 |
| [OPT-004](#opt-004) | pipeline.py | Disk I/O | Low | None |
| [OPT-005](#opt-005) | data_preparer.py | Disk I/O/Runtime | Medium | None |
| [OPT-006](#opt-006) | polylingual_tm.py | Memory | Medium | None |
| [OPT-007](#opt-007) | retriever.py | GPU | Medium | OPT-002 |
| [OPT-008](#opt-008) | translator.py | Runtime | Low | None |
| [OPT-009](#opt-009) | pipeline.py | Runtime | High | OPT-002, OPT-007 |
| [OPT-010](#opt-010) | prompter.py | Runtime | Medium | None |
| [OPT-011](#opt-011) | retriever.py | Memory/Disk | Medium | None |
| [OPT-012](#opt-012) | pipeline.py | GPU | Medium | None |

---

## OPT-001

### Vectorized Document Segmentation

**File**: `src/mind/corpus_building/segmenter.py`  
**Type**: Runtime  
**Effort**: Low  
**Impact**: 10-50x speedup for large corpora

#### Context

The `Segmenter.segment()` method iterates row-by-row over a DataFrame to split documents into paragraphs. This is slow for corpora with 10K+ documents.

#### Current Code Location

Lines 52-63:

```python
for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting paragraphs"):
    full_doc_text = str(row[text_col])
    paragraphs = [p for p in full_doc_text.split(
        sep) if p and len(p) > min_length]
    for idx, p in enumerate(paragraphs):
        entry = {col: row.get(col, None) for col in orig_cols}
        entry[text_col] = p  # replace with paragraph
        entry['full_doc'] = full_doc_text  # add full document text
        entry['id'] = None  # will set below
        entry['id_preproc'] = f"{row.get(id_col, '')}_{idx}"
        new_rows.append(entry)
```

#### Target Implementation

Replace lines 44-72 with:

```python
def segment(
    self,
    path_df: Path,
    path_save: Path,
    text_col: str = "text",
    id_col: str = "id_preproc",
    min_length: int = 100,
    sep: str = "\n"
):
    self._logger.info(f"Loading dataframe from {path_df}")
    df = pd.read_parquet(path_df)
    orig_cols = list(df.columns)
    self._logger.info(f"Loaded {len(df)} rows. Starting vectorized segmentation...")
    
    import time
    start_time = time.time()
    
    # Preserve original document text before exploding
    df["full_doc"] = df[text_col].astype(str)
    df["_orig_id"] = df[id_col].astype(str)
    
    # Split text into list of paragraphs (vectorized)
    df["_paragraphs"] = df[text_col].str.split(sep)
    
    # Explode to one row per paragraph
    df = df.explode("_paragraphs", ignore_index=True)
    
    # Filter short/empty paragraphs
    df = df[df["_paragraphs"].str.len() > min_length].copy()
    
    # Replace text column with paragraph content
    df[text_col] = df["_paragraphs"]
    
    # Generate sequential index per original document
    df["_para_idx"] = df.groupby("_orig_id").cumcount().astype(str)
    df["id_preproc"] = df["_orig_id"] + "_" + df["_para_idx"]
    
    # Clean up temporary columns
    df = df.drop(columns=["_paragraphs", "_orig_id", "_para_idx"])
    
    # Reset global ID
    df["id"] = range(len(df))
    
    elapsed = time.time() - start_time
    self._logger.info(f"Vectorized segmentation took {elapsed:.2f} seconds.")
    self._logger.info(f"Segmented into {len(df)} paragraphs. Saving to {path_save}")
    
    df.to_parquet(path_save, compression="gzip")
    self._logger.info(f"Saved segmented dataframe to {path_save}")
    return path_save
```

#### Verification Steps

1. Run with existing test corpus and compare output row count
2. Verify `id_preproc` format matches pattern `{orig_id}_{idx}`
3. Benchmark: `time python -m cProfile segmenter.py --input test.parquet --output out.parquet`
4. Memory check: `python -m memory_profiler segmenter.py`

---

## OPT-002

### Batched Query Embeddings in Retriever

**File**: `src/mind/pipeline/retriever.py`  
**Type**: GPU/Runtime  
**Effort**: Low  
**Impact**: 5-10x faster retrieval for multiple queries

#### Context

`IndexRetriever.retrieve()` and related methods encode one query embedding at a time, underutilizing GPU parallelism.

#### Current Code Location

Lines 282-285 (retrieve_topic_faiss):

```python
query_embedding = self.embedding_model.encode(
    query, show_progress_bar=False, convert_to_numpy=True
)
```

Similar pattern in `retrieve_enn_ann()` at lines 356-359.

#### Target Implementation

Add new method after line 280:

```python
def encode_queries(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
    """Batch encode multiple queries for efficient GPU utilization."""
    if isinstance(queries, str):
        queries = [queries]
    embeddings = self.embedding_model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    if self.do_norm:
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, axis=1, norm='l2')
    return embeddings

def retrieve_with_embedding(
    self,
    query_embedding: np.ndarray,
    theta_query: np.ndarray = None,
    mode: str = None
) -> Tuple[List[Dict], int]:
    """Retrieve using pre-computed embedding to support batched retrieval."""
    mode = mode or self.mode
    
    if mode == "TB-ENN":
        return self._retrieve_topic_filtered_enn(query_embedding, theta_query)
    elif mode == "TB-ANN":
        return self._retrieve_topic_filtered_ann(query_embedding, theta_query)
    elif mode == "ENN":
        return self._retrieve_enn_with_embedding(query_embedding)
    elif mode == "ANN":
        return self._retrieve_ann_with_embedding(query_embedding)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

def _retrieve_enn_with_embedding(self, query_embedding: np.ndarray) -> Tuple[List[Dict], int]:
    """ENN retrieval with pre-computed embedding."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    distances, indices = self._enn_index.search(
        query_embedding.astype(np.float32), 
        self.top_k
    )
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= 0:
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(dist)
            })
    return results, len(results)
```

Modify existing methods to use internal helper:

```python
def retrieve_enn_ann(self, query: str, index: str = "enn") -> Tuple[List[Dict], int]:
    """Single query wrapper - for backward compatibility."""
    query_embedding = self.encode_queries([query])[0]
    if index == "enn":
        return self._retrieve_enn_with_embedding(query_embedding)
    else:
        return self._retrieve_ann_with_embedding(query_embedding)
```

#### Verification Steps

1. Unit test: Verify `encode_queries(["q1", "q2", "q3"])` returns shape `(3, embedding_dim)`
2. Integration test: Compare retrieval results with original method
3. Benchmark: Compare GPU utilization before/after with `nvidia-smi`

---

## OPT-003

### Chunked DataFrame Loading in Corpus

**File**: `src/mind/pipeline/corpus.py`  
**Type**: Memory  
**Effort**: Medium  
**Impact**: 40-60% RAM reduction for large corpora

#### Context

`Corpus.from_parquet_and_thetas()` loads the entire Parquet file into memory, which is problematic for 100K+ document corpora.

#### Current Code Location

Lines 99-100:

```python
table = pq.read_table(path_parquet)
df = table.to_pandas(self_destruct=True, ignore_metadata=True)
```

#### Target Implementation

Create lazy-loading variant:

```python
@classmethod
def from_parquet_lazy(
    cls,
    path_parquet: Path,
    path_thetas: Path = None,
    batch_size: int = 10000,
    **kwargs
):
    """
    Create Corpus with lazy chunk loading support.
    Only metadata is loaded initially; chunks are streamed on demand.
    """
    logger = kwargs.get("logger") or init_logger(kwargs.get("config_path"), __name__)
    
    # Read only metadata (schema and row groups)
    parquet_file = pq.ParquetFile(path_parquet)
    metadata = parquet_file.metadata
    
    # Initialize with minimal DataFrame (just to get schema)
    first_batch = next(parquet_file.iter_batches(batch_size=10))
    df_schema = first_batch.to_pandas().head(0)
    
    corpus = cls(df_schema, **kwargs)
    
    # Store lazy loading config
    corpus._lazy_mode = True
    corpus._parquet_path = path_parquet
    corpus._thetas_path = path_thetas
    corpus._batch_size = batch_size
    corpus._total_rows = metadata.num_rows
    
    logger.info(f"Lazy corpus initialized for {corpus._total_rows} documents")
    return corpus

def chunks_with_topic_lazy(self, topic_id: int, sample_size: int = None):
    """
    Generator that streams chunks for a specific topic without loading full corpus.
    """
    if not getattr(self, '_lazy_mode', False):
        # Fall back to original method
        yield from self.chunks_with_topic(topic_id, sample_size)
        return
    
    parquet_file = pq.ParquetFile(self._parquet_path)
    required_cols = ["doc_id", "text", "full_doc", self.row_top_k, "main_topic_thetas"]
    
    count = 0
    for batch in parquet_file.iter_batches(batch_size=self._batch_size, columns=required_cols):
        df_batch = batch.to_pandas()
        topic_rows = df_batch[df_batch["main_topic_thetas"] == topic_id]
        
        for _, row in topic_rows.iterrows():
            if sample_size and count >= sample_size:
                return
            
            metadata = {"top_k": row.get(self.row_top_k)}
            yield Chunk(
                id=row["doc_id"],
                text=row["text"],
                full_doc=row.get("full_doc", ""),
                metadata=metadata
            )
            count += 1
```

#### Verification Steps

1. Memory test: Compare RSS memory with `psutil` before/after loading 100K docs
2. Correctness: Verify same chunks yielded as eager loading
3. Performance: Benchmark iteration speed

---

## OPT-004

### Async Checkpoint Writes

**File**: `src/mind/pipeline/pipeline.py`  
**Type**: Disk I/O  
**Effort**: Low  
**Impact**: Eliminates I/O blocking during checkpoint saves

#### Context

Every 200 results, the pipeline synchronously writes checkpoints to Parquet, blocking execution.

#### Current Code Location

Lines 441-464:

```python
if len(self.results) % 200 == 0:
    checkpoint = len(self.results) // 200
    results_checkpoint_path = Path(
        f"{path_save}/results_topic_{topic}_{checkpoint}.parquet")
    discarded_checkpoint_path = Path(
        f"{path_save}/discarded_topic_{topic}_{checkpoint}.parquet")

    df = pd.DataFrame(self.results)
    df_discarded = pd.DataFrame(self.discarded)

    df.to_parquet(results_checkpoint_path, index=False)
    df_discarded.to_parquet(discarded_checkpoint_path, index=False)
    # ... cleanup old checkpoints
```

#### Target Implementation

Add async checkpointer class after imports (around line 20):

```python
import threading
from queue import Queue
from typing import Tuple

class AsyncCheckpointer:
    """Background thread for non-blocking checkpoint writes."""
    
    def __init__(self, logger=None):
        self._queue: Queue[Tuple[pd.DataFrame, Path]] = Queue()
        self._logger = logger
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
    
    def _writer_loop(self):
        while True:
            try:
                df, path, old_path = self._queue.get(timeout=1.0)
                df.to_parquet(path, index=False)
                if old_path and old_path.exists():
                    old_path.unlink()
                self._queue.task_done()
            except Exception:
                pass  # Timeout or error, continue
    
    def save_async(self, df: pd.DataFrame, path: Path, old_path: Path = None):
        """Queue a DataFrame for background saving."""
        self._queue.put((df.copy(), path, old_path))
    
    def wait_complete(self):
        """Wait for all pending saves to complete."""
        self._queue.join()
```

In `MIND.__init__()`, add around line 120:

```python
self._checkpointer = AsyncCheckpointer(logger=self._logger)
```

Replace checkpoint logic in `_evaluate_pair()`:

```python
if len(self.results) % 200 == 0:
    checkpoint = len(self.results) // 200
    results_path = Path(f"{path_save}/results_topic_{topic}_{checkpoint}.parquet")
    discarded_path = Path(f"{path_save}/discarded_topic_{topic}_{checkpoint}.parquet")
    
    old_results = Path(f"{path_save}/results_topic_{topic}_{checkpoint-1}.parquet")
    old_discarded = Path(f"{path_save}/discarded_topic_{topic}_{checkpoint-1}.parquet")
    
    self._checkpointer.save_async(pd.DataFrame(self.results), results_path, old_results)
    self._checkpointer.save_async(pd.DataFrame(self.discarded), discarded_path, old_discarded)
```

Add cleanup in `run_pipeline()` end:

```python
self._checkpointer.wait_complete()
```

#### Verification Steps

1. Run pipeline and verify checkpoint files are created correctly
2. Monitor I/O wait time with `iotop`
3. Unit test: Verify `wait_complete()` blocks until all writes finish

---

## OPT-005

### In-Process spaCy Preprocessing

**File**: `src/mind/corpus_building/data_preparer.py`  
**Type**: Disk I/O, Runtime  
**Effort**: Medium  
**Impact**: 10x faster preprocessing, eliminates temp files

#### Context

`_preprocess_df()` calls NLPipe as a subprocess, creating temporary Parquet files and spawning a new Python process that loads spaCy models each time.

#### Current Code Location

Lines 179-193:

```python
if self.preproc_script and self.config_path and self.stw_path:
    cmd = [
        self.python_exe, str(self.preproc_script),
        "--source_path", str(tmp_parq),
        # ... more args
    ]
    print("Running NLPipe:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

#### Target Implementation

Add to class after `__init__` (around line 80):

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Lazy-loaded spaCy models (replaces subprocess NLPipe)
    self._nlp_cache = {}
    self._stopwords_cache = {}

def _load_nlp(self, lang: str):
    """Load and cache spaCy model for language."""
    lang_upper = lang.upper()
    if lang_upper not in self._nlp_cache:
        import spacy
        model_name = self._spacy_model_for(lang_upper)
        # Disable components we don't need for lemmatization
        nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])
        self._nlp_cache[lang_upper] = nlp
        self._logger.info(f"Loaded spaCy model: {model_name}")
    return self._nlp_cache[lang_upper]

def _load_stopwords(self, lang: str) -> set:
    """Load and cache stopwords for language."""
    lang_lower = lang.lower()
    if lang_lower not in self._stopwords_cache:
        stw_file = self.stw_path / f"{lang_lower}.txt" if self.stw_path else None
        stopwords = set()
        if stw_file and stw_file.exists():
            with open(stw_file, 'r', encoding='utf-8') as f:
                stopwords = {line.strip().lower() for line in f if line.strip()}
        self._stopwords_cache[lang_lower] = stopwords
    return self._stopwords_cache[lang_lower]

def _lemmatize_texts(
    self,
    texts: List[str],
    lang: str,
    batch_size: int = 1000,
    n_process: int = 4
) -> List[str]:
    """
    Lemmatize a list of texts using spaCy in-process.
    Much faster than subprocess NLPipe.
    """
    nlp = self._load_nlp(lang)
    stopwords = self._load_stopwords(lang)
    
    lemmatized = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lemma_.lower() not in stopwords
        ]
        lemmatized.append(" ".join(lemmas))
    
    return lemmatized
```

Replace `_preprocess_df()` implementation (lines 143-219):

```python
def _preprocess_df(
    self,
    df: pd.DataFrame,
    lang_upper: str,
    tag: str,
    path_save: Optional[Path] = None
) -> pd.DataFrame:
    """
    Preprocess DataFrame by lemmatizing text in-process using spaCy.
    Replaces subprocess-based NLPipe for better performance.
    """
    required = {"chunk_id", "text", "lang"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    texts = df["text"].fillna("").astype(str).tolist()
    
    self._logger.info(f"Lemmatizing {len(texts)} texts for lang={lang_upper}...")
    import time
    start = time.time()
    
    lemmas = self._lemmatize_texts(texts, lang_upper)
    
    elapsed = time.time() - start
    self._logger.info(f"Lemmatization took {elapsed:.2f}s ({len(texts)/elapsed:.1f} docs/sec)")
    
    result = df.copy()
    result["lemmas"] = lemmas
    
    return result
```

#### Verification Steps

1. Compare lemmatization output with NLPipe on sample texts
2. Benchmark: Measure time for 10K documents
3. Verify no temp files created during preprocessing

---

## OPT-006

### Streaming Topic State Parsing

**File**: `src/mind/topic_modeling/polylingual_tm.py`  
**Type**: Memory  
**Effort**: Medium  
**Impact**: 50-70% peak memory reduction during training

#### Context

`save_model_info()` loads the entire `output-state.gz` file (often 500MB+) into a pandas DataFrame, causing memory spikes.

#### Current Code Location

Lines 387-392:

```python
topic_state_model = self._mallet_out_folder / "output-state.gz"
with gzip.open(topic_state_model) as fin:
    topic_state_df = pd.read_csv(
        fin, delim_whitespace=True,
        names=['docid', 'lang', 'wd_docid', 'wd_vocabid', 'wd', 'tpc'],
        header=None, skiprows=1)
```

#### Target Implementation

Replace lines 387-464 with:

```python
def _parse_topic_state_streaming(self) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Stream-parse output-state.gz to build betas matrix without loading full file.
    Returns: (betas, vocab_w2id, vocab_id2w)
    """
    topic_state_model = self._mallet_out_folder / "output-state.gz"
    
    # First pass: determine vocabulary size and topic count
    self._logger.info("First pass: counting vocabulary and topics...")
    vocab_set = set()
    max_topic = 0
    max_vocab_id = 0
    
    with gzip.open(topic_state_model, 'rt', encoding='utf-8') as fin:
        next(fin)  # Skip header
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 6:
                vocab_id = int(parts[3])
                word = parts[4]
                topic = int(parts[5])
                vocab_set.add(word)
                max_topic = max(max_topic, topic)
                max_vocab_id = max(max_vocab_id, vocab_id)
    
    num_topics = max_topic + 1
    vocab_size = max_vocab_id + 1
    
    self._logger.info(f"Found {len(vocab_set)} unique words, {num_topics} topics")
    
    # Initialize matrices
    betas = np.zeros((num_topics, vocab_size), dtype=np.float32)
    term_freq = np.zeros(vocab_size, dtype=np.int32)
    vocab_id2w = {}
    
    # Second pass: populate betas
    self._logger.info("Second pass: building word-topic counts...")
    with gzip.open(topic_state_model, 'rt', encoding='utf-8') as fin:
        next(fin)  # Skip header
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 6:
                vocab_id = int(parts[3])
                word = parts[4]
                topic = int(parts[5])
                
                betas[topic, vocab_id] += 1
                term_freq[vocab_id] += 1
                vocab_id2w[vocab_id] = word
    
    # Normalize betas
    from sklearn.preprocessing import normalize
    betas = normalize(betas, axis=1, norm='l1')
    
    # Build vocab mappings
    vocab_w2id = {w: i for i, w in vocab_id2w.items()}
    
    return betas, vocab_w2id, vocab_id2w, term_freq
```

Update `save_model_info()` to use streaming parser:

```python
def save_model_info(self):
    # ... thetas processing (lines 320-376) stays the same ...
    
    ########################################################################
    # VOCABS (using streaming parser)
    ########################################################################
    self._logger.info("Getting vocab via streaming parser...")
    betas, vocab_w2id, vocab_id2w, term_freq = self._parse_topic_state_streaming()
    
    # Save shared vocab and betas
    vocab_file = self._mallet_out_folder / "vocab.txt"
    betas_file = self._mallet_out_folder / "betas.npy"
    
    with vocab_file.open('w', encoding='utf8') as fout:
        for vocab_id in sorted(vocab_id2w.keys()):
            word = vocab_id2w[vocab_id]
            freq = int(term_freq[vocab_id])
            fout.write(f"{word}\t{freq}\n")
    
    np.save(betas_file, betas)
    
    # ... rest of method continues ...
```

#### Verification Steps

1. Compare output vocab.txt and betas.npy with original implementation
2. Monitor peak memory during training with `memory_profiler`
3. Verify betas normalization is correct (rows sum to 1)

---

## OPT-007

### GPU-Accelerated FAISS Search

**File**: `src/mind/pipeline/retriever.py`  
**Type**: GPU  
**Effort**: Medium  
**Impact**: 10x+ faster search for large indices

#### Context

FAISS indices are built on CPU by default. GPU indices offer massive speedup for batch searches.

#### Current Code Location

Lines 119-127 (index building):

```python
if n_docs > self.min_clusters:
    quantizer = faiss.IndexFlatIP(dim)
    self._ann_index = faiss.IndexIVFFlat(
        quantizer, dim, self.n_clusters_ann, faiss.METRIC_INNER_PRODUCT
    )
    self._ann_index.train(all_embeddings)
    self._ann_index.add(all_embeddings)
```

#### Target Implementation

Add GPU support with automatic fallback:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # GPU resources (lazy initialization)
    self._gpu_resources = None
    self._use_gpu = self._check_gpu_available()

def _check_gpu_available(self) -> bool:
    """Check if GPU FAISS is available."""
    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            return faiss.get_num_gpus() > 0
    except Exception:
        pass
    return False

def _get_gpu_resources(self):
    """Lazy initialize GPU resources."""
    if self._gpu_resources is None and self._use_gpu:
        import faiss
        self._gpu_resources = faiss.StandardGpuResources()
        # Limit GPU memory usage
        self._gpu_resources.setTempMemory(512 * 1024 * 1024)  # 512MB
    return self._gpu_resources

def _to_gpu(self, index):
    """Move index to GPU if available."""
    if not self._use_gpu:
        return index
    
    import faiss
    res = self._get_gpu_resources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    self._logger.info("Moved FAISS index to GPU")
    return gpu_index

def index(self, corpus_chunk: List[Dict], theta: np.ndarray = None, ...):
    # ... existing embedding code ...
    
    if n_docs > self.min_clusters:
        quantizer = faiss.IndexFlatIP(dim)
        self._ann_index = faiss.IndexIVFFlat(
            quantizer, dim, self.n_clusters_ann, faiss.METRIC_INNER_PRODUCT
        )
        self._ann_index.train(all_embeddings)
        self._ann_index.add(all_embeddings)
        self._ann_index = self._to_gpu(self._ann_index)  # Move to GPU
    # ...
```

#### Verification Steps

1. Install `faiss-gpu`: `pip install faiss-gpu`
2. Run with GPU and verify `nvidia-smi` shows FAISS memory usage
3. Benchmark search time with CPU vs GPU for 100K+ document corpus

---

## OPT-008

### Parallel Sentence Splitting in Translator

**File**: `src/mind/corpus_building/translator.py`  
**Type**: Runtime  
**Effort**: Low  
**Impact**: 2-4x speedup for sentence splitting phase

#### Context

`_split()` method iterates over DataFrame rows sequentially to split paragraphs into sentences for the NMT model.

#### Current Code Location

Lines 62-75:

```python
for _, row in df.iterrows():
    sentences = [s for s in str(row[text_col]).split(". ") if s]
    any_kept = False
    for j, s in enumerate(sentences):
        if token_len(s) < max_tokens:
            entry = {col: row.get(col, None) for col in orig_cols}
            # ...
            rows.append(entry)
```

#### Target Implementation

Replace with vectorized approach:

```python
def _split(
    self,
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    text_col: str = "text",
    lang_col: str = "lang"
) -> pd.DataFrame:
    """
    Vectorized sentence splitting with token length filtering.
    """
    tok = self.tokenizers[(src_lang, tgt_lang)]
    model_max = getattr(tok, "model_max_length", 512)
    max_tokens = int(model_max * 0.9)
    
    orig_cols = list(df.columns)
    
    # Preserve original id and text
    df = df.copy()
    df["_orig_id"] = df.get("id_preproc", df.index.astype(str))
    
    # Split text into sentences (vectorized)
    # Note: Using ". " as delimiter, with fallback for edge cases
    df["_sentences"] = df[text_col].astype(str).str.split(r"\.\s+", regex=True)
    
    # Explode to one row per sentence
    df = df.explode("_sentences", ignore_index=True)
    
    # Filter empty sentences
    df = df[df["_sentences"].str.len() > 0].copy()
    
    # Calculate token lengths in batch
    sentences = df["_sentences"].tolist()
    token_lengths = [len(tok.encode(s, truncation=False)) for s in sentences]
    df["_token_len"] = token_lengths
    
    # Filter by token length
    df = df[df["_token_len"] < max_tokens].copy()
    
    # Track dropped document IDs
    if df.empty:
        return df
    
    # Generate new id_preproc with sentence index
    df["_sent_idx"] = df.groupby("_orig_id").cumcount()
    df["id_preproc"] = df["_orig_id"] + "_" + df["_sent_idx"].astype(str)
    
    # Set text column to sentence
    df[text_col] = df["_sentences"]
    
    # Clean up temp columns
    df = df.drop(columns=["_sentences", "_orig_id", "_token_len", "_sent_idx"])
    df["index"] = range(len(df))
    
    return df
```

#### Verification Steps

1. Compare output with original `_split()` on sample data
2. Verify token length filtering works correctly
3. Benchmark time for 10K paragraph corpus

---

## OPT-009

### Parallel Topic Processing

**File**: `src/mind/pipeline/pipeline.py`  
**Type**: Runtime  
**Effort**: High  
**Impact**: Near-linear speedup with number of workers

#### Context

Topics are processed sequentially in `run_pipeline()`. Each topic's processing is independent.

#### Dependencies

- OPT-002 (batched embeddings)
- OPT-007 (GPU FAISS)

#### Current Code Location

Lines 191-198:

```python
def run_pipeline(self, topics, sample_size=None, previous_check=None, path_save="mind_results.parquet"):
    Path(path_save).mkdir(parents=True, exist_ok=True)
    
    for topic in topics:
        self._process_topic(
            topic, path_save, previous_check=previous_check, sample_size=sample_size)
```

#### Target Implementation

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import copy

def run_pipeline(
    self,
    topics,
    sample_size=None,
    previous_check=None,
    path_save="mind_results.parquet",
    parallel: bool = False,
    max_workers: int = 4
):
    Path(path_save).mkdir(parents=True, exist_ok=True)
    
    if not parallel or len(topics) <= 1:
        # Sequential processing (original behavior)
        for topic in topics:
            self._process_topic(
                topic, path_save, previous_check=previous_check, sample_size=sample_size
            )
        return
    
    # Parallel processing
    self._logger.info(f"Starting parallel processing with {max_workers} workers")
    
    # Create process-safe shared state
    manager = Manager()
    shared_results = manager.list()
    shared_discarded = manager.list()
    
    # Process topics in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                self._process_topic_isolated,
                topic,
                path_save,
                previous_check,
                sample_size
            ): topic
            for topic in topics
        }
        
        for future in as_completed(futures):
            topic = futures[future]
            try:
                topic_results, topic_discarded = future.result()
                shared_results.extend(topic_results)
                shared_discarded.extend(topic_discarded)
                self._logger.info(f"Topic {topic} completed: {len(topic_results)} results")
            except Exception as e:
                self._logger.error(f"Topic {topic} failed: {e}")
    
    # Merge results back to self
    self.results = list(shared_results)
    self.discarded = list(shared_discarded)

def _process_topic_isolated(
    self,
    topic: int,
    path_save: str,
    previous_check: str = None,
    sample_size: int = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single topic in isolation (for multiprocessing).
    Returns (results, discarded) lists.
    """
    # Create isolated state for this process
    local_results = []
    local_discarded = []
    local_questions_id = {topic: set()}
    
    # Store original references
    original_results = self.results
    original_discarded = self.discarded
    original_questions_id = self.questions_id
    
    try:
        # Swap to local state
        self.results = local_results
        self.discarded = local_discarded
        self.questions_id = local_questions_id
        
        # Process topic
        self._process_topic(topic, path_save, previous_check, sample_size)
        
        return local_results, local_discarded
    finally:
        # Restore original references
        self.results = original_results
        self.discarded = original_discarded
        self.questions_id = original_questions_id
```

#### Verification Steps

1. Run with `parallel=False` and compare output with `parallel=True`
2. Verify no data races with shared checkpoint files
3. Benchmark speedup with 2, 4, 8 workers

#### Caveats

- LLM caching may not work across processes (need Redis/shared cache)
- GPU FAISS needs careful handling in multiprocessing context
- Consider using `torch.multiprocessing` if GPU is involved

---

## OPT-010

### Async LLM Calls

**File**: `src/mind/prompter/prompter.py`  
**Type**: Runtime  
**Effort**: Medium  
**Impact**: 2-4x throughput for I/O-bound LLM calls

#### Context

All LLM calls are synchronous. For OpenAI/Ollama backends, network latency is significant.

#### Current Code Location

Lines 266-306 (prompt method).

#### Target Implementation

Add async methods:

```python
import asyncio
from typing import Optional

# Add to imports at top of file
try:
    from openai import AsyncOpenAI
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    ASYNC_OPENAI_AVAILABLE = False

class Prompter:
    def __init__(self, ...):
        # ... existing code ...
        
        # Async client initialization
        self._async_client = None
        if self.backend == "openai" and ASYNC_OPENAI_AVAILABLE:
            self._async_client = AsyncOpenAI(api_key=openai_key)
    
    async def prompt_async(
        self,
        question: str,
        system_prompt_template_path: str = None,
        use_context: bool = False,
        temperature: float = None,
        dry_run: bool = False,
    ) -> Tuple[str, Optional[Any]]:
        """Async version of prompt() for concurrent LLM calls."""
        
        if dry_run:
            return "Dry run mode is ON — no LLM calls will be made.", None
        
        if self._async_client is None:
            # Fall back to sync call wrapped in running loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.prompt(question, system_prompt_template_path, use_context, temperature, dry_run)
            )
        
        # Load system prompt
        system_prompt_template = None
        if system_prompt_template_path:
            async with aiofiles.open(system_prompt_template_path, "r") as f:
                system_prompt_template = await f.read()
        
        # Check cache first (sync is fine for cache, it's local)
        if temperature is not None:
            self.params["temperature"] = temperature
        params_tuple = tuple(sorted(self.params.items()))
        
        cache_key = (system_prompt_template, question, self.model_type, self.backend, params_tuple)
        # Check memory cache here if needed
        
        # Build messages
        messages = []
        if system_prompt_template:
            messages.append({"role": "system", "content": system_prompt_template})
        messages.append({"role": "user", "content": question})
        
        # Async OpenAI call
        response = await self._async_client.chat.completions.create(
            model=self.model_type,
            messages=messages,
            temperature=self.params.get("temperature", 0),
            max_tokens=self.params.get("max_tokens", 1000),
            seed=self.params.get("seed", 1234),
        )
        
        result = response.choices[0].message.content
        
        # Handle thinking tokens
        if "<think>" in result:
            result = result.split("</think>")[-1].strip()
        
        return result, None
    
    async def prompt_batch_async(
        self,
        questions: List[str],
        system_prompt_template_path: str = None,
        **kwargs
    ) -> List[Tuple[str, Optional[Any]]]:
        """Process multiple prompts concurrently."""
        tasks = [
            self.prompt_async(q, system_prompt_template_path, **kwargs)
            for q in questions
        ]
        return await asyncio.gather(*tasks)
```

#### Verification Steps

1. Install `aiofiles`: `pip install aiofiles`
2. Test with: `asyncio.run(prompter.prompt_async("Hello"))`
3. Benchmark: Compare sync vs async for 10 concurrent requests

---

## OPT-011

### Memory-Mapped FAISS Index Loading

**File**: `src/mind/pipeline/retriever.py`  
**Type**: Memory, Disk  
**Effort**: Medium  
**Impact**: 30-50% RAM reduction for large indices

#### Context

`faiss.read_index()` loads the entire index into RAM. Memory mapping allows OS to manage pages.

#### Current Code Location

Lines 127 and 141:

```python
faiss.write_index(self._enn_index, str(save_path_enn))
# ...
self._enn_index = faiss.read_index(str(enn_path))
```

#### Target Implementation

```python
def load_indices_from(self, indices_path: Path, mmap: bool = True):
    """Load FAISS indices, optionally with memory mapping."""
    enn_path = indices_path / "enn_index.faiss"
    ann_path = indices_path / "ann_index.faiss"
    
    io_flag = faiss.IO_FLAG_MMAP if mmap else 0
    
    if enn_path.exists():
        self._enn_index = faiss.read_index(str(enn_path), io_flag)
        self._logger.info(f"Loaded ENN index (mmap={mmap})")
    
    if ann_path.exists():
        self._ann_index = faiss.read_index(str(ann_path), io_flag)
        # Note: Cannot use mmap with IVF indices that need training state
        # Fall back to regular loading for IVF
        self._logger.info(f"Loaded ANN index (mmap={mmap})")
    
    # ... load other data ...

def __init__(self, ..., use_mmap: bool = True):
    # ... existing code ...
    self._use_mmap = use_mmap
```

Modify `index()` to save in mmap-compatible format:

```python
def save_indices_to(self, save_path: Path):
    """Save indices in format compatible with mmap loading."""
    save_path.mkdir(parents=True, exist_ok=True)
    
    save_path_enn = save_path / "enn_index.faiss"
    save_path_ann = save_path / "ann_index.faiss"
    
    # For Flat indices, standard save works with mmap
    faiss.write_index(self._enn_index, str(save_path_enn))
    
    # For IVF indices, use on-disk format for mmap support
    if hasattr(self, '_ann_index') and self._ann_index is not None:
        # IVF indices need special handling for mmap
        faiss.write_index(self._ann_index, str(save_path_ann))
```

#### Verification Steps

1. Save index, load with `mmap=True` and `mmap=False`, compare search results
2. Monitor RSS memory with `psutil.Process().memory_info().rss`
3. Verify search latency doesn't significantly increase with mmap

---

## OPT-012

### Batched NLI Entailment Checking

**File**: `src/mind/pipeline/pipeline.py`  
**Type**: GPU  
**Effort**: Medium  
**Impact**: 5-10x faster entailment checking

#### Context

`_check_entailment()` processes one text pair at a time through the NLI model.

#### Current Code Location

Lines 640-670 (approximate, in `_check_entailment` method).

#### Target Implementation

Add batch entailment method:

```python
def _batch_check_entailment(
    self,
    pairs: List[Tuple[str, str]],
    threshold: float = 0.5,
    batch_size: int = 16
) -> List[Tuple[float, float, bool]]:
    """
    Batch process entailment pairs for efficiency.
    
    Args:
        pairs: List of (premise, hypothesis) tuples
        threshold: Entailment probability threshold
        batch_size: Batch size for inference
    
    Returns:
        List of (entailment_prob, contradiction_prob, is_entailed) tuples
    """
    if not hasattr(self, '_nli_model') or self._nli_model is None:
        return [(0.0, 0.0, False)] * len(pairs)
    
    results = []
    
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]
        
        # Tokenize batch
        inputs = self._nli_tokenizer.batch_encode_plus(
            batch_pairs,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to GPU if available
        device = next(self._nli_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        # Extract entailment/contradiction probabilities
        # Assuming labels: [entailment, neutral, contradiction] or similar
        for prob in probs:
            entailment_prob = float(prob[0])  # Adjust index based on model
            contradiction_prob = float(prob[2]) if len(prob) > 2 else 0.0
            is_entailed = entailment_prob >= threshold
            results.append((entailment_prob, contradiction_prob, is_entailed))
    
    return results
```

Modify relevant method to collect and batch:

```python
def _process_question_with_batched_nli(self, question, chunk, topic, path_save):
    """Modified question processing with batched NLI."""
    # ... generate subqueries and retrieve targets ...
    
    # Collect all pairs for NLI
    nli_pairs = []
    pair_metadata = []  # Store metadata for later processing
    
    for target_chunk in target_chunks:
        # Generate answer
        a_t, _ = self._generate_answer(question, target_chunk)
        nli_pairs.append((question + " " + a_s, a_t))
        pair_metadata.append((target_chunk, a_t))
    
    # Batch NLI check
    nli_results = self._batch_check_entailment(nli_pairs)
    
    # Process results
    for (target_chunk, a_t), (ent_prob, cont_prob, is_entailed) in zip(pair_metadata, nli_results):
        # ... process each result ...
```

#### Verification Steps

1. Compare output with single-pair NLI processing
2. Verify GPU utilization increases during batch processing
3. Benchmark: Compare time for 100 pairs (batch vs sequential)

---

## Implementation Notes

### Testing Strategy

For each optimization chunk:

1. **Create feature branch**: `git checkout -b opt-XXX-description`
2. **Write unit tests first**: Cover edge cases specific to the optimization
3. **Implement change**: Follow target implementation
4. **Run regression tests**: Ensure existing tests pass
5. **Benchmark**: Compare before/after metrics
6. **Document**: Update docstrings and inline comments

### Rollback Strategy

Each optimization should be:
- Toggleable via configuration or flags where possible
- Backward compatible with existing data formats
- Independent enough to revert without cascading effects

### Dependency Management

Some optimizations require new packages:

```bash
# For GPU FAISS
pip install faiss-gpu

# For async file operations
pip install aiofiles

# For memory profiling during testing
pip install memory_profiler psutil
```

---

**End of Implementation Chunks Document**
