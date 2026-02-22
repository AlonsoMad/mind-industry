# MIND Profiling Test Suite

> Performance benchmarking for the MIND system

## Overview

This profiling suite measures performance across four key metrics:
- **Memory (RAM)**: Peak usage, allocation patterns
- **Runtime**: Execution speed, throughput
- **Disk I/O**: Read/write performance, checkpoint costs
- **GPU**: Utilization, batch size optimization

## Quick Start

```bash
# Run all benchmarks with default corpus sizes (1K, 10K documents)
python aux_scripts/profiling/run_profiling.py --all

# Run specific suites
python aux_scripts/profiling/run_profiling.py --suite memory runtime

# Use custom corpus sizes
python aux_scripts/profiling/run_profiling.py --all --sizes 1000 5000 20000

# Label results with version
python aux_scripts/profiling/run_profiling.py --all --version v1.0.0

# Compare two result sets
python aux_scripts/profiling/run_profiling.py --compare v1.0.0 v1.1.0
```

## Test Suites

### Memory Suite (`memory_profiler.py`)
| Test | Description |
|------|-------------|
| `segmenter` | Measures RAM during document segmentation |
| `corpus_loading` | Compares full vs chunked DataFrame loading |
| `retriever_indexing` | Profiles embedding + FAISS index memory |
| `translator` | Tests sentence splitting approaches |

### Runtime Suite (`runtime_profiler.py`)
| Test | Description |
|------|-------------|
| `segmenter` | Documents/second throughput |
| `data_preparer` | spaCy lemmatization speed |
| `retriever` | Single vs batched query performance |
| `embedding` | Optimal batch size identification |

### I/O Suite (`io_profiler.py`)
| Test | Description |
|------|-------------|
| `parquet_read` | Full vs chunked Parquet reading |
| `parquet_write` | Compression comparison (gzip, zstd, snappy) |
| `checkpoint_write` | Sync vs async checkpoint writes |
| `faiss_io` | Index save/load, mmap support |

### GPU Suite (`gpu_profiler.py`)
| Test | Description |
|------|-------------|
| `embedding_batch_sizes` | Optimal GPU batch size |
| `faiss_gpu_vs_cpu` | GPU acceleration speedup |
| `nli_batch_sizes` | NLI model batching |
| `mixed_precision` | FP16 vs FP32 performance |

## Output Structure

```
aux_scripts/profiling/
├── run_profiling.py          # Main orchestrator
├── benchmark_data/           # Generated test corpora
│   ├── corpus_1000.parquet
│   └── corpus_10000.parquet
└── results/                  # Profiling results
    ├── profiling_results_v1.0.0.json
    └── comparison_v1.0.0_vs_v1.1.0.md
```

## Interpreting Results

Results are saved as JSON with this structure:

```json
{
  "metadata": {
    "timestamp": "2026-02-01T17:00:00",
    "version": "v1.0.0",
    "corpus_sizes": [1000, 10000]
  },
  "memory": {
    "1000": {
      "segmenter": {
        "peak_memory_mb": 245.3,
        "allocated_mb": 180.2
      }
    }
  }
}
```

## Workflow for Optimization Validation

1. **Establish baseline**: Run with current code
   ```bash
   python run_profiling.py --all --version baseline
   ```

2. **Implement optimization**: Apply changes from `optimization-implementation-chunks.md`

3. **Measure improvement**: Run again with new version
   ```bash
   python run_profiling.py --all --version opt-001
   ```

4. **Compare results**:
   ```bash
   python run_profiling.py --compare baseline opt-001
   ```

## Requirements

```bash
pip install psutil pandas pyarrow faiss-cpu sentence-transformers
# For GPU tests:
pip install faiss-gpu torch
```

## Extending the Suite

Add new tests by creating a method in the appropriate profiler class:

```python
# In memory_profiler.py
def profile_new_component(self, corpus_path: Path) -> Dict:
    def run_test():
        # Your test code here
        return result
    
    print("  Profiling: New component")
    result = self._run_with_memory_tracking(run_test)
    result["test"] = "new_component"
    return result
```

Then register it in the suite runner in `run_profiling.py`.
