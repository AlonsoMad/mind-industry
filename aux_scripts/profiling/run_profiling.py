#!/usr/bin/env python3
"""
MIND Project Profiling Test Suite - Main Orchestrator

This script orchestrates all profiling tests for the MIND system.
It measures: Memory (RAM), Disk I/O, GPU utilization, and Runtime.

Usage:
    python run_profiling.py --all                  # Run all benchmarks
    python run_profiling.py --suite memory         # Run memory benchmarks only
    python run_profiling.py --compare v1.0 v1.1    # Compare two result sets

Results are saved to: aux_scripts/profiling/results/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aux_scripts.profiling.memory_profiler import MemoryProfiler
from aux_scripts.profiling.runtime_profiler import RuntimeProfiler
from aux_scripts.profiling.io_profiler import IOProfiler
from aux_scripts.profiling.gpu_profiler import GPUProfiler
from aux_scripts.profiling.data_generator import BenchmarkDataGenerator


RESULTS_DIR = Path(__file__).parent / "profiling" / "results"
BENCHMARK_DATA_DIR = Path(__file__).parent / "profiling" / "benchmark_data"


def setup_directories():
    """Create necessary directories for profiling."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_benchmark_data(sizes: List[int] = None, force: bool = False):
    """Generate synthetic benchmark data if not present."""
    sizes = sizes or [1000, 10000, 50000]
    generator = BenchmarkDataGenerator(BENCHMARK_DATA_DIR)
    
    for size in sizes:
        corpus_path = BENCHMARK_DATA_DIR / f"corpus_{size}.parquet"
        if not corpus_path.exists() or force:
            print(f"Generating benchmark corpus with {size} documents...")
            generator.generate_corpus(size, corpus_path)
        else:
            print(f"Benchmark corpus {corpus_path.name} already exists.")
    
    return [BENCHMARK_DATA_DIR / f"corpus_{s}.parquet" for s in sizes]


def run_memory_suite(corpus_paths: List[Path], results_dir: Path) -> Dict:
    """Run memory profiling tests."""
    print("\n" + "="*60)
    print("MEMORY PROFILING SUITE")
    print("="*60)
    
    profiler = MemoryProfiler(results_dir)
    results = {}
    
    for corpus_path in corpus_paths:
        size = corpus_path.stem.split("_")[-1]
        print(f"\n--- Testing with {size} documents ---")
        
        results[size] = {
            "segmenter": profiler.profile_segmenter(corpus_path),
            "corpus_loading": profiler.profile_corpus_loading(corpus_path),
            "retriever_indexing": profiler.profile_retriever_indexing(corpus_path),
            "translator": profiler.profile_translator(corpus_path),
        }
    
    return results


def run_runtime_suite(corpus_paths: List[Path], results_dir: Path) -> Dict:
    """Run runtime profiling tests."""
    print("\n" + "="*60)
    print("RUNTIME PROFILING SUITE")
    print("="*60)
    
    profiler = RuntimeProfiler(results_dir)
    results = {}
    
    for corpus_path in corpus_paths:
        size = corpus_path.stem.split("_")[-1]
        print(f"\n--- Testing with {size} documents ---")
        
        results[size] = {
            "segmenter": profiler.profile_segmenter(corpus_path),
            "data_preparer": profiler.profile_data_preparer(corpus_path),
            "retriever": profiler.profile_retriever(corpus_path),
            "embedding": profiler.profile_embedding(corpus_path),
        }
    
    return results


def run_io_suite(corpus_paths: List[Path], results_dir: Path) -> Dict:
    """Run I/O profiling tests."""
    print("\n" + "="*60)
    print("I/O PROFILING SUITE")
    print("="*60)
    
    profiler = IOProfiler(results_dir)
    results = {}
    
    for corpus_path in corpus_paths:
        size = corpus_path.stem.split("_")[-1]
        print(f"\n--- Testing with {size} documents ---")
        
        results[size] = {
            "parquet_read": profiler.profile_parquet_read(corpus_path),
            "parquet_write": profiler.profile_parquet_write(corpus_path),
            "checkpoint_write": profiler.profile_checkpoint_write(corpus_path),
            "faiss_index_io": profiler.profile_faiss_io(corpus_path),
        }
    
    return results


def run_gpu_suite(corpus_paths: List[Path], results_dir: Path) -> Dict:
    """Run GPU profiling tests."""
    print("\n" + "="*60)
    print("GPU PROFILING SUITE")
    print("="*60)
    
    profiler = GPUProfiler(results_dir)
    
    if not profiler.gpu_available:
        print("WARNING: No GPU detected. Skipping GPU benchmarks.")
        return {"error": "No GPU available"}
    
    results = {}
    
    for corpus_path in corpus_paths:
        size = corpus_path.stem.split("_")[-1]
        print(f"\n--- Testing with {size} documents ---")
        
        results[size] = {
            "embedding_batch_sizes": profiler.profile_embedding_batch_sizes(corpus_path),
            "faiss_gpu_vs_cpu": profiler.profile_faiss_gpu_vs_cpu(corpus_path),
            "nli_batch_sizes": profiler.profile_nli_batch_sizes(),
        }
    
    return results


def save_results(results: Dict, version: str = None):
    """Save profiling results to JSON."""
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"profiling_results_{version}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    return results_file


def compare_results(version1: str, version2: str):
    """Compare two profiling result sets."""
    file1 = RESULTS_DIR / f"profiling_results_{version1}.json"
    file2 = RESULTS_DIR / f"profiling_results_{version2}.json"
    
    if not file1.exists() or not file2.exists():
        print(f"Error: Result files not found.")
        return
    
    with open(file1) as f:
        results1 = json.load(f)
    with open(file2) as f:
        results2 = json.load(f)
    
    print(f"\nComparing {version1} vs {version2}")
    print("="*60)
    
    # Generate comparison report
    from aux_scripts.profiling.comparison_report import generate_comparison
    generate_comparison(results1, results2, version1, version2, RESULTS_DIR)


def print_summary(results: Dict):
    """Print a summary of profiling results."""
    print("\n" + "="*60)
    print("PROFILING SUMMARY")
    print("="*60)
    
    for suite, suite_results in results.items():
        if suite == "metadata":
            continue
        print(f"\n{suite.upper()}")
        print("-" * 40)
        
        if isinstance(suite_results, dict) and "error" not in suite_results:
            for size, size_results in suite_results.items():
                print(f"  Corpus size: {size}")
                if isinstance(size_results, dict):
                    for test, metrics in size_results.items():
                        if isinstance(metrics, dict):
                            runtime = metrics.get("runtime_seconds", "N/A")
                            peak_mem = metrics.get("peak_memory_mb", "N/A")
                            # Handle non-numeric values gracefully
                            if isinstance(runtime, (int, float)) and isinstance(peak_mem, (int, float)):
                                print(f"    {test}: {runtime:.2f}s, {peak_mem:.1f}MB peak")
                            elif isinstance(runtime, (int, float)):
                                print(f"    {test}: {runtime:.2f}s, {peak_mem} peak")
                            elif isinstance(peak_mem, (int, float)):
                                print(f"    {test}: {runtime}, {peak_mem:.1f}MB peak")
                            else:
                                print(f"    {test}: {runtime}, {peak_mem}")


def main():
    parser = argparse.ArgumentParser(
        description="MIND Project Profiling Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python run_profiling.py --all
    
    # Run only memory benchmarks
    python run_profiling.py --suite memory
    
    # Run with custom corpus sizes
    python run_profiling.py --all --sizes 1000 5000 20000
    
    # Compare two result sets
    python run_profiling.py --compare v1.0 v1.1
    
    # Regenerate benchmark data
    python run_profiling.py --generate-data --sizes 1000 10000
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all profiling suites")
    parser.add_argument("--suite", choices=["memory", "runtime", "io", "gpu"],
                       nargs="+", help="Run specific suite(s)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000],
                       help="Corpus sizes for benchmarking")
    parser.add_argument("--version", type=str, help="Version label for results")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("V1", "V2"),
                       help="Compare two result versions")
    parser.add_argument("--generate-data", action="store_true",
                       help="Force regenerate benchmark data")
    parser.add_argument("--list-results", action="store_true",
                       help="List available result files")
    
    args = parser.parse_args()
    
    setup_directories()
    
    if args.list_results:
        print("Available result files:")
        for f in sorted(RESULTS_DIR.glob("profiling_results_*.json")):
            print(f"  {f.name}")
        return
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    if not args.all and not args.suite:
        parser.print_help()
        return
    
    # Generate benchmark data
    corpus_paths = generate_benchmark_data(args.sizes, args.generate_data)
    
    # Determine which suites to run
    suites_to_run = []
    if args.all:
        suites_to_run = ["memory", "runtime", "io", "gpu"]
    elif args.suite:
        suites_to_run = args.suite
    
    # Run profiling
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": args.version or "dev",
            "corpus_sizes": args.sizes,
            "suites": suites_to_run,
        }
    }
    
    if "memory" in suites_to_run:
        all_results["memory"] = run_memory_suite(corpus_paths, RESULTS_DIR)
    
    if "runtime" in suites_to_run:
        all_results["runtime"] = run_runtime_suite(corpus_paths, RESULTS_DIR)
    
    if "io" in suites_to_run:
        all_results["io"] = run_io_suite(corpus_paths, RESULTS_DIR)
    
    if "gpu" in suites_to_run:
        all_results["gpu"] = run_gpu_suite(corpus_paths, RESULTS_DIR)
    
    # Save and summarize
    save_results(all_results, args.version)
    print_summary(all_results)


if __name__ == "__main__":
    main()
