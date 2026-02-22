"""
MIND Profiling Test Suite

This package provides comprehensive profiling tools for measuring
and benchmarking MIND system performance across:
- Memory (RAM) usage
- Disk I/O operations
- GPU utilization
- Runtime performance

Usage:
    from aux_scripts.profiling import run_profiling
    # or run directly:
    python -m aux_scripts.profiling.run_profiling --all
"""

from aux_scripts.profiling.memory_profiler import MemoryProfiler
from aux_scripts.profiling.runtime_profiler import RuntimeProfiler
from aux_scripts.profiling.io_profiler import IOProfiler
from aux_scripts.profiling.gpu_profiler import GPUProfiler
from aux_scripts.profiling.data_generator import BenchmarkDataGenerator
from aux_scripts.profiling.comparison_report import generate_comparison

__all__ = [
    "MemoryProfiler",
    "RuntimeProfiler",
    "IOProfiler",
    "GPUProfiler",
    "BenchmarkDataGenerator",
    "generate_comparison",
]
