"""
MIND Profiling - Comparison Report Generator

Generates comparison reports between two profiling result sets.
Useful for measuring optimization impact.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def generate_comparison(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    version1: str,
    version2: str,
    output_dir: Path
) -> Path:
    """
    Generate a comparison report between two profiling result sets.
    
    Parameters:
        results1: First result set (baseline)
        results2: Second result set (optimized)
        version1: Label for first results
        version2: Label for second results
        output_dir: Directory to save report
    
    Returns:
        Path to the generated report
    """
    report_lines = []
    report_lines.append(f"# MIND Profiling Comparison Report")
    report_lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    report_lines.append(f"\n**Comparing:** {version1} (baseline) vs {version2} (optimized)")
    report_lines.append("")
    
    # Summary table
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Metric | Baseline | Optimized | Change |")
    report_lines.append("|--------|----------|-----------|--------|")
    
    improvements = []
    
    for suite in ["memory", "runtime", "io", "gpu"]:
        if suite not in results1 or suite not in results2:
            continue
        
        suite_r1 = results1[suite]
        suite_r2 = results2[suite]
        
        if isinstance(suite_r1, dict) and "error" not in suite_r1:
            for size in suite_r1:
                if size not in suite_r2:
                    continue
                
                size_r1 = suite_r1[size]
                size_r2 = suite_r2[size]
                
                if not isinstance(size_r1, dict) or not isinstance(size_r2, dict):
                    continue
                
                for test, metrics1 in size_r1.items():
                    if test not in size_r2:
                        continue
                    
                    metrics2 = size_r2[test]
                    
                    if not isinstance(metrics1, dict) or not isinstance(metrics2, dict):
                        continue
                    
                    # Compare key metrics
                    for metric_key in ["runtime_seconds", "peak_memory_mb", "docs_per_second", "queries_per_second"]:
                        if metric_key in metrics1 and metric_key in metrics2:
                            v1 = metrics1[metric_key]
                            v2 = metrics2[metric_key]
                            
                            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 > 0:
                                if metric_key in ["runtime_seconds", "peak_memory_mb"]:
                                    # Lower is better
                                    change = (v1 - v2) / v1 * 100
                                    change_str = f"{change:+.1f}%" if change >= 0 else f"{change:.1f}%"
                                    emoji = "✅" if change > 5 else ("⚠️" if change > -5 else "❌")
                                else:
                                    # Higher is better
                                    change = (v2 - v1) / v1 * 100
                                    change_str = f"{change:+.1f}%" if change >= 0 else f"{change:.1f}%"
                                    emoji = "✅" if change > 5 else ("⚠️" if change > -5 else "❌")
                                
                                report_lines.append(
                                    f"| {suite}/{test}/{metric_key} | {v1:.2f} | {v2:.2f} | {emoji} {change_str} |"
                                )
                                
                                if change > 5:
                                    improvements.append({
                                        "suite": suite,
                                        "test": test,
                                        "metric": metric_key,
                                        "change": change,
                                    })
    
    report_lines.append("")
    
    # Key improvements section
    if improvements:
        report_lines.append("## Key Improvements")
        report_lines.append("")
        
        improvements.sort(key=lambda x: x["change"], reverse=True)
        
        for imp in improvements[:10]:
            report_lines.append(
                f"- **{imp['suite']}/{imp['test']}**: {imp['metric']} improved by {imp['change']:.1f}%"
            )
        
        report_lines.append("")
    
    # Detailed comparison sections
    for suite in ["memory", "runtime", "io", "gpu"]:
        if suite not in results1 or suite not in results2:
            continue
        
        report_lines.append(f"## {suite.upper()} Detailed Comparison")
        report_lines.append("")
        report_lines.append("```json")
        
        comparison = {
            "baseline": results1.get(suite, {}),
            "optimized": results2.get(suite, {}),
        }
        report_lines.append(json.dumps(comparison, indent=2, default=str))
        report_lines.append("```")
        report_lines.append("")
    
    # Save report
    report_content = "\n".join(report_lines)
    report_path = output_dir / f"comparison_{version1}_vs_{version2}.md"
    
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Also print summary to console
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if improvements:
        print(f"\n✅ {len(improvements)} metrics improved by >5%:")
        for imp in improvements[:5]:
            print(f"   - {imp['suite']}/{imp['test']}: {imp['change']:.1f}%")
    else:
        print("\nNo significant improvements detected.")
    
    return report_path


def calculate_aggregate_improvements(results1: Dict, results2: Dict) -> Dict:
    """
    Calculate aggregate improvement metrics across all suites.
    
    Returns dict with:
        - total_runtime_reduction: Percentage runtime reduction
        - total_memory_reduction: Percentage memory reduction
        - total_throughput_increase: Percentage throughput increase
    """
    runtime_ratios = []
    memory_ratios = []
    throughput_ratios = []
    
    for suite in ["memory", "runtime", "io", "gpu"]:
        if suite not in results1 or suite not in results2:
            continue
        
        # Walk through all nested metrics and collect ratios
        def extract_metrics(d1, d2, path=""):
            if not isinstance(d1, dict) or not isinstance(d2, dict):
                return
            
            for key in set(d1.keys()) & set(d2.keys()):
                v1, v2 = d1[key], d2[key]
                
                if isinstance(v1, dict) and isinstance(v2, dict):
                    extract_metrics(v1, v2, f"{path}/{key}")
                elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v1 > 0 and v2 > 0:
                        if "runtime" in key or "seconds" in key:
                            runtime_ratios.append(v2 / v1)
                        elif "memory" in key:
                            memory_ratios.append(v2 / v1)
                        elif "per_second" in key or "throughput" in key:
                            throughput_ratios.append(v2 / v1)
        
        extract_metrics(results1[suite], results2[suite])
    
    def avg_reduction(ratios):
        if not ratios:
            return 0
        return (1 - sum(ratios) / len(ratios)) * 100
    
    def avg_increase(ratios):
        if not ratios:
            return 0
        return (sum(ratios) / len(ratios) - 1) * 100
    
    return {
        "runtime_reduction_pct": avg_reduction(runtime_ratios),
        "memory_reduction_pct": avg_reduction(memory_ratios),
        "throughput_increase_pct": avg_increase(throughput_ratios),
        "num_runtime_metrics": len(runtime_ratios),
        "num_memory_metrics": len(memory_ratios),
        "num_throughput_metrics": len(throughput_ratios),
    }
