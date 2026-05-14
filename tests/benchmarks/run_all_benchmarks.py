#!/usr/bin/env python3
"""
Unified Benchmark Runner – executes all 6 benchmarks sequentially.
"""
import sys, time, traceback

benchmarks = [
    ("A – Anomaly Detection",     "test_anomaly_detection",    "run_anomaly_detection_benchmark"),
    ("B – DP Impact",             "test_dp_impact",            "run_dp_benchmark"),
    ("C – FL Overhead",           "test_fl_overhead",          "run_overhead_benchmark"),
    ("D – RCA Accuracy",          "test_rca_accuracy",         "run_rca_benchmark"),
    ("E – Adaptive Threshold",    "test_adaptive_threshold",   "run_adaptive_threshold_benchmark"),
    ("F – Scalability",           "test_scalability",          "run_scalability_benchmark"),
]

def main():
    results = {}
    total_start = time.perf_counter()

    for name, module_name, func_name in benchmarks:
        print(f"\n{'='*70}")
        print(f"  BENCHMARK {name}")
        print(f"{'='*70}")
        t0 = time.perf_counter()
        try:
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            func()
            elapsed = time.perf_counter() - t0
            results[name] = f"OK ({elapsed:.1f}s)"
            print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results[name] = f"FAILED ({elapsed:.1f}s): {e}"
            print(f"\n  ✗ {name} FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()

    total = time.perf_counter() - total_start
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (total: {total:.1f}s)")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"    {name}: {status}")
    print()

if __name__ == "__main__":
    main()
