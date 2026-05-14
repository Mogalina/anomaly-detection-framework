"""
Benchmark F – System Scalability (Thesis §11.6)
================================================
Measures FedAvg aggregation time and memory across 5–1000 edge nodes
using the best LSTM architecture. Includes Alibaba-inspired large-scale test.
"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import psutil
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

_models = _import_from_file('edge_models', os.path.join(ROOT, 'src', 'edge', 'models.py'))
_logger = _import_from_file('thesis_logger', os.path.join(ROOT, 'tests', 'utils', 'thesis_logger.py'))
LSTMAnomalyDetector = _models.LSTMAnomalyDetector
ThesisLogger = _logger.ThesisLogger

INPUT_SIZE  = 38
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
np.random.seed(42); torch.manual_seed(42)


def _fedavg(updates, weights):
    """FedAvg aggregation."""
    result = {}
    for key in updates[0]:
        stacked = torch.stack([u[key].float() for u in updates])
        w = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)
        while w.ndim < stacked.ndim:
            w = w.unsqueeze(-1)
        result[key] = (stacked * w).sum(dim=0)
    return result


def run_scalability_benchmark():
    logger = ThesisLogger("benchmark_F_scalability")

    process = psutil.Process()
    node_counts = [5, 10, 25, 50, 100, 200, 500, 1000]

    for N in node_counts:
        print(f"  [F] Aggregating {N} nodes ...")
        ref_model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
        base_state = ref_model.state_dict()

        # Generate simulated client updates
        updates = []
        for _ in range(N):
            update = {}
            for k, v in base_state.items():
                delta = torch.randn_like(v) * 0.01
                update[k] = v + delta
            updates.append(update)
        weights = [1.0 / N] * N

        mem_before = process.memory_info().rss / (1024 * 1024)
        cpu_before = psutil.cpu_percent(interval=None)

        # Aggregation
        t0 = time.perf_counter()
        aggregated = _fedavg(updates, weights)
        agg_ms = (time.perf_counter() - t0) * 1000

        mem_after = process.memory_info().rss / (1024 * 1024)
        cpu_after = psutil.cpu_percent(interval=None)

        # Verify aggregation
        ref_model.load_state_dict(aggregated)

        logger.log_metric(f"scale_{N}", {
            'num_nodes': N,
            'aggregation_time_ms': round(agg_ms, 4),
            'memory_before_mb': round(mem_before, 2),
            'memory_after_mb': round(mem_after, 2),
            'memory_delta_mb': round(mem_after - mem_before, 2),
            'cpu_usage_percent': round(cpu_after, 2),
            'params_per_model': sum(p.numel() for p in ref_model.parameters()),
        })
        print(f"      {N} nodes: agg={agg_ms:.1f}ms  mem_delta={mem_after-mem_before:.1f}MB")

    print("  [F] Complete.")


if __name__ == "__main__":
    run_scalability_benchmark()