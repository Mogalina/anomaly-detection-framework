"""
Benchmark C – Federated Learning Communication Overhead (Thesis §11.3)
======================================================================
Measures payload size and processing time across serialisation/compression
combinations, adaptive compression under CPU loads, and per-round FL
latency breakdown.
"""
import os, sys, time, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

def _import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

_models = _import_from_file('edge_models', os.path.join(ROOT, 'src', 'edge', 'models.py'))
_logger = _import_from_file('thesis_logger', os.path.join(ROOT, 'tests', 'utils', 'thesis_logger.py'))
LSTMAnomalyDetector = _models.LSTMAnomalyDetector
ThesisLogger = _logger.ThesisLogger

_compression = _import_from_file('compression', os.path.join(ROOT, 'src', 'utils', 'compression.py'))
compress = _compression.compress
decompress = _compression.decompress
CompressionType = _compression.CompressionType
serialize_state_dict = _compression.serialize_state_dict
choose_algorithm = _compression.choose_algorithm

INPUT_SIZE  = 38
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
SEQ_LEN     = 50
np.random.seed(42); torch.manual_seed(42)


def _protobuf_serialize(state_dict):
    from google.protobuf.struct_pb2 import Struct
    s = Struct()
    for key, tensor in state_dict.items():
        s.fields[key].string_value = json.dumps(tensor.numpy().tolist())
    return s.SerializeToString()


def run_overhead_benchmark():
    logger = ThesisLogger("benchmark_C_fl_overhead")

    model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    state_dict = model.state_dict()

    # ── serialisation ──────────────────────────────────────────────────
    raw_torch = serialize_state_dict(state_dict)
    raw_json  = json.dumps({k: v.numpy().tolist() for k, v in state_dict.items()}).encode()
    raw_proto = _protobuf_serialize(state_dict)

    serializations = {'TorchSave': raw_torch, 'JSON': raw_json, 'Protobuf': raw_proto}
    compression_algos = {
        'None':    (CompressionType.NONE, {}),
        'LZ4':     (CompressionType.LZ4,  {}),
        'Zstd_L1': (CompressionType.ZSTD, {'cpu_usage_percent': 70}),
        'Zstd_L3': (CompressionType.ZSTD, {'cpu_usage_percent': 20}),
    }

    # ── Part 1: Payload size & compression matrix ─────────────────────
    print("  [C] Payload size matrix ...")
    for s_name, raw in serializations.items():
        for c_name, (algo, kw) in compression_algos.items():
            t0 = time.perf_counter()
            compressed = compress(raw, algo, **kw)
            comp_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            decompress(compressed, algo)
            decomp_ms = (time.perf_counter() - t0) * 1000
            orig_kb = len(raw) / 1024
            pay_kb  = len(compressed) / 1024
            logger.log_metric("payload_eval", {
                'serialization': s_name, 'compression': c_name,
                'original_size_kb': round(orig_kb, 2),
                'payload_size_kb':  round(pay_kb, 2),
                'compression_ratio': round(pay_kb / orig_kb, 4) if orig_kb else 1.0,
                'compress_time_ms':  round(comp_ms, 4),
                'decompress_time_ms': round(decomp_ms, 4),
            })

    # ── Part 2: Adaptive compression under CPU loads ──────────────────
    print("  [C] Adaptive compression ...")
    cpu_loads = [10, 30, 50, 70, 85, 95]
    raw = raw_torch
    for cpu in cpu_loads:
        algo = choose_algorithm(cpu)
        t0 = time.perf_counter()
        compressed = compress(raw, algo, cpu_usage_percent=cpu)
        comp_ms = (time.perf_counter() - t0) * 1000
        logger.log_metric("adaptive_compression", {
            'cpu_load_percent': cpu, 'algorithm_chosen': algo.name,
            'original_size_kb': round(len(raw) / 1024, 2),
            'payload_size_kb':  round(len(compressed) / 1024, 2),
            'compress_time_ms': round(comp_ms, 4),
        })

    # ── Part 3: Per-round FL latency breakdown ────────────────────────
    print("  [C] FL round latency breakdown ...")
    train_data = np.random.randn(200, SEQ_LEN, INPUT_SIZE).astype(np.float32)
    x_train = torch.FloatTensor(train_data)
    crit = torch.nn.MSELoss()
    num_clients = 5

    for rnd in range(10):
        client_payloads = []
        t0 = time.perf_counter()
        for _ in range(num_clients):
            m = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
            opt = torch.optim.Adam(m.parameters(), lr=0.001)
            m.train(); opt.zero_grad()
            loss = crit(m(x_train), x_train); loss.backward(); opt.step()
            client_payloads.append(m.state_dict())
        train_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        compressed_payloads = [compress(serialize_state_dict(sd), CompressionType.ZSTD)
                               for sd in client_payloads]
        compress_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        agg = {}
        for sd in client_payloads:
            for k, v in sd.items():
                agg[k] = agg.get(k, torch.zeros_like(v)) + v / num_clients
        agg_ms = (time.perf_counter() - t0) * 1000

        total_payload_kb = sum(len(p) for p in compressed_payloads) / 1024
        logger.log_metric(f"fl_round_{rnd}", {
            'round': rnd, 'num_clients': num_clients,
            'local_train_ms': round(train_ms, 2),
            'compression_ms': round(compress_ms, 2),
            'aggregation_ms': round(agg_ms, 2),
            'total_round_ms': round(train_ms + compress_ms + agg_ms, 2),
            'total_payload_kb': round(total_payload_kb, 2),
        })

    print("  [C] Complete.")


if __name__ == "__main__":
    run_overhead_benchmark()