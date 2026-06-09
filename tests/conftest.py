import sys
import os
import pytest
import numpy as np
import torch

# Mock prometheus_client to avoid global metric registration collisions
class DummyMetric:
    def __init__(self, *args, **kwargs):
        pass
    def labels(self, *args, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        pass
    def dec(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass
    def observe(self, *args, **kwargs):
        pass

class DummyPrometheusClient:
    Counter = DummyMetric
    Gauge = DummyMetric
    Histogram = DummyMetric
    Summary = DummyMetric
    @staticmethod
    def start_http_server(*args, **kwargs):
        pass

sys.modules['prometheus_client'] = DummyPrometheusClient

# Ensure the `src/` package tree is importable regardless of working directory
SRC_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))


# Minimal in-memory configuration (no YAML file required)
MINIMAL_CONFIG: dict = {
    "edge": {
        "model": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "sequence_length": 20,
        },
        "detection": {
            "initial_threshold": 3.0,
            "threshold_percentile": 95,
            "min_anomaly_duration": 3,
            "smoothing_window": 5,
        },
        "inference": {
            "batch_size": 8,
            "quantization": False,
        },
        "data": {
            "window_size": 20,
            "stride": 1,
        },
    },
    "federated": {
        "coordinator": {
            "host": "127.0.0.1",
            "port": 50051,
            "max_workers": 2,
            "num_rounds": 5,
            "min_clients_per_round": 2,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "aggregation_strategy": "fedavg",
            "staleness_tolerance": 2,
            "round_timeout_seconds": 5,
            "checkpoint_dir": "tests/checkpoints",
        },
        "client": {
            "epochs_per_round": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "max_retries": 1,
            "retry_delay_seconds": 0,
            "gradient_compression": {
                "enabled": True,
                "method": "topk",
                "compression_ratio": 0.1,
            },
            "payload_compression": {
                "enabled": True,
                "cpu_usage_percent": {
                    "high_cpu_threshold": 85.0,
                    "moderate_cpu_threshold": 60.0,
                },
                "zstd_level": {"moderate": 1, "idle": 3},
            },
            "differential_privacy": {
                "enabled": False,
                "noise_multiplier": 1.0,
                "max_grad_norm": 1.0,
            },
        },
        "poisoning_detection": {
            "enabled": True,
            "zscore_threshold": 3.0,
            "autoencoder_threshold": 0.8,
            "validation_samples": 100,
        },
    },
    "tracing": {
        "graph": {
            "update_interval": 60,
            "edge_weight_decay": 0.95,
            "min_edge_weight": 0.01,
            "snapshot_interval": 300,
        },
    },
    "root_cause": {
        "pagerank": {
            "alpha": 0.85,
            "max_iterations": 100,
            "tolerance": 1.0e-6,
            "personalization_weight": 0.7,
        },
        "classification": {
            "propagation_threshold": 0.3,
            "max_hops": 5,
            "min_impact_score": 0.1,
        },
        "explanation": {
            "max_chain_length": 10,
            "confidence_threshold": 0.6,
        },
    },
    "thresholding": {
        "slo": {
            "latency_p95_ms": 200,
            "latency_p99_ms": 500,
            "error_rate_threshold": 0.01,
            "collection_interval": 60,
        },
        "rl_tuner": {
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "min_epsilon": 0.01,
            "state_features": [
                "current_threshold",
                "false_positive_rate",
                "false_negative_rate",
                "slo_violation_rate",
            ],
            "actions": [-0.5, -0.2, 0.0, 0.2, 0.5],
            "reward": {
                "precision_weight": 0.3,
                "recall_weight": 0.3,
                "slo_compliance_weight": 0.4,
                "false_positive_penalty": -1.0,
                "false_negative_penalty": -2.0,
                "slo_violation_penalty": -3.0,
            },
        },
    },
}


# Patch global singletons so tests never reach external services
@pytest.fixture(autouse=True)
def _patch_global_config(monkeypatch):
    """
    Replace the global Config singleton with MINIMAL_CONFIG for every test.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    from utils.config import Config
    import utils.config as config_mod

    monkeypatch.setattr(config_mod, "_config_instance", Config(MINIMAL_CONFIG))


@pytest.fixture(autouse=True)
def _clear_cache():
    """
    Clear global fallback in-memory cache and bypass Redis check to prevent 
    connection timeouts.
    """
    import utils.cache
    utils.cache._fallback_store.clear()
    utils.cache._redis_checked = True
    utils.cache._redis_available = False


# Reusable fixtures
@pytest.fixture
def config():
    """
    Return a plain dict copy of the minimal test configuration.

    Returns:
        dict: A copy of the minimal test configuration settings.
    """
    from copy import deepcopy
    return deepcopy(MINIMAL_CONFIG)


@pytest.fixture
def lstm_model(config):
    """
    Create a small LSTM-AE model for testing.

    Args:
        config: Test configuration fixture dict

    Returns:
        LSTMAnomalyDetector: A small initialized LSTM-AE model instance.
    """
    from edge.models import LSTMAnomalyDetector

    mc = config["edge"]["model"]
    model = LSTMAnomalyDetector(
        input_size=mc["input_size"],
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    )
    return model


@pytest.fixture
def autoencoder():
    """
    Create a small AutoEncoder model for testing.

    Returns:
        AutoEncoder: A small initialized AutoEncoder model instance.
    """
    from edge.models import AutoEncoder

    return AutoEncoder(input_dim=128, encoding_dim=16, hidden_dims=[64, 32])


@pytest.fixture
def normal_data(config):
    """
    Generate synthetic normal time-series data (low variance sine waves).

    Args:
        config: Test configuration fixture dict

    Returns:
        np.ndarray: A float32 array representing synthetic normal telemetry.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = config["edge"]["model"]["input_size"]
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.column_stack(
        [np.sin(t + i) + np.random.normal(0, 0.05, n_samples) for i in range(n_features)]
    )
    return data.astype(np.float32)


@pytest.fixture
def anomalous_data(config):
    """
    Generate synthetic data with an injected anomaly in the second half.

    Args:
        config: Test configuration fixture dict

    Returns:
        np.ndarray: A float32 array representing synthetic telemetry with anomalies.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = config["edge"]["model"]["input_size"]
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.column_stack(
        [np.sin(t + i) + np.random.normal(0, 0.05, n_samples) for i in range(n_features)]
    )
    # Inject a large spike in the last 20 samples
    data[-20:] += np.random.normal(10.0, 2.0, (20, n_features))
    return data.astype(np.float32)


@pytest.fixture
def causal_graph(config):
    """
    Build a small service dependency graph for RCA tests.

    Args:
        config: Test configuration fixture dict

    Returns:
        CausalGraph: A small initialized dependency graph of simulated services.
    """
    from tracing.causal_graph import CausalGraph

    cg = CausalGraph(config)
    for svc in ["gateway", "svc-a", "svc-b", "svc-c", "svc-d"]:
        cg.add_service(svc)
    cg.add_dependency("gateway", "svc-a", call_count=100, latency=5.0)
    cg.add_dependency("svc-a", "svc-c", call_count=80, latency=3.0)
    cg.add_dependency("svc-a", "svc-b", call_count=60, latency=4.0)
    cg.add_dependency("svc-b", "svc-d", call_count=40, latency=2.0)
    return cg

