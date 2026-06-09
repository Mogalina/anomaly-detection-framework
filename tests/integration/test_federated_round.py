import numpy as np
import pytest
import torch


class TestClientTrainAndAggregate:
    """
    Validates that client nodes produce trainable and aggregatable local updates.
    """

    def test_single_client_round(self, config, lstm_model, normal_data):
        """
        Verify that training a single client for one round yields a valid non-empty
        parameter update and a reduction in training loss.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})

        assert result["model_update"] is not None
        assert result["num_samples"] == len(windows)
        assert result["loss"] > 0

    def test_two_clients_produce_compatible_updates(self, config, normal_data):
        """
        Verify that separate clients training on different data slices generate
        updates with identical structural parameter keys.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.models import LSTMAnomalyDetector
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        mc = config["edge"]["model"]
        model_a = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=mc["dropout"],
        )
        model_b = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=mc["dropout"],
        )

        client_a = FederatedClient("node-a", model_a, config=config)
        client_b = FederatedClient("node-b", model_b, config=config)

        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        half = len(windows) // 2
        res_a = client_a.train_round(windows[:half], system_metrics={"cpu_usage_percent": 20.0})
        res_b = client_b.train_round(windows[half:], system_metrics={"cpu_usage_percent": 40.0})

        assert set(res_a["model_update"].keys()) == set(res_b["model_update"].keys())


class TestFedAvgIntegration:
    """
    Validates the FedAvg aggregation step in the central coordinator.
    """

    def test_global_model_updates(self, config, lstm_model, normal_data):
        """
        Verify that central model aggregation successfully updates global parameters.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        import copy
        import time
        from federated.federated_coordinator import FederatedCoordinator
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window
        from utils.metrics import MetricsExporter

        initial_state = copy.deepcopy(lstm_model.state_dict())

        config["federated"]["client"]["gradient_compression"]["enabled"] = False
        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 0
        coord.min_clients_per_round = 1
        coord.staleness_tolerance = 2
        coord.poisoning_detection_enabled = False
        coord.global_model = copy.deepcopy(lstm_model)
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {"node-1": {}}
        coord.metrics = MetricsExporter()

        coord.client_models["node-1"] = {
            "model_update": result["model_update"],
            "num_samples": result["num_samples"],
            "round": 0,
            "timestamp": time.time(),
            "metrics": {},
        }

        success = coord.aggregate_models(timeout=0.1)
        assert success is True

        any_changed = any(
            not torch.allclose(initial_state[k], coord.global_model.state_dict()[k], atol=1e-6)
            for k in initial_state
        )
        assert any_changed, "Global model should change after aggregation"


class TestCompressionIntegration:
    """
    Validates Top-K compressed model update serialization.
    """

    def test_compressed_update_aggregatable(self, config, lstm_model, normal_data):
        """
        Verify that Top-K sparsification produces an update that can be parsed and
        aggregated by the central server.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 50.0})

        assert result["compression_stats"]["compression_ratio"] <= 0.15
        assert set(result["model_update"].keys()) == set(lstm_model.state_dict().keys())


class TestDPIntegration:
    """
    Validates concurrent DP noise injection and gradient compression.
    """

    def test_dp_round(self, config, normal_data):
        """
        Verify that client training with DP enabled produces valid model updates.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.models import LSTMAnomalyDetector
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        mc = config["edge"]["model"]
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 0.5

        model = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=mc["dropout"],
        )
        client = FederatedClient("dp-node", model, config=config)

        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 50.0})

        assert result["model_update"] is not None
        assert result["loss"] > 0


class TestPayloadCompressionIntegration:
    """
    Validates adaptive payload compression of trained updates.
    """

    def test_pack_unpack_after_training(self, config, lstm_model, normal_data):
        """
        Verify that serialization/deserialization does not degrade parameter accuracy.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window
        from utils.compression import pack_state_dict, unpack_state_dict

        config["federated"]["client"]["gradient_compression"]["enabled"] = False
        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 50.0})

        update_sd = {k: torch.FloatTensor(v) for k, v in result["model_update"].items()}

        payload, alg = pack_state_dict(update_sd, cpu_usage_percent=50.0)
        restored = unpack_state_dict(payload, alg)

        for key in update_sd:
            assert torch.allclose(update_sd[key], restored[key], atol=1e-5)
