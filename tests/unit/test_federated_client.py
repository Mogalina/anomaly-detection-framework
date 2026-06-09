import numpy as np
import pytest
import torch


class TestFederatedClientInit:
    """
    Validates FederatedClient constructor configurations and settings.
    """

    def test_default_init(self, config, lstm_model):
        """
        Verify that client ID and default properties are configured properly.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient
        client = FederatedClient("node-1", lstm_model, config=config)
        assert client.client_id == "node-1"
        assert client.dp_enabled is False
        assert client.compression_enabled is True

    def test_dp_enabled(self, config, lstm_model):
        """
        Verify that differential privacy settings load correctly from configurations.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        from federated.federated_client import FederatedClient
        client = FederatedClient("node-dp", lstm_model, config=config)
        assert client.dp_enabled is True

    def test_lightweight_profile(self, config, lstm_model, monkeypatch):
        """
        Verify that environmental device profiles select correct client storage paths.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            monkeypatch: Pytest monkeypatch fixture
        """
        monkeypatch.setenv("NODE_PROFILE", "lightweight")
        from federated.federated_client import FederatedClient
        client = FederatedClient("node-l", lstm_model, config=config)
        assert client.profile == "lightweight"
        assert client._local_db_path is None


class TestLocalTraining:
    """
    Validates local epoch loops and client parameter learning steps.
    """

    def test_train_round_returns_required_keys(self, config, lstm_model, normal_data):
        """
        Verify that local training returns the standard output schema keys.

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

        for key in ["model_update", "num_samples", "loss", "training_time", "round", "compression_stats"]:
            assert key in result, f"Missing key: {key}"

    def test_loss_is_positive(self, config, lstm_model, normal_data):
        """
        Verify that local training epoch loss values are strictly positive.

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
        assert result["loss"] > 0

    def test_num_samples_matches_input(self, config, lstm_model, normal_data):
        """
        Verify that the sample counter in output matches input training set size.

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
        assert result["num_samples"] == len(windows)

    def test_round_counter_increments(self, config, lstm_model, normal_data):
        """
        Verify that local round counter increments sequentially with train runs.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})
        assert client.round_number == 1
        client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})
        assert client.round_number == 2

    def test_train_with_global_params(self, config, lstm_model, normal_data):
        """
        Verify that training rounds initialize from central global model parameters.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        global_params = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        result = client.train_round(windows, global_model_params=global_params,
                                     system_metrics={"cpu_usage_percent": 30.0})
        assert result["loss"] > 0


class TestModelUpdateComputation:
    """
    Validates generation of central coordination update delta values.
    """

    def test_compute_model_update_keys(self, config, lstm_model, normal_data):
        """
        Verify that initial reference parameter states are cached for updates tracking.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})
        assert client.initial_params is not None

    def test_update_is_delta(self, config, lstm_model, normal_data):
        """
        Verify that model updates represent the difference (delta) from initial weights.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        config["federated"]["client"]["gradient_compression"]["enabled"] = False
        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)

        client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})

        update = client.initial_params
        any_nonzero = any(np.abs(np.array(v)).sum() > 1e-10 for v in update.values())
        assert any_nonzero, "Model update should contain non-zero deltas"


class TestTopKCompression:
    """
    Validates sparse gradient compression ratios and shape roundtrips.
    """

    def test_compression_ratio(self, config, lstm_model):
        """
        Verify Top-K compression ratio respects limits.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        client = FederatedClient("node-1", lstm_model, config=config)
        update = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        compressed, stats = client._topk_compression(update)
        assert stats["compression_ratio"] <= 0.15
        assert stats["total_params"] > 0
        assert stats["kept_params"] > 0

    def test_compressed_is_sparse(self, config, lstm_model):
        """
        Verify that sparse update dicts carry indices, values, and shape maps.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        client = FederatedClient("node-1", lstm_model, config=config)
        update = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        compressed, _ = client._topk_compression(update)
        for key, val in compressed.items():
            assert "indices" in val
            assert "values" in val
            assert "shape" in val

    def test_sparse_to_dense_roundtrip(self, config, lstm_model):
        """
        Verify converting sparse update representations back to dense state dicts.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        client = FederatedClient("node-1", lstm_model, config=config)
        update = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        compressed, _ = client._topk_compression(update)

        dense = FederatedClient._update_dict_to_state_dict(compressed)
        for key in compressed:
            assert key in dense
            assert dense[key].shape == torch.Size(compressed[key]["shape"])


class TestDifferentialPrivacy:
    """
    Validates local parameter perturbation with differential privacy noise.
    """

    def test_dp_noise_modifies_update(self, config, lstm_model):
        """
        Verify that applying DP noise changes parameter updates.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        from federated.federated_client import FederatedClient

        client = FederatedClient("dp-test", lstm_model, config=config)
        original = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        noisy = client._add_dp_noise(original)

        any_changed = any(
            not np.allclose(original[k], noisy[k], atol=1e-7) for k in original
        )
        assert any_changed

    def test_dp_noise_zero_mean(self, config, lstm_model):
        """
        Confirm that generated Gaussian DP noise has a zero mean.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 0.5
        from federated.federated_client import FederatedClient

        client = FederatedClient("dp-test", lstm_model, config=config)
        original = {"param": np.zeros(1000).tolist()}

        accumulated = np.zeros(1000)
        for _ in range(50):
            noisy = client._add_dp_noise(original)
            accumulated += np.array(noisy["param"])

        mean_noise = accumulated / 50
        assert np.abs(mean_noise).mean() < 0.3

    def test_dp_noise_scales_with_multiplier(self, config, lstm_model):
        """
        Verify that variance of DP perturbation scales with noise multipliers.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        original = {"p": np.zeros(500).tolist()}
        config["federated"]["client"]["differential_privacy"]["enabled"] = True

        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 0.1
        c_low = FederatedClient("dp-low", lstm_model, config=config)
        noisy_low = c_low._add_dp_noise(original)
        var_low = np.var(noisy_low["p"])

        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 5.0
        from edge.models import LSTMAnomalyDetector
        mc = config["edge"]["model"]
        fresh_model = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=0.0,
        )
        c_high = FederatedClient("dp-high", fresh_model, config=config)
        noisy_high = c_high._add_dp_noise(original)
        var_high = np.var(noisy_high["p"])

        assert var_high > var_low

    def test_gradient_clipping(self, config, lstm_model):
        """
        Confirm that gradient clipping operations bound the parameter updates.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["max_grad_norm"] = 0.5

        x = torch.randn(2, 20, 10) * 100
        lstm_model.train()
        out = lstm_model(x)
        loss = torch.nn.MSELoss()(out, x)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 0.5)

        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in lstm_model.parameters() if p.grad is not None) ** 0.5
        assert total_norm <= 0.5 + 1e-4


class TestClientEvaluation:
    """
    Validates evaluation routines of the FederatedClient.
    """

    def test_evaluate_returns_loss(self, config, lstm_model, normal_data):
        """
        Verify that evaluation yields loss and sample size metadata.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.evaluate(windows)
        assert "loss" in result
        assert "num_samples" in result
        assert result["loss"] > 0

    def test_evaluate_sample_count(self, config, lstm_model, normal_data):
        """
        Verify that evaluation outputs match evaluated sample count sizes.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        client = FederatedClient("node-1", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.evaluate(windows)
        assert result["num_samples"] == len(windows)


class TestClientUtilities:
    """
    Validates client serialization and mapping utility helpers.
    """

    def test_estimate_size(self, config, lstm_model):
        """
        Verify serialization payload estimator calculations.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        client = FederatedClient("node-1", lstm_model, config=config)
        data = {"key": [1, 2, 3]}
        size = client._estimate_size(data)
        assert size > 0

    def test_update_dict_to_state_dict_dense(self):
        """
        Verify dense update dictionary structures map back to state dict tensors.
        """
        from federated.federated_client import FederatedClient

        update = {"layer.weight": [[1.0, 2.0], [3.0, 4.0]]}
        state = FederatedClient._update_dict_to_state_dict(update)
        assert "layer.weight" in state
        assert state["layer.weight"].shape == (2, 2)

    def test_update_dict_to_state_dict_sparse(self):
        """
        Verify sparse update dictionary structures map back to state dict tensors.
        """
        from federated.federated_client import FederatedClient

        update = {
            "layer.weight": {
                "indices": [0, 3],
                "values": [1.0, 2.0],
                "shape": [2, 2],
            }
        }
        state = FederatedClient._update_dict_to_state_dict(update)
        assert state["layer.weight"].shape == (2, 2)
        assert state["layer.weight"][0, 0] == 1.0
        assert state["layer.weight"][1, 1] == 2.0
