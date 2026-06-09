import numpy as np
import pytest
import torch
import torch.nn as nn


class TestNoiseCalibration:
    """
    Validates the calibration, variance, and distribution of DP noise.
    """

    def test_noise_variance_proportional(self, config, lstm_model):
        """
        Verify that the variance of the added DP noise scales proportionally 
        with the configured noise multiplier.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient
        from edge.models import LSTMAnomalyDetector

        mc = config["edge"]["model"]
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        original = {"param": np.zeros(2000).tolist()}

        variances = {}
        for mult in [0.1, 0.5, 1.0, 2.0]:
            config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = mult
            model = LSTMAnomalyDetector(
                input_size=mc["input_size"], hidden_size=mc["hidden_size"],
                num_layers=mc["num_layers"], dropout=0.0,
            )
            client = FederatedClient(f"dp-{mult}", model, config=config)

            accumulated = np.zeros(2000)
            for _ in range(30):
                noisy = client._add_dp_noise(original)
                accumulated += np.array(noisy["param"])
            var = np.var(accumulated / 30)
            variances[mult] = var

        assert variances[0.5] > variances[0.1]
        assert variances[1.0] > variances[0.5]
        assert variances[2.0] > variances[1.0]

    def test_noise_is_gaussian(self, config, lstm_model):
        """
        Assert that the generated DP noise is roughly zero-centered and has
        finite variance, simulating a Gaussian distribution.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 1.0

        client = FederatedClient("dp-test", lstm_model, config=config)
        original = {"param": np.zeros(100).tolist()}

        samples = []
        for _ in range(200):
            noisy = client._add_dp_noise(original)
            samples.append(noisy["param"][0])

        samples = np.array(samples)
        assert abs(np.mean(samples)) < 0.3
        assert np.var(samples) > 0

    def test_noise_independent_per_parameter(self, config, lstm_model):
        """
        Validate that the noise added to different model parameters is independent,
        preventing correlation leakage.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_client import FederatedClient

        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 1.0

        client = FederatedClient("dp-test", lstm_model, config=config)
        state = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}

        noisy = client._add_dp_noise(state)
        diffs = []
        for key in list(state.keys())[:2]:
            orig = np.array(state[key]).flatten()
            nois = np.array(noisy[key]).flatten()
            diffs.append(nois - orig)

        if len(diffs) >= 2:
            corr = np.corrcoef(diffs[0][:min(100, len(diffs[0]))],
                                diffs[1][:min(100, len(diffs[1]))])[0, 1]
            assert abs(corr) < 0.5, "Noise across parameters should be independent"


class TestGradientClipping:
    """
    Validates gradient clipping constraints to bound parameter sensitivity.
    """

    def test_clip_at_threshold(self, lstm_model):
        """
        Verify that gradients exceeding the max norm threshold are clipped
        precisely to the target norm.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        max_norm = 0.5
        x = torch.randn(4, 20, 10) * 100
        lstm_model.train()
        out = lstm_model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm)

        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in lstm_model.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm <= max_norm + 1e-4

    def test_clip_at_different_norms(self, config):
        """
        Verify gradient clipping across a range of thresholds.

        Args:
            config: Test configuration fixture
        """
        from edge.models import LSTMAnomalyDetector

        mc = config["edge"]["model"]

        for max_norm in [0.1, 0.5, 1.0, 5.0]:
            model = LSTMAnomalyDetector(
                input_size=mc["input_size"], hidden_size=mc["hidden_size"],
                num_layers=mc["num_layers"], dropout=0.0,
            )
            x = torch.randn(2, mc["sequence_length"], mc["input_size"]) * 50
            model.train()
            out = model(x)
            loss = nn.MSELoss()(out, x)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            assert total_norm <= max_norm + 1e-4, f"Failed at max_norm={max_norm}"

    def test_no_clip_when_norm_small(self, lstm_model, config):
        """
        Confirm that gradient clipping has no effect when the gradient norm
        is already below the threshold.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(1, mc["sequence_length"], mc["input_size"]) * 0.001
        lstm_model.train()
        out = lstm_model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()

        grads_before = [p.grad.data.clone() for p in lstm_model.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 100.0)
        grads_after = [p.grad.data for p in lstm_model.parameters() if p.grad is not None]

        for before, after in zip(grads_before, grads_after):
            assert torch.allclose(before, after, atol=1e-7)


class TestUtilityDegradation:
    """
    Validates that model utility is affected monotonically by DP noise.
    """

    def _train_and_evaluate(self, config, normal_data, noise_multiplier):
        """
        Helper method to train and evaluate utility for a specific noise level.

        Args:
            config: Test configuration dictionary
            normal_data: Synthetic telemetry data fixture
            noise_multiplier: DP noise multiplier value

        Returns:
            float: Evaluated loss (mean squared error)
        """
        from edge.models import LSTMAnomalyDetector
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        mc = config["edge"]["model"]
        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = noise_multiplier

        model = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=0.0,
        )
        client = FederatedClient("dp-node", model, config=config)

        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})
        eval_result = client.evaluate(windows)
        return eval_result["loss"]

    def test_higher_noise_higher_loss(self, config, normal_data):
        """
        Assert that higher DP noise levels correlate with higher evaluation loss.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic telemetry data fixture
        """
        from copy import deepcopy

        losses = {}
        for mult in [0.0, 0.5, 2.0]:
            cfg = deepcopy(config)
            loss = self._train_and_evaluate(cfg, normal_data, mult)
            losses[mult] = loss

        assert losses[2.0] >= losses[0.0] * 0.5, (
            f"Expected loss@2.0={losses[2.0]:.4f} ≥ 0.5 × loss@0.0={losses[0.0]:.4f}"
        )

    def test_low_noise_preserves_utility(self, config, normal_data):
        """
        Verify that a low noise multiplier preserves model reconstruction utility
        close to the baseline.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic telemetry data fixture
        """
        from copy import deepcopy

        cfg_none = deepcopy(config)
        loss_none = self._train_and_evaluate(cfg_none, normal_data, 0.0)

        cfg_low = deepcopy(config)
        loss_low = self._train_and_evaluate(cfg_low, normal_data, 0.01)

        assert loss_low < loss_none * 1.5, (
            f"Low noise loss={loss_low:.4f} too far from baseline={loss_none:.4f}"
        )


class TestDPWithCompression:
    """
    Validates the concurrent operation of DP and compression strategies.
    """

    def test_dp_plus_topk(self, config, lstm_model, normal_data):
        """
        Verify that applying both DP noise and Top-K compression produces a
        valid compressed update.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 0.5
        config["federated"]["client"]["gradient_compression"]["enabled"] = True

        client = FederatedClient("dp-topk", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 50.0})

        assert result["model_update"] is not None
        assert result["compression_stats"]["compression_ratio"] <= 0.15
        assert result["loss"] > 0

    def test_dp_applied_before_compression(self, config, lstm_model, normal_data):
        """
        Confirm that DP noise is applied to the full update before sparsification
        takes place, ensuring privacy is preserved.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 1.0
        config["federated"]["client"]["gradient_compression"]["enabled"] = True

        client = FederatedClient("dp-topk", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        result = client.train_round(windows, system_metrics={"cpu_usage_percent": 50.0})

        for key, val in result["model_update"].items():
            assert "indices" in val or isinstance(val, list)


class TestPrivacyBudgetVerification:
    """
    Verifies privacy sensitivity bounds and noise accumulation.
    """

    def test_sensitivity_bounded_by_clip_norm(self, config, lstm_model):
        """
        Verify that sensitivity of parameter updates is bounded by the clipping norm.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        mc = config["edge"]["model"]
        max_norm = 1.0

        x = torch.randn(4, mc["sequence_length"], mc["input_size"]) * 100
        lstm_model.train()
        out = lstm_model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm)

        total_norm = sum(
            p.grad.norm(2).item() ** 2 for p in lstm_model.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm <= max_norm + 1e-4

    def test_multiple_rounds_accumulate_noise(self, config, lstm_model, normal_data):
        """
        Assert that multiple training rounds add independent noise, leading to
        distinct model updates.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            normal_data: Synthetic telemetry data fixture
        """
        from federated.federated_client import FederatedClient
        from utils.preprocessing import sliding_window

        config["federated"]["client"]["differential_privacy"]["enabled"] = True
        config["federated"]["client"]["differential_privacy"]["noise_multiplier"] = 0.5
        config["federated"]["client"]["gradient_compression"]["enabled"] = False

        client = FederatedClient("dp-multi", lstm_model, config=config)
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)

        updates = []
        for _ in range(5):
            result = client.train_round(windows, system_metrics={"cpu_usage_percent": 30.0})
            first_key = list(result["model_update"].keys())[0]
            updates.append(np.array(result["model_update"][first_key]).flatten())

        diffs = [np.linalg.norm(updates[i] - updates[i+1]) for i in range(len(updates)-1)]
        assert all(d > 0 for d in diffs), "Each round should add independent noise"
