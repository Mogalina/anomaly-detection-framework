import copy
import time
import numpy as np
import pytest
import torch


def _make_coordinator(config, lstm_model, **overrides):
    """
    Helper function to build a lightweight FederatedCoordinator without DB/Redis.

    Args:
        config: Test configuration fixture
        lstm_model: Small LSTM anomaly detector fixture
        **overrides: Optional attributes to override defaults

    Returns:
        FederatedCoordinator: Lightly initialized coordinator instance.
    """
    from federated.federated_coordinator import FederatedCoordinator
    from utils.metrics import MetricsExporter

    defaults = dict(
        min_clients_per_round=2,
        staleness_tolerance=2,
        poisoning_detection_enabled=False,
    )
    defaults.update(overrides)

    coord = FederatedCoordinator.__new__(FederatedCoordinator)
    coord.config = config
    coord.current_round = 0
    coord.global_model = copy.deepcopy(lstm_model)
    coord.device = torch.device("cpu")
    coord.round_history = []
    coord.client_models = {}
    coord.registered_clients = {}
    coord.client_metrics = {}
    coord.metrics = MetricsExporter()

    for key, val in defaults.items():
        setattr(coord, key, val)

    return coord


def _state_as_lists(model):
    """
    Helper function to serialize parameters as plain list values.

    Args:
        model: PyTorch model whose state should be serialized

    Returns:
        dict: The model's state dict with lists as values.
    """
    return {k: v.numpy().tolist() for k, v in model.state_dict().items()}


# Client Registration

class TestClientRegistration:
    """
    Validates central coordinator registration workflows and validations.
    """

    def test_register_new_client(self, config, lstm_model):
        """
        Verify unregistered nodes cannot submit update payloads.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        from collections import defaultdict
        coord.client_metrics = defaultdict(list)
        coord.registered_clients = {}

        result = coord.receive_update("unknown", {}, 100, {})
        assert result["status"] == "error"

    def test_receive_update_from_registered_client(self, config, lstm_model):
        """
        Verify registered clients are accepted and updates cached.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from collections import defaultdict
        coord = _make_coordinator(config, lstm_model)
        coord.client_metrics = defaultdict(list)
        coord.registered_clients = {"node-1": {"rounds_participated": 0, "last_seen": 0}}

        state = _state_as_lists(lstm_model)
        result = coord.receive_update("node-1", state, 100, {"loss": 0.1})
        assert result["status"] == "success"
        assert "node-1" in coord.client_models


class TestFedAvgAggregation:
    """
    Validates model aggregation using Federated Averaging (FedAvg).
    """

    def test_aggregation_with_equal_weights(self, config, lstm_model):
        """
        Verify aggregation of unchanged model weights yields an unchanged global model.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        state = _state_as_lists(lstm_model)

        for cid in ["n1", "n2"]:
            coord.registered_clients[cid] = {}
            coord.client_models[cid] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }

        original = copy.deepcopy(coord.global_model.state_dict())
        result = coord.aggregate_models(timeout=0.1)

        assert result is True
        for key in original:
            assert torch.allclose(coord.global_model.state_dict()[key], original[key], atol=1e-4)

    def test_weighted_average(self, config, lstm_model):
        """
        Verify client aggregation values are weighted by sample sizes.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        state_a = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        state_b = {k: (v * 3).numpy().tolist() for k, v in lstm_model.state_dict().items()}

        coord.registered_clients = {"a": {}, "b": {}}
        coord.client_models["a"] = {
            "model_update": state_a, "num_samples": 100,
            "round": 0, "timestamp": time.time(), "metrics": {},
        }
        coord.client_models["b"] = {
            "model_update": state_b, "num_samples": 300,
            "round": 0, "timestamp": time.time(), "metrics": {},
        }

        result = coord.aggregate_models(timeout=0.1)
        assert result is True

        for key in lstm_model.state_dict():
            expected = 0.25 * lstm_model.state_dict()[key] + 0.75 * (lstm_model.state_dict()[key] * 3)
            actual = coord.global_model.state_dict()[key]
            assert torch.allclose(actual, expected, atol=1e-4), f"Mismatch on {key}"

    def test_aggregation_clears_client_models(self, config, lstm_model):
        """
        Verify client cache buffer is cleared on successful aggregation.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        state = _state_as_lists(lstm_model)
        for cid in ["n1", "n2"]:
            coord.registered_clients[cid] = {}
            coord.client_models[cid] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }
        coord.aggregate_models(timeout=0.1)
        assert len(coord.client_models) == 0

    def test_aggregation_records_round_history(self, config, lstm_model):
        """
        Verify metadata values of completed rounds are recorded in history logs.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        state = _state_as_lists(lstm_model)
        for cid in ["n1", "n2"]:
            coord.registered_clients[cid] = {}
            coord.client_models[cid] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }
        coord.aggregate_models(timeout=0.1)
        assert len(coord.round_history) == 1
        assert coord.round_history[0]["num_clients"] == 2


class TestQuorum:
    """
    Validates minimum client participation checks during rounds.
    """

    def test_fails_below_quorum(self, config, lstm_model):
        """
        Verify aggregation aborts if client counts fall below quorum.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model, min_clients_per_round=3)
        state = _state_as_lists(lstm_model)
        coord.registered_clients = {"n1": {}}
        coord.client_models["n1"] = {
            "model_update": state, "num_samples": 100,
            "round": 0, "timestamp": time.time(), "metrics": {},
        }
        assert coord.aggregate_models(timeout=0.1) is False

    def test_succeeds_at_exact_quorum(self, config, lstm_model):
        """
        Verify aggregation runs when exact quorum limits are met.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model, min_clients_per_round=2)
        state = _state_as_lists(lstm_model)
        for cid in ["n1", "n2"]:
            coord.registered_clients[cid] = {}
            coord.client_models[cid] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }
        assert coord.aggregate_models(timeout=0.1) is True


class TestStalenessFiltering:
    """
    Validates filtering of obsolete client update epochs.
    """

    def test_stale_updates_rejected(self, config, lstm_model):
        """
        Verify updates from obsolete training rounds are filtered out.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model, staleness_tolerance=1, min_clients_per_round=2)
        coord.current_round = 5
        state = _state_as_lists(lstm_model)

        coord.registered_clients = {"n1": {}, "n2": {}}
        coord.client_models["n1"] = {
            "model_update": state, "num_samples": 100,
            "round": 5, "timestamp": time.time(), "metrics": {},
        }
        coord.client_models["n2"] = {
            "model_update": state, "num_samples": 100,
            "round": 2, "timestamp": time.time(), "metrics": {},
        }
        assert coord.aggregate_models(timeout=0.1) is False

    def test_within_tolerance_accepted(self, config, lstm_model):
        """
        Verify updates within historical tolerance are accepted.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model, staleness_tolerance=2, min_clients_per_round=2)
        coord.current_round = 5
        state = _state_as_lists(lstm_model)

        coord.registered_clients = {"n1": {}, "n2": {}}
        coord.client_models["n1"] = {
            "model_update": state, "num_samples": 100,
            "round": 5, "timestamp": time.time(), "metrics": {},
        }
        coord.client_models["n2"] = {
            "model_update": state, "num_samples": 100,
            "round": 4, "timestamp": time.time(), "metrics": {},
        }
        assert coord.aggregate_models(timeout=0.1) is True


class TestPoisoningDetection:
    """
    Validates outlier update identification and filtering (poisoning detection).
    """

    def _make_poisoning_coord(self, config, lstm_model):
        """
        Helper method to construct coordinator configured for poisoning checks.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture

        Returns:
            FederatedCoordinator: A coordinator instance ready for poisoning testing.
        """
        coord = _make_coordinator(
            config, lstm_model,
            poisoning_detection_enabled=True,
            min_clients_per_round=3,
        )
        coord.zscore_threshold = 1.5
        coord.autoencoder_threshold = 100.0
        coord.validation_samples = []
        return coord

    def test_normal_updates_pass(self, config, lstm_model):
        """
        Verify consistent non-poisonous updates pass checks unimpeded.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = self._make_poisoning_coord(config, lstm_model)
        state = _state_as_lists(lstm_model)
        updates = {}
        for i in range(4):
            updates[f"n{i}"] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }
        valid = coord._detect_poisoned_updates(updates)
        assert len(valid) == 4

    def test_extreme_outlier_filtered(self, config, lstm_model):
        """
        Verify that anomalously scaled update updates are filtered as outliers.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = self._make_poisoning_coord(config, lstm_model)
        state = _state_as_lists(lstm_model)
        poisoned = {k: (torch.FloatTensor(v) * 1000).numpy().tolist() for k, v in state.items()}

        updates = {}
        for i in range(4):
            updates[f"n{i}"] = {
                "model_update": state, "num_samples": 100,
                "round": 0, "timestamp": time.time(), "metrics": {},
            }
        updates["poisoned"] = {
            "model_update": poisoned, "num_samples": 100,
            "round": 0, "timestamp": time.time(), "metrics": {},
        }
        valid = coord._detect_poisoned_updates(updates)
        assert "poisoned" not in valid

    def test_too_few_clients_skips_detection(self, config, lstm_model):
        """
        Verify that checks are skipped if round has too few participants.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = self._make_poisoning_coord(config, lstm_model)
        state = _state_as_lists(lstm_model)
        updates = {
            "n1": {"model_update": state, "num_samples": 100, "round": 0, "timestamp": time.time(), "metrics": {}},
            "n2": {"model_update": state, "num_samples": 100, "round": 0, "timestamp": time.time(), "metrics": {}},
        }
        valid = coord._detect_poisoned_updates(updates)
        assert len(valid) == 2


class TestCoordinatorSerialization:
    """
    Validates model weight serialization routines.
    """

    def test_serialize_model(self, config, lstm_model):
        """
        Verify model parameter dictionaries map to serialized list structures.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        serialized = coord._serialize_model(coord.global_model)
        assert isinstance(serialized, dict)
        for key in lstm_model.state_dict():
            assert key in serialized
            assert isinstance(serialized[key], list)

    def test_get_global_model(self, config, lstm_model):
        """
        Verify retrieval of serialized global model representation.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        coord = _make_coordinator(config, lstm_model)
        gm = coord.get_global_model()
        assert isinstance(gm, dict)
        assert len(gm) == len(lstm_model.state_dict())


class TestClientMetrics:
    """
    Validates client performance and system telemetry collection.
    """

    def test_get_client_metrics_none(self, config, lstm_model):
        """
        Verify asking for unregistered client metrics yields None.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from collections import defaultdict
        coord = _make_coordinator(config, lstm_model)
        coord.client_metrics = defaultdict(list)
        assert coord.get_client_metrics("nonexistent") is None

    def test_get_client_metrics_latest(self, config, lstm_model):
        """
        Verify asking for client metrics fetches the most recent entry.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from collections import defaultdict
        coord = _make_coordinator(config, lstm_model)
        coord.client_metrics = defaultdict(list)
        coord.client_metrics["n1"].append({"loss": 0.5})
        coord.client_metrics["n1"].append({"loss": 0.3})
        assert coord.get_client_metrics("n1") == {"loss": 0.3}

    def test_get_all_clients_system_metrics(self, config, lstm_model):
        """
        Verify aggregation of CPU and memory stats across clients.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from collections import defaultdict
        coord = _make_coordinator(config, lstm_model)
        coord.client_metrics = defaultdict(list)
        coord.client_metrics["n1"].append({"system_metrics": {"cpu": 50}})
        coord.client_metrics["n2"].append({"system_metrics": {"cpu": 70}})

        all_sys = coord.get_all_clients_system_metrics()
        assert "n1" in all_sys
        assert all_sys["n1"]["cpu"] == 50
