import copy
import time
import numpy as np
import pytest
import torch


class TestNodeDropout:
    """
    Fault tolerance: validation of aggregation behavior when clients drop out.
    """

    def test_aggregation_fails_below_quorum(self, config, lstm_model):
        """
        Verify that aggregate_models() returns False if fewer than the 
        configured min_clients submit their local updates.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_coordinator import FederatedCoordinator

        config["federated"]["coordinator"]["round_timeout_seconds"] = 1
        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 0
        coord.min_clients_per_round = 3
        coord.staleness_tolerance = 2
        coord.poisoning_detection_enabled = False
        coord.global_model = lstm_model
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {"node-1": {}}

        from utils.metrics import MetricsExporter
        coord.metrics = MetricsExporter()

        state = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        coord.client_models["node-1"] = {
            "model_update": state,
            "num_samples": 100,
            "round": 0,
            "timestamp": time.time(),
            "metrics": {},
        }

        result = coord.aggregate_models(timeout=1)
        assert result is False, "Aggregation should fail below quorum"

    def test_aggregation_succeeds_at_quorum(self, config, lstm_model):
        """
        Verify that aggregate_models() succeeds when exactly min_clients
        updates arrive in time.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_coordinator import FederatedCoordinator
        from utils.metrics import MetricsExporter

        config["federated"]["coordinator"]["round_timeout_seconds"] = 1

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 0
        coord.min_clients_per_round = 2
        coord.staleness_tolerance = 2
        coord.poisoning_detection_enabled = False
        coord.global_model = copy.deepcopy(lstm_model)
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {"node-1": {}, "node-2": {}}
        coord.metrics = MetricsExporter()

        state = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}
        for cid in ["node-1", "node-2"]:
            coord.client_models[cid] = {
                "model_update": state,
                "num_samples": 100,
                "round": 0,
                "timestamp": time.time(),
                "metrics": {},
            }

        result = coord.aggregate_models(timeout=1)
        assert result is True


class TestStaleUpdates:
    """
    Fault tolerance: validation of stale model update rejection logic.
    """

    def test_stale_updates_rejected(self, config, lstm_model):
        """
        Verify that updates originating from old federated rounds are rejected 
        according to the staleness tolerance configuration.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_coordinator import FederatedCoordinator
        from utils.metrics import MetricsExporter

        config["federated"]["coordinator"]["round_timeout_seconds"] = 1

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 5
        coord.min_clients_per_round = 2
        coord.staleness_tolerance = 1  # Only accept round >= 4
        coord.poisoning_detection_enabled = False
        coord.global_model = copy.deepcopy(lstm_model)
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {"n1": {}, "n2": {}, "n3": {}}
        coord.metrics = MetricsExporter()

        state = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}

        coord.client_models["n1"] = {
            "model_update": state, "num_samples": 100,
            "round": 5, "timestamp": time.time(), "metrics": {},
        }
        coord.client_models["n2"] = {
            "model_update": state, "num_samples": 100,
            "round": 2, "timestamp": time.time(), "metrics": {},  # stale
        }
        coord.client_models["n3"] = {
            "model_update": state, "num_samples": 100,
            "round": 1, "timestamp": time.time(), "metrics": {},  # stale
        }

        result = coord.aggregate_models(timeout=1)
        assert result is False, "Should fail because only 1 non-stale update"


class TestPoisoningDetection:
    """
    Fault tolerance: validation of poisoned update detection algorithms.
    """

    def test_poisoned_update_detected(self, config, lstm_model):
        """
        Verify that extreme outlier parameter updates from a compromised client 
        are correctly flagged and filtered via the Z-score logic.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_coordinator import FederatedCoordinator
        from utils.metrics import MetricsExporter

        config["federated"]["coordinator"]["round_timeout_seconds"] = 1
        config["federated"]["poisoning_detection"]["zscore_threshold"] = 1.5

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 0
        coord.min_clients_per_round = 3
        coord.staleness_tolerance = 2
        coord.poisoning_detection_enabled = True
        coord.zscore_threshold = 1.5
        coord.autoencoder_threshold = 100.0  # Disable AE detection
        coord.global_model = copy.deepcopy(lstm_model)
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {f"n{i}": {} for i in range(5)}
        coord.metrics = MetricsExporter()
        coord.validation_samples = []

        normal_state = {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}

        for i in range(4):
            coord.client_models[f"n{i}"] = {
                "model_update": normal_state,
                "num_samples": 100,
                "round": 0,
                "timestamp": time.time(),
                "metrics": {},
            }

        poisoned_state = {k: (v * 1000).numpy().tolist() for k, v in lstm_model.state_dict().items()}
        coord.client_models["n4"] = {
            "model_update": poisoned_state,
            "num_samples": 100,
            "round": 0,
            "timestamp": time.time(),
            "metrics": {},
        }

        valid = coord._detect_poisoned_updates(coord.client_models)
        assert "n4" not in valid, "Poisoned client should be filtered out"
        assert len(valid) >= 3, "Normal clients should remain"


class TestMalformedInput:
    """
    Fault tolerance: validation of malformed input handling and error bounds.
    """

    def test_edge_detector_wrong_feature_count(self, config):
        """
        Verify that EdgeDetector pads or truncates incoming features when the
        feature dimension does not match the active model's input size.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="test", config=config)
        expected_features = config["edge"]["model"]["input_size"]

        small = np.ones(3, dtype=np.float32)
        det.update_data(small)
        assert len(det.data_buffer[-1]) == expected_features

        big = np.ones(expected_features + 5, dtype=np.float32)
        det.update_data(big)
        assert len(det.data_buffer[-1]) == expected_features

    def test_unregistered_client_rejected(self, config, lstm_model):
        """
        Verify that the coordinator immediately rejects model updates submitted 
        by unregistered client nodes.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from federated.federated_coordinator import FederatedCoordinator
        from utils.metrics import MetricsExporter

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.registered_clients = {}
        coord.client_models = {}
        coord.client_metrics = {}
        coord.metrics = MetricsExporter()

        result = coord.receive_update(
            client_id="unknown-node",
            model_update={},
            num_samples=100,
            metrics={},
        )
        assert result["status"] == "error"

    def test_causal_graph_missing_service(self, config):
        """
        Confirm that querying operations on missing services in the causal graph 
        returns empty results instead of crashing.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph

        cg = CausalGraph(config)
        assert cg.get_downstream_services("nonexistent") == set()
        assert cg.get_upstream_services("nonexistent") == set()
        assert cg.get_propagation_path("a", "b") is None
        assert cg.get_impact_score("nonexistent") == 0.0

    def test_rca_with_unknown_services(self, config, causal_graph):
        """
        Verify that the Root Cause Analyzer handles anomalous services that are 
        missing from the dependency graph gracefully.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"totally-unknown-service"})
        assert len(result["root_causes"]) >= 0

    def test_threshold_tuner_uninitialized_service(self, config):
        """
        Verify that tune_threshold() automatically registers and initializes a 
        service that was not previously registered.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        new_th = tuner.tune_threshold("never-seen-before")
        assert isinstance(new_th, float)
        assert new_th > 0


class TestGraphResilience:
    """
    Fault tolerance: validation of causal graph weight decay and weak edge pruning.
    """

    def test_edge_weight_decay(self, config):
        """
        Verify that edge weights decay properly over time under active decay steps.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph

        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=100)
        initial_weight = cg.graph["a"]["b"]["weight"]

        cg._apply_edge_decay()
        decayed_weight = cg.graph["a"]["b"]["weight"]
        assert decayed_weight < initial_weight

    def test_weak_edge_pruning(self, config):
        """
        Verify that edges with weights falling below the minimum threshold are pruned.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph

        cg = CausalGraph(config)
        cg.add_dependency("a", "b", call_count=1)
        cg.graph["a"]["b"]["weight"] = 0.001

        cg._prune_weak_edges()
        assert not cg.graph.has_edge("a", "b")
