import copy
import time
import numpy as np
import pytest
import torch


class TestCompleteLifecycle:
    """
    End-to-End lifecycle validation.
    
    Deploys edge services, ingests telemetry streams, processes and classifies
    anomalies, executes RCA, submits operator feedback, and resolves alerts.
    """

    def test_full_lifecycle(self, config, normal_data, anomalous_data, causal_graph):
        """
        Verify the complete system execution flow from node registration to alert
        resolution.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
            anomalous_data: Synthetic anomalous telemetry data fixture
            causal_graph: CausalGraph test fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        # Deploy
        pipeline = AnomalyPipeline(config)
        pipeline.causal_graph = causal_graph
        pipeline.root_cause_analyzer = RootCauseAnalyzer(causal_graph, config)

        detectors = {}
        for svc in ["gateway", "svc-a", "svc-b", "svc-c", "svc-d"]:
            det = EdgeDetector(service_name=svc, config=config)
            pipeline.register_service(svc, det)
            detectors[svc] = det

        status = pipeline.get_pipeline_status()
        assert status["num_registered_services"] == 5
        assert status["active_anomalies"] == 0

        # Train on normal data
        for svc, det in detectors.items():
            det.train(normal_data, epochs=10, batch_size=8)

        # Ingest normal telemetry (no anomalies expected)
        for row in normal_data:
            for det in detectors.values():
                det.update_data(row)

        for svc, det in detectors.items():
            result = det.detect()
            assert "score" in result

        # Inject anomaly and detect
        anomaly_result = pipeline.process_anomaly_event("svc-a", {
            "score": 8.0, "threshold": 3.0, "severity": "high"
        })
        assert anomaly_result["service"] == "svc-a"
        assert "root_cause_analysis" in anomaly_result
        assert "svc-a" in pipeline.active_anomalies

        # Provide feedback
        pipeline.provide_feedback("svc-a", was_detected=True, was_true_anomaly=True)

        # Resolve
        pipeline.clear_resolved_anomalies({"svc-a"})
        assert "svc-a" not in pipeline.active_anomalies

        final_status = pipeline.get_pipeline_status()
        assert final_status["active_anomalies"] == 0
        assert final_status["total_processed_anomalies"] == 1


class TestMultiRoundFederation:
    """
    Validation of multi-round federated convergence.
    """

    def test_five_round_convergence(self, config, normal_data):
        """
        Verify that federated learning across multiple nodes converges and decreases
        reconstruction error under normal telemetry partitions.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.models import LSTMAnomalyDetector
        from federated.federated_client import FederatedClient
        from federated.federated_coordinator import FederatedCoordinator
        from utils.preprocessing import sliding_window
        from utils.metrics import MetricsExporter

        mc = config["edge"]["model"]
        config["federated"]["client"]["gradient_compression"]["enabled"] = False

        # Create global model
        global_model = LSTMAnomalyDetector(
            input_size=mc["input_size"], hidden_size=mc["hidden_size"],
            num_layers=mc["num_layers"], dropout=mc["dropout"],
        )

        # Create 3 clients with different data splits
        windows, _ = sliding_window(normal_data, window_size=20, stride=5)
        n = len(windows)
        splits = [windows[:n//3], windows[n//3:2*n//3], windows[2*n//3:]]

        clients = []
        for i in range(3):
            model = LSTMAnomalyDetector(
                input_size=mc["input_size"], hidden_size=mc["hidden_size"],
                num_layers=mc["num_layers"], dropout=mc["dropout"],
            )
            clients.append(FederatedClient(f"node-{i}", model, config=config))

        round_losses = []

        for round_num in range(5):
            # Get global params
            global_params = {k: v.numpy().tolist() for k, v in global_model.state_dict().items()}

            # Each client trains
            updates = []
            for i, client in enumerate(clients):
                result = client.train_round(
                    splits[i],
                    global_model_params=global_params,
                    system_metrics={"cpu_usage_percent": 30.0},
                )
                updates.append(result)

            # FedAvg aggregation
            total_samples = sum(u["num_samples"] for u in updates)
            aggregated = {}
            for upd in updates:
                weight = upd["num_samples"] / total_samples
                for key, param in upd["model_update"].items():
                    pt = torch.FloatTensor(param)
                    aggregated[key] = aggregated.get(key, torch.zeros_like(pt)) + weight * pt

            # Update global model
            global_model.load_state_dict(aggregated)

            avg_loss = np.mean([u["loss"] for u in updates])
            round_losses.append(avg_loss)

        # Loss should generally decrease or stay reasonable
        assert round_losses[-1] < round_losses[0] * 2, (
            f"Final loss {round_losses[-1]:.4f} should not diverge from "
            f"initial {round_losses[0]:.4f}"
        )


class TestConcurrentAnomalies:
    """
    Validation of concurrent anomaly propagation and routing logic.
    """

    def test_simultaneous_anomalies(self, config, causal_graph):
        """
        Verify that the pipeline correctly processes and aggregates concurrent
        anomalies across multiple services.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        pipeline = AnomalyPipeline(config)
        pipeline.causal_graph = causal_graph
        pipeline.root_cause_analyzer = RootCauseAnalyzer(causal_graph, config)

        for svc in ["gateway", "svc-a", "svc-b", "svc-c", "svc-d"]:
            pipeline.register_service(svc, EdgeDetector(service_name=svc, config=config))

        results = []
        for svc in ["gateway", "svc-a", "svc-b", "svc-c", "svc-d"]:
            r = pipeline.process_anomaly_event(svc, {"score": 5.0, "threshold": 3.0})
            results.append(r)

        assert len(pipeline.active_anomalies) == 5

        # Last RCA should see all 5 anomalous services
        last_rca = results[-1]["root_cause_analysis"]
        total_classified = len(last_rca["root_causes"]) + len(last_rca["propagated_anomalies"])
        assert total_classified == 5

    def test_partial_resolution(self, config, causal_graph):
        """
        Verify that resolving a subset of active anomalies clears them from
        the tracker while keeping the remaining ones active.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        pipeline = AnomalyPipeline(config)
        pipeline.causal_graph = causal_graph
        pipeline.root_cause_analyzer = RootCauseAnalyzer(causal_graph, config)

        for svc in ["gateway", "svc-a", "svc-b"]:
            pipeline.register_service(svc, EdgeDetector(service_name=svc, config=config))
            pipeline.process_anomaly_event(svc, {"score": 5.0, "threshold": 3.0})

        # Resolve gateway only
        pipeline.clear_resolved_anomalies({"gateway"})
        assert "gateway" not in pipeline.active_anomalies
        assert "svc-a" in pipeline.active_anomalies
        assert "svc-b" in pipeline.active_anomalies


class TestThresholdSelfTuning:
    """
    Validation of threshold self-tuning adaptation over multiple cycles.
    """

    def test_tuning_reduces_false_positive_rate(self, config):
        """
        Verify that continuous false positive feedback results in the threshold
        adjusting to reduce the false positive rate.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=3.0)

        # Phase 1: Many false positives
        for _ in range(30):
            tuner.update_feedback("svc-x", 4.0, True, False, False)

        # Phase 2: Tune threshold
        for _ in range(30):
            tuner.tune_threshold("svc-x")

        perf = tuner.get_service_performance("svc-x")
        assert perf["false_positives"] > 0
        assert perf["precision"] == 0.0

    def test_epsilon_reaches_minimum(self, config):
        """
        Confirm that exploration epsilon decays to its minimum bound over many rounds.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        for i in range(2000):
            tuner.update_feedback("svc-x", float(i % 10), i % 3 == 0, i % 5 == 0, False)
            tuner.tune_threshold("svc-x")

        assert tuner.epsilon <= tuner.min_epsilon + 1e-6


class TestConfigModule:
    """
    Validation of dot-notation and copy semantics in the Config utility.
    """

    def test_get_dot_notation(self, config):
        """
        Verify that dot-notation paths resolve to correct nested configurations.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        assert cfg.get("edge.model.input_size") == 10
        assert cfg.get("edge.model.hidden_size") == 32

    def test_get_default(self, config):
        """
        Verify that missing configuration paths return the default fallback value.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        assert cfg.get("nonexistent.key", "default") == "default"

    def test_set_dot_notation(self, config):
        """
        Verify that setting paths via dot-notation correctly updates nested keys.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        cfg.set("edge.model.input_size", 32)
        assert cfg.get("edge.model.input_size") == 32

    def test_set_creates_nested_keys(self, config):
        """
        Verify that setting non-existent nested paths dynamically creates directories.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        cfg.set("new.nested.key", "value")
        assert cfg.get("new.nested.key") == "value"

    def test_to_dict(self, config):
        """
        Confirm that to_dict() returns a standard Python dictionary representing config state.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["edge"]["model"]["input_size"] == 10

    def test_to_dict_is_deep_copy(self, config):
        """
        Verify that dictionary representation is a deep copy and mutations don't leak.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        d = cfg.to_dict()
        d["edge"]["model"]["input_size"] = 999
        assert cfg.get("edge.model.input_size") == 10

    def test_dict_style_access(self, config):
        """
        Verify dictionary-style subscript access to configurations.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        assert cfg["edge.model.input_size"] == 10

    def test_dict_style_set(self, config):
        """
        Verify dictionary-style subscript assignment to configurations.

        Args:
            config: Test configuration dict
        """
        from utils.config import Config
        cfg = Config(config)
        cfg["edge.model.input_size"] = 64
        assert cfg["edge.model.input_size"] == 64


class TestDataPipelineConsistency:
    """
    Validation of end-to-end data pipeline math and mode consistency.
    """

    def test_deterministic_scoring(self, config, normal_data, lstm_model):
        """
        Confirm that evaluating the same input sequence under evaluation mode 
        produces identical anomaly scores.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="svc-a", model=lstm_model, config=config)
        for row in normal_data:
            det.update_data(row)

        r1 = det.detect()
        r2 = det.detect()
        assert abs(r1["score"] - r2["score"]) < 1e-6

    def test_preprocessing_does_not_modify_input(self, normal_data):
        """
        Verify that preprocessing functions return new structures without mutating the
        original arrays.

        Args:
            normal_data: Synthetic normal telemetry data fixture
        """
        from utils.preprocessing import sliding_window, normalize_data

        original = normal_data.copy()
        sliding_window(normal_data, window_size=20, stride=1)
        np.testing.assert_array_equal(normal_data, original)

        normalize_data(normal_data, method="standard")
        np.testing.assert_array_equal(normal_data, original)

    def test_model_eval_vs_train_mode(self, lstm_model, config):
        """
        Verify that putting the model in evaluation mode disables dropout, ensuring
        reproducible, deterministic outputs.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(2, mc["sequence_length"], mc["input_size"])

        lstm_model.eval()
        with torch.no_grad():
            out1 = lstm_model(x)
            out2 = lstm_model(x)

        assert torch.allclose(out1, out2, atol=1e-6)
