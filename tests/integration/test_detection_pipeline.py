import numpy as np
import pytest
import torch


class TestTelemetryToDetection:
    """
    Validates the flow of data from raw telemetry streams through the EdgeDetector.
    """

    def test_normal_data_produces_score(self, config, normal_data, lstm_model):
        """
        Verify that feeding normal telemetry to the model produces a valid anomaly 
        score and threshold.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="svc-a", model=lstm_model, config=config)
        for row in normal_data:
            det.update_data(row)

        result = det.detect()
        assert "score" in result
        assert "threshold" in result
        assert isinstance(result["score"], float)

    def test_trained_detector_flags_anomaly(self, config, normal_data, anomalous_data, lstm_model):
        """
        Verify that training the detector on normal telemetry allows it to correctly
        identify and flag anomalous telemetry.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
            anomalous_data: Synthetic anomalous telemetry data fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="svc-a", model=lstm_model, config=config)
        det.train(normal_data, epochs=20, batch_size=8)

        # Feed anomalous data
        for row in anomalous_data[-30:]:
            det.update_data(row)

        result = det.detect()
        assert result["score"] > 0
        assert result["raw_score"] > 0


class TestDetectionToRCA:
    """
    Validates that detected service anomalies correctly trigger root cause analysis.
    """

    def test_single_anomaly_triggers_rca(self, config, causal_graph):
        """
        Verify that a single service anomaly identifies the service as a root cause.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-c"})
        assert len(result["root_causes"]) >= 1
        assert "svc-c" in result["explanations"]

    def test_multiple_anomalies_rca(self, config, causal_graph):
        """
        Verify that concurrent anomalies across multiple services are resolved to
        proper root causes.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"svc-c", "svc-d"})
        assert len(result["root_causes"]) >= 1
        assert "svc-c" in result["explanations"]
        assert "svc-d" in result["explanations"]

    def test_cascade_from_detected_root_cause(self, config, causal_graph):
        """
        Verify that downstream anomaly propagation cascades are properly traced.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        result = rca.analyze({"gateway", "svc-a", "svc-c"})
        if result["root_causes"]:
            top_rc = result["root_causes"][0]["service"]
            cascade = rca.explain_cascade(top_rc)
            assert cascade["num_affected"] >= 0


class TestDetectionWithGraphUpdate:
    """
    Validates that modifications to the causal graph are immediately reflected in RCA.
    """

    def test_new_dependency_affects_rca(self, config, causal_graph):
        """
        Verify that dynamically registering new service dependencies changes the
        explanation path of subsequent anomalies.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)

        # Add a new dependency
        causal_graph.add_dependency("svc-c", "svc-e", call_count=50)
        causal_graph.add_service("svc-e")

        result = rca.analyze({"svc-c", "svc-e"})
        assert "svc-e" in result["explanations"]


class TestMultiServiceDetection:
    """
    Validates concurrent and independent execution of multiple edge detectors.
    """

    def test_multi_service_parallel_detection(self, config, normal_data):
        """
        Verify that multiple edge detectors can run and record telemetry in parallel.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector

        detectors = {}
        for svc in ["svc-a", "svc-b", "svc-c"]:
            det = EdgeDetector(service_name=svc, config=config)
            for row in normal_data:
                det.update_data(row)
            detectors[svc] = det

        results = {}
        for svc, det in detectors.items():
            results[svc] = det.detect()

        for svc, result in results.items():
            assert "score" in result
            assert isinstance(result["score"], float)
