import time
import numpy as np
import pytest


class TestPipelineSetup:
    """
    Validates pipeline initialization and edge service registration.
    """

    def test_pipeline_initializes(self, config):
        """
        Verify that the pipeline initializes with default inactive state.

        Args:
            config: Test configuration fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        pipeline = AnomalyPipeline(config)
        assert pipeline.is_running is False
        assert len(pipeline.edge_detectors) == 0

    def test_register_service(self, config):
        """
        Verify that registering a service registers its detector and inserts the
        service node into the causal graph.

        Args:
            config: Test configuration fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector

        pipeline = AnomalyPipeline(config)
        det = EdgeDetector(service_name="svc-a", config=config)
        pipeline.register_service("svc-a", det)
        assert "svc-a" in pipeline.edge_detectors
        assert "svc-a" in pipeline.causal_graph.graph.nodes

    def test_register_multiple_services(self, config):
        """
        Verify that the pipeline handles multiple edge service registrations.

        Args:
            config: Test configuration fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector

        pipeline = AnomalyPipeline(config)
        for svc in ["gateway", "svc-a", "svc-b", "svc-c"]:
            det = EdgeDetector(service_name=svc, config=config)
            pipeline.register_service(svc, det)
        assert len(pipeline.edge_detectors) == 4


class TestAnomalyProcessing:
    """
    Validates complete anomaly event processing through the orchestration pipeline.
    """

    def _setup_pipeline(self, config, causal_graph):
        """
        Helper method to instantiate and register a pre-configured pipeline.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture

        Returns:
            AnomalyPipeline: The initialized pipeline instance.
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        pipeline = AnomalyPipeline(config)
        pipeline.causal_graph = causal_graph
        pipeline.root_cause_analyzer = RootCauseAnalyzer(causal_graph, config)

        for svc in ["gateway", "svc-a", "svc-b", "svc-c", "svc-d"]:
            det = EdgeDetector(service_name=svc, config=config)
            pipeline.register_service(svc, det)
        return pipeline

    def test_process_single_anomaly(self, config, causal_graph):
        """
        Verify that processing a single anomaly event yields correct explanations
        and SLO correlation data.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        result = pipeline.process_anomaly_event("svc-a", {
            "score": 5.0, "threshold": 3.0, "severity": "high"
        })
        assert result["service"] == "svc-a"
        assert result["anomaly_score"] == 5.0
        assert "root_cause_analysis" in result
        assert "explanation" in result
        assert "slo_correlation" in result

    def test_process_updates_active_anomalies(self, config, causal_graph):
        """
        Confirm that incoming anomalies update the pipeline's active anomaly map.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})
        assert "svc-a" in pipeline.active_anomalies

    def test_process_marks_graph_anomaly(self, config, causal_graph):
        """
        Verify that processed anomalies flag the service node in the causal graph.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        pipeline.process_anomaly_event("svc-b", {"score": 5.0, "threshold": 3.0})
        assert "svc-b" in causal_graph.anomalous_services

    def test_process_multiple_anomalies(self, config, causal_graph):
        """
        Verify that multiple concurrent anomalies are processed independently.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})
        pipeline.process_anomaly_event("svc-c", {"score": 4.0, "threshold": 3.0})
        assert len(pipeline.active_anomalies) == 2
        assert len(pipeline.anomaly_history) == 2

    def test_anomaly_history_grows(self, config, causal_graph):
        """
        Verify that anomaly events are archived sequentially in the history log.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        for i in range(5):
            pipeline.process_anomaly_event("svc-a", {"score": float(i+1), "threshold": 3.0})
        assert len(pipeline.anomaly_history) == 5

    def test_rca_runs_with_accumulating_anomalies(self, config, causal_graph):
        """
        Confirm that RCA analyzes the joint probability context of all active anomalies.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)

        r1 = pipeline.process_anomaly_event("gateway", {"score": 5.0, "threshold": 3.0})
        r2 = pipeline.process_anomaly_event("svc-a", {"score": 4.0, "threshold": 3.0})
        r3 = pipeline.process_anomaly_event("svc-c", {"score": 3.5, "threshold": 3.0})

        rca = r3["root_cause_analysis"]
        total = len(rca["root_causes"]) + len(rca["propagated_anomalies"])
        assert total == 3


class TestAnomalyResolution:
    """
    Validates anomaly resolution, state cleanup, and graph resetting.
    """

    def _setup_pipeline(self, config, causal_graph):
        """
        Helper method to instantiate and register a pre-configured pipeline.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture

        Returns:
            AnomalyPipeline: The initialized pipeline instance.
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        pipeline = AnomalyPipeline(config)
        pipeline.causal_graph = causal_graph
        pipeline.root_cause_analyzer = RootCauseAnalyzer(causal_graph, config)
        for svc in ["gateway", "svc-a", "svc-b"]:
            det = EdgeDetector(service_name=svc, config=config)
            pipeline.register_service(svc, det)
        return pipeline

    def test_clear_resolved(self, config, causal_graph):
        """
        Verify that resolving specific anomalies clears them from active tracking.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})
        pipeline.process_anomaly_event("svc-b", {"score": 4.0, "threshold": 3.0})
        assert len(pipeline.active_anomalies) == 2

        pipeline.clear_resolved_anomalies({"svc-a"})
        assert "svc-a" not in pipeline.active_anomalies
        assert "svc-b" in pipeline.active_anomalies

    def test_clear_removes_graph_anomaly_flag(self, config, causal_graph):
        """
        Verify that clearing an anomaly removes its active flag in the causal graph.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})
        pipeline.clear_resolved_anomalies({"svc-a"})
        assert "svc-a" not in causal_graph.anomalous_services

    def test_clear_all(self, config, causal_graph):
        """
        Verify that clearing all resolved anomalies fully resets pipeline state.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        pipeline = self._setup_pipeline(config, causal_graph)
        for svc in ["gateway", "svc-a", "svc-b"]:
            pipeline.process_anomaly_event(svc, {"score": 5.0, "threshold": 3.0})
        pipeline.clear_resolved_anomalies({"gateway", "svc-a", "svc-b"})
        assert len(pipeline.active_anomalies) == 0


class TestPipelineStatus:
    """
    Validates pipeline runtime status and metrics generation.
    """

    def test_initial_status(self, config):
        """
        Verify status reports correctly when no services are registered.

        Args:
            config: Test configuration fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        pipeline = AnomalyPipeline(config)
        status = pipeline.get_pipeline_status()
        assert status["num_registered_services"] == 0
        assert status["active_anomalies"] == 0

    def test_status_after_registration(self, config):
        """
        Verify status metrics include registration, graph, SLO, and tuner stats.

        Args:
            config: Test configuration fixture
        """
        from coordinator.anomaly_pipeline import AnomalyPipeline
        from edge.edge_detector import EdgeDetector

        pipeline = AnomalyPipeline(config)
        for svc in ["svc-a", "svc-b", "svc-c"]:
            pipeline.register_service(svc, EdgeDetector(service_name=svc, config=config))

        status = pipeline.get_pipeline_status()
        assert status["num_registered_services"] == 3
        assert "graph_stats" in status
        assert "slo_stats" in status
        assert "tuner_stats" in status

    def test_status_after_anomaly(self, config, causal_graph):
        """
        Verify status fields accurately reflect active and lifetime processed anomalies.

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
        pipeline.register_service("svc-a", EdgeDetector(service_name="svc-a", config=config))
        pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})

        status = pipeline.get_pipeline_status()
        assert status["active_anomalies"] == 1
        assert status["total_processed_anomalies"] == 1


class TestFeedbackLoop:
    """
    Validates telemetry feedback loop integration with RL-based threshold tuning.
    """

    def test_provide_feedback(self, config, causal_graph):
        """
        Verify that operator feedback is successfully routed to the threshold tuner.

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
        pipeline.register_service("svc-a", EdgeDetector(service_name="svc-a", config=config))

        pipeline.provide_feedback("svc-a", was_detected=True, was_true_anomaly=True)
        pipeline.provide_feedback("svc-a", was_detected=True, was_true_anomaly=False)

        assert len(pipeline.threshold_tuner.service_history["svc-a"]) == 2

    def test_threshold_updates_after_feedback(self, config, causal_graph):
        """
        Verify that accumulating multiple feedback events triggers active threshold updates.

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
        det = EdgeDetector(service_name="svc-a", config=config)
        pipeline.register_service("svc-a", det)

        for i in range(10):
            pipeline.provide_feedback("svc-a", was_detected=True, was_true_anomaly=(i % 2 == 0))

        assert isinstance(det.threshold, float)
