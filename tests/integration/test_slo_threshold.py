import time
import pytest


class TestSLOViolationToCorrelation:
    """
    Validates that SLO violations are correctly correlated with detected anomalies.
    """

    def test_high_latency_correlates(self, config):
        """
        Verify that sustained high latency triggers SLO violations and correlates
        positively with active service anomalies.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker

        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)

        tracker.check_slo_violations("svc-a")
        corr = tracker.correlate_with_anomaly("svc-a", time.time())
        assert corr["correlated"] is True
        assert corr["correlation_strength"] >= 1

    def test_error_rate_correlates(self, config):
        """
        Verify that elevated error rates trigger SLO violations and correlate with anomalies.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker

        tracker = SLOTracker(config)
        for _ in range(20):
            tracker.record_request("svc-a", 10.0, is_error=True)

        tracker.check_slo_violations("svc-a")
        corr = tracker.correlate_with_anomaly("svc-a", time.time())
        assert corr["correlated"] is True

    def test_no_violation_no_correlation(self, config):
        """
        Verify that compliant latency and zero errors do not yield anomaly correlation.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker

        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 10.0)

        tracker.check_slo_violations("svc-a")
        corr = tracker.correlate_with_anomaly("svc-a", time.time())
        assert corr["correlated"] is False


class TestSLODrivenThresholdTuning:
    """
    Validates that SLO violations influence reinforcement learning reward signals.
    """

    def test_slo_violation_affects_reward(self, config):
        """
        Confirm that reward signals are penalized when concurrent SLO violations are detected.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        r_ok = tuner._compute_reward("svc-x", detected=True, true_anomaly=True, slo_violated=False)
        r_slo = tuner._compute_reward("svc-x", detected=True, true_anomaly=True, slo_violated=True)
        assert r_slo < r_ok, "SLO violation should reduce reward"

    def test_threshold_adapts_to_fp_slo(self, config):
        """
        Verify that many false positive feedback events coupled with SLO violations
        drive threshold adaptation.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=3.0)

        for _ in range(50):
            tuner.update_feedback("svc-x", 4.0, True, False, True)

        for _ in range(20):
            tuner.tune_threshold("svc-x")

        perf = tuner.get_service_performance("svc-x")
        assert perf["false_positives"] == 50

    def test_threshold_adapts_to_fn_slo(self, config):
        """
        Verify that many false negative feedback events coupled with SLO violations
        drive threshold adaptation.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=3.0)

        for _ in range(50):
            tuner.update_feedback("svc-x", 1.0, False, True, True)

        for _ in range(20):
            tuner.tune_threshold("svc-x")

        perf = tuner.get_service_performance("svc-x")
        assert perf["false_negatives"] == 50


class TestPipelineSLOIntegration:
    """
    Validates pipeline-wide integration of the SLO Tracker and Threshold Tuner.
    """

    def test_pipeline_slo_enrichment(self, config, causal_graph):
        """
        Verify that processed anomaly events are enriched with active SLO correlation metrics.

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

        for _ in range(30):
            pipeline.slo_tracker.record_request("svc-a", 999.0)
        pipeline.slo_tracker.check_slo_violations("svc-a")

        result = pipeline.process_anomaly_event("svc-a", {"score": 5.0, "threshold": 3.0})
        assert result["slo_correlation"]["correlated"] is True

    def test_pipeline_feedback_with_slo(self, config, causal_graph):
        """
        Verify that the feedback loop utilizes concurrent SLO status in reward updates.

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

        for _ in range(30):
            pipeline.slo_tracker.record_request("svc-a", 999.0)

        pipeline.provide_feedback("svc-a", was_detected=True, was_true_anomaly=True)
        assert len(pipeline.threshold_tuner.service_history["svc-a"]) == 1


class TestMultiServiceSLOScenario:
    """
    Validates independent tracking and tuning across multiple distinct services.
    """

    def test_independent_slo_tracking(self, config):
        """
        Confirm that SLO violations are tracked independently for separate services.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker

        tracker = SLOTracker(config)

        for _ in range(30):
            tracker.record_request("svc-a", 999.0)

        for _ in range(30):
            tracker.record_request("svc-b", 10.0)

        v_a = tracker.check_slo_violations("svc-a")
        v_b = tracker.check_slo_violations("svc-b")

        assert len(v_a["violations"]) > 0
        assert len(v_b["violations"]) == 0

    def test_independent_threshold_tuning(self, config):
        """
        Verify that distinct services adjust their thresholds independently based on
        individual feedback profiles.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-a", initial_threshold=3.0)
        tuner.initialize_service("svc-b", initial_threshold=3.0)

        for _ in range(30):
            tuner.update_feedback("svc-a", 4.0, True, False, False)

        for _ in range(30):
            tuner.update_feedback("svc-b", 4.0, True, True, False)

        perf_a = tuner.get_service_performance("svc-a")
        perf_b = tuner.get_service_performance("svc-b")

        assert perf_a["precision"] == 0.0
        assert perf_b["precision"] == 1.0
