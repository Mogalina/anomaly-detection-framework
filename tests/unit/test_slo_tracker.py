import time
import pytest


class TestSLOTrackerInit:
    """
    Validates SLOTracker initialization from config settings.
    """

    def test_default_init(self, config):
        """
        Verify that latency and error thresholds map correctly to class attributes.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        assert tracker.latency_p95_threshold == config["thresholding"]["slo"]["latency_p95_ms"]
        assert tracker.error_rate_threshold == config["thresholding"]["slo"]["error_rate_threshold"]


class TestRequestRecording:
    """
    Validates request latency and error counting records.
    """

    def test_record_single_request(self, config):
        """
        Verify recording a single request increments the service query counter.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        tracker.record_request("svc-a", 50.0)
        assert len(tracker.service_latencies["svc-a"]) == 1
        assert tracker.service_requests["svc-a"] == 1

    def test_record_error(self, config):
        """
        Verify recording a query failure appends a 1 to the errors buffer.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        tracker.record_request("svc-a", 50.0, is_error=True)
        assert list(tracker.service_errors["svc-a"]) == [1]

    def test_record_no_error(self, config):
        """
        Verify recording a successful query appends a 0 to the errors buffer.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        tracker.record_request("svc-a", 50.0, is_error=False)
        assert list(tracker.service_errors["svc-a"]) == [0]

    def test_multiple_requests(self, config):
        """
        Verify recording multiple requests increments request and latency buffers correctly.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for i in range(100):
            tracker.record_request("svc-a", float(i))
        assert tracker.service_requests["svc-a"] == 100
        assert len(tracker.service_latencies["svc-a"]) == 100


class TestSLOViolationDetection:
    """
    Validates latency percentile and error rate breach evaluations.
    """

    def test_latency_p95_violation(self, config):
        """
        Verify p95 latency breaches are flagged when p95 values exceed thresholds.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        violations = tracker.check_slo_violations("svc-a")
        types = [v["type"] for v in violations["violations"]]
        assert "latency_p95" in types

    def test_latency_p99_violation(self, config):
        """
        Verify p99 latency breaches are flagged when p99 values exceed thresholds.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        violations = tracker.check_slo_violations("svc-a")
        types = [v["type"] for v in violations["violations"]]
        assert "latency_p99" in types

    def test_no_violation_normal_latency(self, config):
        """
        Verify no violations are returned under low latency conditions.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 10.0)
        violations = tracker.check_slo_violations("svc-a")
        assert violations["violations"] == []

    def test_error_rate_violation(self, config):
        """
        Verify high error rates trigger error_rate breaches.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(20):
            tracker.record_request("svc-a", 10.0, is_error=True)
        violations = tracker.check_slo_violations("svc-a")
        types = [v["type"] for v in violations["violations"]]
        assert "error_rate" in types

    def test_no_violation_low_error_rate(self, config):
        """
        Verify low error rates do not trigger breaches.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(100):
            tracker.record_request("svc-a", 10.0, is_error=False)
        violations = tracker.check_slo_violations("svc-a")
        error_v = [v for v in violations["violations"] if v["type"] == "error_rate"]
        assert len(error_v) == 0

    def test_unknown_service_no_violations(self, config):
        """
        Verify asking for violations of unregistered services yields empty list results.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        violations = tracker.check_slo_violations("nonexistent")
        assert violations == {"violations": []}

    def test_insufficient_data_no_latency_check(self, config):
        """
        Verify percentile checks are skipped if latency samples are below minimum requirements.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(5):
            tracker.record_request("svc-a", 999.0)
        violations = tracker.check_slo_violations("svc-a")
        lat_v = [v for v in violations["violations"] if "latency" in v["type"]]
        assert len(lat_v) == 0

    def test_violation_stored_in_history(self, config):
        """
        Verify registered SLO breaches are stored in the historical log tracker.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        tracker.check_slo_violations("svc-a")
        assert len(tracker.slo_violations["svc-a"]) > 0


class TestSLOAnomalyCorrelation:
    """
    Validates temporal correlation of SLO breaches with anomaly alarms.
    """

    def test_correlation_found(self, config):
        """
        Verify temporal correlation is found when alarms and breaches happen close in time.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        tracker.check_slo_violations("svc-a")
        now = time.time()
        corr = tracker.correlate_with_anomaly("svc-a", now)
        assert corr["correlated"] is True

    def test_no_correlation_different_service(self, config):
        """
        Verify correlation fails if the alarm service name doesn't match the breach service.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        tracker.check_slo_violations("svc-a")
        corr = tracker.correlate_with_anomaly("svc-b", time.time())
        assert corr["correlated"] is False

    def test_no_correlation_outside_window(self, config):
        """
        Verify correlation fails if the anomaly timestamp falls outside the temporal window.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        tracker.check_slo_violations("svc-a")
        corr = tracker.correlate_with_anomaly("svc-a", time.time() + 99999, time_window=10)
        assert corr["correlated"] is False


class TestSLOServiceStatus:
    """
    Validates service latency and request stats status queries.
    """

    def test_status_for_known_service(self, config):
        """
        Verify status queries yield request count and p50 latency metrics.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 50.0)
        status = tracker.get_service_slo_status("svc-a")
        assert status["num_requests"] == 30
        assert status["latency_p50"] > 0

    def test_status_for_unknown_service(self, config):
        """
        Verify status queries for unregistered services yield empty dict results.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        assert tracker.get_service_slo_status("nonexistent") == {}


class TestSLOStatistics:
    """
    Validates general tracker statistics summaries.
    """

    def test_statistics(self, config):
        """
        Verify statistics summaries record tracked service counts and violation rates.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        for _ in range(30):
            tracker.record_request("svc-a", 999.0)
        tracker.check_slo_violations("svc-a")
        stats = tracker.get_statistics()
        assert stats["num_tracked_services"] >= 1
        assert stats["total_violations"] >= 1

    def test_empty_statistics(self, config):
        """
        Verify empty statistics summaries return zero values.

        Args:
            config: Test configuration fixture
        """
        from thresholding.slo_tracker import SLOTracker
        tracker = SLOTracker(config)
        stats = tracker.get_statistics()
        assert stats["total_requests"] == 0
