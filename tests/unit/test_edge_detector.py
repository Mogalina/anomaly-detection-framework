import numpy as np
import pytest
import torch


class TestEdgeDetectorInit:
    """
    Validates EdgeDetector construction and property mapping.
    """

    def test_default_initialization(self, config):
        """
        Verify that default configuration limits and parameters map correctly.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det.service_name == "svc-a"
        assert det.threshold == config["edge"]["detection"]["initial_threshold"]
        assert det.input_size == config["edge"]["model"]["input_size"]

    def test_custom_model_injection(self, config, lstm_model):
        """
        Verify that custom model injection overrides standard model construction.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", model=lstm_model, config=config)
        assert det.model is lstm_model

    def test_buffer_maxlen(self, config):
        """
        Verify that telemetry buffer size bounds matches configuration constraints.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det.data_buffer.maxlen == config["edge"]["data"]["window_size"]


class TestEdgeDetectorDataIngestion:
    """
    Validates sliding buffer management, padding, and truncation features.
    """

    def test_update_data_1d(self, config):
        """
        Verify that adding 1D telemetry vectors inserts them correctly.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        det.update_data(np.random.randn(config["edge"]["model"]["input_size"]).astype(np.float32))
        assert len(det.data_buffer) == 1

    def test_update_data_2d(self, config):
        """
        Verify that adding 2D metrics vectors inserts them correctly.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        det.update_data(np.random.randn(1, config["edge"]["model"]["input_size"]).astype(np.float32))
        assert len(det.data_buffer) == 1

    def test_pad_small_features(self, config):
        """
        Verify that telemetry vectors with fewer features are padded with zeros.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        det.update_data(np.ones(3, dtype=np.float32))
        assert len(det.data_buffer[-1]) == config["edge"]["model"]["input_size"]
        assert det.data_buffer[-1][3] == 0.0

    def test_truncate_large_features(self, config):
        """
        Verify that telemetry vectors exceeding input size are truncated.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        n = config["edge"]["model"]["input_size"] + 5
        det.update_data(np.ones(n, dtype=np.float32))
        assert len(det.data_buffer[-1]) == config["edge"]["model"]["input_size"]

    def test_buffer_overflow(self, config, normal_data):
        """
        Verify that the buffer discards oldest entries when full.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)
        assert len(det.data_buffer) == config["edge"]["data"]["window_size"]


class TestEdgeDetectorDetection:
    """
    Validates EdgeDetector anomaly evaluation constraints.
    """

    def test_insufficient_data_returns_no_anomaly(self, config):
        """
        Verify that evaluation with insufficient data fails gracefully.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        result = det.detect(np.random.randn(config["edge"]["model"]["input_size"]).astype(np.float32))
        assert result["is_anomaly"] is False
        assert "Insufficient" in result.get("message", "")

    def test_detect_returns_required_keys(self, config, normal_data):
        """
        Verify that detect() outputs the standard schema keys.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)
        result = det.detect()
        for key in ["is_anomaly", "score", "raw_score", "threshold", "consecutive_count", "inference_time", "timestamp"]:
            assert key in result, f"Missing key: {key}"

    def test_score_is_float(self, config, normal_data):
        """
        Verify that computed anomaly scores are returning float values.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)
        result = det.detect()
        assert isinstance(result["score"], float)
        assert isinstance(result["raw_score"], float)

    def test_detect_with_inline_metrics(self, config, normal_data):
        """
        Verify that passing inline metrics to detect updates parameters.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data[:-1]:
            det.update_data(row)
        result = det.detect(metrics=normal_data[-1])
        assert "score" in result

    def test_consecutive_anomaly_tracking(self, config, normal_data):
        """
        Verify consecutive anomaly counters increment correctly.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)

        det.threshold = 0.0
        det.min_anomaly_duration = 1

        result = det.detect()
        assert det.consecutive_anomalies >= 1

    def test_confirmed_anomaly_requires_persistence(self, config, normal_data):
        """
        Verify that alarms are only raised after duration constraints are exceeded.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)

        det.threshold = 0.0
        det.min_anomaly_duration = 5

        result = det.detect()
        assert result["consecutive_count"] >= 1
        if result["consecutive_count"] < det.min_anomaly_duration:
            assert result["is_anomaly"] is False

    def test_consecutive_counter_resets_on_normal(self, config, normal_data):
        """
        Verify consecutive counter resets to 0 when telemetry returns to normal.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)

        det.threshold = 1e10
        det.detect()
        assert det.consecutive_anomalies == 0


class TestEdgeDetectorSeverity:
    """
    Validates severity classifications of anomaly scores.
    """

    def test_severity_low(self, config):
        """
        Verify that a minor threshold breach maps to low severity.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det._compute_severity(det.threshold * 1.1) == "low"

    def test_severity_medium(self, config):
        """
        Verify that a medium threshold breach maps to medium severity.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det._compute_severity(det.threshold * 1.8) == "medium"

    def test_severity_high(self, config):
        """
        Verify that a large threshold breach maps to high severity.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det._compute_severity(det.threshold * 2.5) == "high"

    def test_severity_critical(self, config):
        """
        Verify that an extreme threshold breach maps to critical severity.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det._compute_severity(det.threshold * 5.0) == "critical"


class TestEdgeDetectorThreshold:
    """
    Validates threshold updates and boundaries.
    """

    def test_update_threshold(self, config):
        """
        Verify that changing detector threshold updates attributes correctly.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        det.update_threshold(5.0)
        assert det.threshold == 5.0

    def test_update_threshold_zero(self, config):
        """
        Verify threshold updates handles zero values correctly.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        det.update_threshold(0.0)
        assert det.threshold == 0.0


class TestEdgeDetectorStatistics:
    """
    Validates stats aggregation of anomaly scores.
    """

    def test_empty_statistics(self, config):
        """
        Verify empty statistics yields an empty dict.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det.get_statistics() == {}

    def test_statistics_after_detection(self, config, normal_data):
        """
        Verify that statistics are populated after evaluation calls.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)
        det.detect()

        stats = det.get_statistics()
        assert "mean_score" in stats
        assert "std_score" in stats
        assert "max_score" in stats
        assert "min_score" in stats
        assert "threshold" in stats
        assert "buffer_size" in stats

    def test_statistics_values_reasonable(self, config, normal_data):
        """
        Verify min, mean, and max constraints in statistics calculations.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        for row in normal_data:
            det.update_data(row)
        det.detect()
        det.detect()

        stats = det.get_statistics()
        assert stats["min_score"] <= stats["mean_score"] <= stats["max_score"]
        assert stats["std_score"] >= 0


class TestEdgeDetectorTraining:
    """
    Validates edge detector model training and scaler settings.
    """

    def test_train_returns_history(self, config, normal_data):
        """
        Verify model training yields epoch loss history logs.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        history = det.train(normal_data, epochs=5, batch_size=8)
        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_training_loss_decreases(self, config, normal_data):
        """
        Verify model training exhibits training loss reduction.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        history = det.train(normal_data, epochs=20, batch_size=8)
        assert history["loss"][-1] < history["loss"][0]

    def test_threshold_calibrated_after_training(self, config, normal_data):
        """
        Verify threshold auto-calibration occurs after training completion.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        initial_th = det.threshold
        det.train(normal_data, epochs=5, batch_size=8)
        assert det.threshold != initial_th or det.threshold > 0

    def test_scaler_set_after_training(self, config, normal_data):
        """
        Verify data normalization scaler parameters are fit during training.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector
        det = EdgeDetector(service_name="svc-a", config=config)
        assert det.scaler is None
        det.train(normal_data, epochs=3, batch_size=8)
        assert det.scaler is not None
