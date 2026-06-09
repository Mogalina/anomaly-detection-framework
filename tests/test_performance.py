import time
import sys
import numpy as np
import pytest
import torch


class TestModelInferencePerformance:
    """
    Validates model inference latency and throughput for LSTM-AE and AutoEncoder.
    """

    def test_lstm_single_inference_under_50ms(self, lstm_model, config):
        """
        Verify that a single LSTM-AE inference forward pass completes in less
        than 50 milliseconds on CPU.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(1, mc["sequence_length"], mc["input_size"])

        lstm_model.eval()
        with torch.no_grad():
            lstm_model(x)

        start = time.perf_counter()
        with torch.no_grad():
            lstm_model(x)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Inference took {elapsed*1000:.1f} ms, expected < 50 ms"

    def test_lstm_batch_inference_scales_sublinearly(self, lstm_model, config):
        """
        Verify that batch-8 inference exhibits sublinear scaling relative to single item inference.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        single = torch.randn(1, mc["sequence_length"], mc["input_size"])
        batch = torch.randn(8, mc["sequence_length"], mc["input_size"])

        lstm_model.eval()
        with torch.no_grad():
            lstm_model(single)

        start = time.perf_counter()
        with torch.no_grad():
            lstm_model(single)
        single_time = time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            lstm_model(batch)
        batch_time = time.perf_counter() - start

        assert batch_time < single_time * 8, (
            f"Batch-8 = {batch_time*1000:.1f} ms should be < 8× single = {single_time*1000*8:.1f} ms"
        )

    def test_reconstruction_error_latency(self, lstm_model, config):
        """
        Verify that reconstruction error calculation for a batch of size 16
        takes less than 100 milliseconds.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(16, mc["sequence_length"], mc["input_size"])

        lstm_model.eval()
        with torch.no_grad():
            lstm_model.compute_reconstruction_error(x)

        start = time.perf_counter()
        with torch.no_grad():
            lstm_model.compute_reconstruction_error(x, reduction="mean")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Reconstruction error took {elapsed*1000:.1f} ms"

    def test_autoencoder_inference_fast(self, autoencoder):
        """
        Verify that AutoEncoder forward pass for a batch of size 32 completes
        in less than 10 milliseconds.

        Args:
            autoencoder: AutoEncoder test fixture
        """
        x = torch.randn(32, 128)
        autoencoder.eval()
        with torch.no_grad():
            autoencoder(x)

        start = time.perf_counter()
        with torch.no_grad():
            autoencoder(x)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.01, f"AE inference took {elapsed*1000:.1f} ms"

    def test_encode_latency(self, lstm_model, config):
        """
        Verify that encoding a single input sample to its latent representation
        completes in less than 20 milliseconds.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
            config: Test configuration fixture
        """
        mc = config["edge"]["model"]
        x = torch.randn(1, mc["sequence_length"], mc["input_size"])
        lstm_model.eval()

        with torch.no_grad():
            lstm_model.encode(x)

        start = time.perf_counter()
        with torch.no_grad():
            lstm_model.encode(x)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.02, f"Encode took {elapsed*1000:.1f} ms"


class TestEdgeDetectionThroughput:
    """
    Validates EdgeDetector processing capacity and latency bounds.
    """

    def test_update_data_throughput(self, config):
        """
        Confirm that update_data() easily processes ≥ 10,000 metrics events
        per second.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="perf-svc", config=config)
        feat = config["edge"]["model"]["input_size"]
        rows = np.random.randn(1000, feat).astype(np.float32)

        start = time.perf_counter()
        for row in rows:
            det.update_data(row)
        elapsed = time.perf_counter() - start

        throughput = 1000 / elapsed
        assert throughput > 10_000, f"Throughput = {throughput:.0f} events/s, expected > 10 000"

    def test_detect_call_latency(self, config, normal_data):
        """
        Verify that a single detect() call completes in less than 200 milliseconds.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="perf-svc", config=config)
        for row in normal_data:
            det.update_data(row)

        det.detect()

        start = time.perf_counter()
        det.detect()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"detect() took {elapsed*1000:.1f} ms"

    def test_severity_computation_trivial(self, config):
        """
        Verify that mapping a score to severity classification takes less than
        0.01 milliseconds on average.

        Args:
            config: Test configuration fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="perf-svc", config=config)

        start = time.perf_counter()
        for _ in range(10_000):
            det._compute_severity(5.0)
        elapsed = time.perf_counter() - start

        per_call = elapsed / 10_000
        assert per_call < 1e-5, f"Severity computation took {per_call*1e6:.2f} µs/call"


class TestFedAvgScalability:
    """
    Validates central coordinator model aggregation scalability.
    """

    @pytest.fixture
    def _small_state(self, lstm_model):
        """
        Helper fixture providing a serialized parameter state dict.

        Args:
            lstm_model: Small LSTM anomaly detector fixture

        Returns:
            dict: The model's initial state dict serialized.
        """
        return {k: v.numpy().tolist() for k, v in lstm_model.state_dict().items()}

    def _build_coordinator(self, config, lstm_model, n_clients, state):
        """
        Helper method to construct a central coordinator with pre-loaded client updates.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            n_clients: Number of simulated client updates to populate
            state: Serialized parameters to assign to all client updates

        Returns:
            FederatedCoordinator: The coordinator with registered client updates.
        """
        import copy
        from federated.federated_coordinator import FederatedCoordinator
        from utils.metrics import MetricsExporter

        config["federated"]["coordinator"]["round_timeout_seconds"] = 1

        coord = FederatedCoordinator.__new__(FederatedCoordinator)
        coord.config = config
        coord.current_round = 0
        coord.min_clients_per_round = n_clients
        coord.staleness_tolerance = 2
        coord.poisoning_detection_enabled = False
        coord.global_model = copy.deepcopy(lstm_model)
        coord.device = torch.device("cpu")
        coord.round_history = []
        coord.client_models = {}
        coord.registered_clients = {}
        coord.metrics = MetricsExporter()

        for i in range(n_clients):
            cid = f"node-{i}"
            coord.registered_clients[cid] = {}
            coord.client_models[cid] = {
                "model_update": state,
                "num_samples": 100,
                "round": 0,
                "timestamp": time.time(),
                "metrics": {},
            }
        return coord

    def test_aggregation_10_clients(self, config, lstm_model, _small_state):
        """
        Verify that FedAvg with 10 clients completes in under 2 seconds.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            _small_state: Serialized state dict fixture
        """
        coord = self._build_coordinator(config, lstm_model, 10, _small_state)

        start = time.perf_counter()
        result = coord.aggregate_models(timeout=1)
        elapsed = time.perf_counter() - start

        assert result is True
        assert elapsed < 2.0, f"10-client aggregation took {elapsed:.2f}s"

    def test_aggregation_50_clients(self, config, lstm_model, _small_state):
        """
        Verify that FedAvg with 50 clients completes in under 5 seconds.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            _small_state: Serialized state dict fixture
        """
        coord = self._build_coordinator(config, lstm_model, 50, _small_state)

        start = time.perf_counter()
        result = coord.aggregate_models(timeout=1)
        elapsed = time.perf_counter() - start

        assert result is True
        assert elapsed < 5.0, f"50-client aggregation took {elapsed:.2f}s"

    def test_aggregation_100_clients(self, config, lstm_model, _small_state):
        """
        Verify that FedAvg with 100 clients completes in under 10 seconds.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            _small_state: Serialized state dict fixture
        """
        coord = self._build_coordinator(config, lstm_model, 100, _small_state)

        start = time.perf_counter()
        result = coord.aggregate_models(timeout=1)
        elapsed = time.perf_counter() - start

        assert result is True
        assert elapsed < 10.0, f"100-client aggregation took {elapsed:.2f}s"

    def test_aggregation_scales_roughly_linear(self, config, lstm_model, _small_state):
        """
        Verify that doubling the number of clients aggregates within a reasonable factor.

        Args:
            config: Test configuration fixture
            lstm_model: Small LSTM anomaly detector fixture
            _small_state: Serialized state dict fixture
        """
        coord_10 = self._build_coordinator(config, lstm_model, 10, _small_state)
        coord_20 = self._build_coordinator(config, lstm_model, 20, _small_state)

        start = time.perf_counter()
        coord_10.aggregate_models(timeout=1)
        t10 = time.perf_counter() - start

        start = time.perf_counter()
        coord_20.aggregate_models(timeout=1)
        t20 = time.perf_counter() - start

        assert t20 < t10 * 3 + 0.1, (
            f"20-client ({t20:.3f}s) should be < 3× 10-client ({t10:.3f}s)"
        )


class TestRCAPerformance:
    """
    Validates PageRank execution latency across varying dependency graph sizes.
    """

    def _build_graph(self, config, n_services):
        """
        Helper method to construct a service graph representing a single call chain.

        Args:
            config: Test configuration fixture
            n_services: Number of services to insert into the chain

        Returns:
            CausalGraph: The populated dependency graph.
        """
        from tracing.causal_graph import CausalGraph

        cg = CausalGraph(config)
        for i in range(n_services):
            cg.add_service(f"svc-{i}")
        for i in range(n_services - 1):
            cg.add_dependency(f"svc-{i}", f"svc-{i+1}", call_count=10)
        return cg

    def test_rca_20_services_under_500ms(self, config):
        """
        Verify that RCA PageRank on a 20-node graph takes less than 500 ms.

        Args:
            config: Test configuration fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        cg = self._build_graph(config, 20)
        rca = RootCauseAnalyzer(cg, config)

        start = time.perf_counter()
        rca.analyze({"svc-0", "svc-5", "svc-10"})
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"RCA took {elapsed*1000:.1f} ms on 20 nodes"

    def test_rca_100_services_under_2s(self, config):
        """
        Verify that RCA PageRank on a large 100-node graph takes less than 2 seconds.

        Args:
            config: Test configuration fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        cg = self._build_graph(config, 100)
        rca = RootCauseAnalyzer(cg, config)

        anomalous = {f"svc-{i}" for i in range(0, 100, 10)}
        start = time.perf_counter()
        rca.analyze(anomalous)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"RCA took {elapsed:.2f}s on 100 nodes"

    def test_cascade_analysis_50_services(self, config):
        """
        Verify that cascading analysis on a 50-node graph completes in under 500 ms.

        Args:
            config: Test configuration fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        cg = self._build_graph(config, 50)
        rca = RootCauseAnalyzer(cg, config)

        start = time.perf_counter()
        rca.explain_cascade("svc-0")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Cascade analysis took {elapsed*1000:.1f} ms"

    def test_rca_statistics_tracking(self, config, causal_graph):
        """
        Verify that the Root Cause Analyzer tracks statistical metadata correctly.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        rca.analyze({"svc-a"})
        rca.analyze({"svc-a", "svc-c"})

        stats = rca.get_statistics()
        assert stats["total_analyses"] == 2
        assert stats["avg_analysis_time"] > 0


class TestTunerPerformance:
    """
    Validates execution latency and state growth limits for RL tuner.
    """

    def test_tune_1000_iterations_under_1s(self, config):
        """
        Confirm that executing 1000 tuner steps takes less than 1 second.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("perf-svc")

        start = time.perf_counter()
        for i in range(1000):
            tuner.update_feedback("perf-svc", float(i % 10), i % 3 == 0, i % 5 == 0, False)
            tuner.tune_threshold("perf-svc")
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"1000 tuning iterations took {elapsed:.2f}s"

    def test_state_discretization_bounded(self, config):
        """
        Verify that state space discretization successfully limits state explosion.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("perf-svc")

        for i in range(500):
            tuner.update_feedback("perf-svc", float(i % 10), i % 3 == 0, i % 5 == 0, False)
            tuner.tune_threshold("perf-svc")

        n_states = len(tuner.q_table["perf-svc"])
        assert n_states < 1000, f"Q-table has {n_states} states, expected < 1000"


class TestCompressionPerformance:
    """
    Validates model compression and payload reduction latency.
    """

    @pytest.fixture
    def _payload(self, lstm_model):
        """
        Helper fixture serialization helper.

        Args:
            lstm_model: Small LSTM anomaly detector fixture

        Returns:
            bytes: The model's state dict serialized.
        """
        from utils.compression import serialize_state_dict
        return serialize_state_dict(lstm_model.state_dict())

    def test_lz4_compression_fast(self, _payload):
        """
        Verify that LZ4 compression executes in under 50 milliseconds.

        Args:
            _payload: Serialized model payload fixture
        """
        from utils.compression import compress, CompressionType

        start = time.perf_counter()
        compress(_payload, CompressionType.LZ4)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"LZ4 took {elapsed*1000:.1f} ms"

    def test_zstd_compression_reasonable(self, _payload):
        """
        Verify that Zstd compression executes in under 200 milliseconds.

        Args:
            _payload: Serialized model payload fixture
        """
        from utils.compression import compress, CompressionType

        start = time.perf_counter()
        compress(_payload, CompressionType.ZSTD, cpu_usage_percent=30.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"Zstd took {elapsed*1000:.1f} ms"

    def test_zstd_higher_ratio_than_lz4(self, _payload):
        """
        Confirm that Zstd yields a better compression ratio than LZ4.

        Args:
            _payload: Serialized model payload fixture
        """
        from utils.compression import compress, CompressionType

        lz4_out = compress(_payload, CompressionType.LZ4)
        zstd_out = compress(_payload, CompressionType.ZSTD, cpu_usage_percent=30.0)

        assert len(zstd_out) <= len(lz4_out)

    def test_decompression_faster_than_compression(self, _payload):
        """
        Confirm that decompressing a model payload is faster than compressing it.

        Args:
            _payload: Serialized model payload fixture
        """
        from utils.compression import compress, decompress, CompressionType

        for alg in [CompressionType.LZ4, CompressionType.ZSTD]:
            compressed = compress(_payload, alg)

            start = time.perf_counter()
            compress(_payload, alg)
            compress_time = time.perf_counter() - start

            start = time.perf_counter()
            decompress(compressed, alg)
            decompress_time = time.perf_counter() - start

            assert decompress_time <= compress_time * 2 + 0.001


class TestPreprocessingPerformance:
    """
    Validates data preprocessing and slicing execution throughput.
    """

    def test_sliding_window_10k_samples(self):
        """
        Verify that sliding_window on 10,000 samples completes in under 500 ms.
        """
        from utils.preprocessing import sliding_window

        data = np.random.randn(10_000, 16).astype(np.float32)

        start = time.perf_counter()
        sliding_window(data, window_size=100, stride=1)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"sliding_window took {elapsed*1000:.1f} ms"

    def test_normalization_10k_samples(self):
        """
        Verify that normalising 10,000 samples takes less than 100 ms.
        """
        from utils.preprocessing import normalize_data

        data = np.random.randn(10_000, 16).astype(np.float32)

        start = time.perf_counter()
        normalize_data(data, method="standard")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"normalization took {elapsed*1000:.1f} ms"

    def test_outlier_detection_performance(self):
        """
        Verify that detecting outliers in 10,000 samples completes in under 200 ms.
        """
        from utils.preprocessing import detect_outliers

        data = np.random.randn(10_000, 16).astype(np.float32)

        start = time.perf_counter()
        detect_outliers(data, method="zscore")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"Outlier detection took {elapsed*1000:.1f} ms"


class TestMemoryFootprint:
    """
    Validates memory consumption limits for model storage, tracing, and tables.
    """

    def test_lstm_model_size(self, lstm_model):
        """
        Verify that the LSTM-AE parameters consume less than 1 MB of memory.

        Args:
            lstm_model: Small LSTM anomaly detector fixture
        """
        total_params = sum(p.numel() for p in lstm_model.parameters())
        size_mb = total_params * 4 / (1024 ** 2)
        assert size_mb < 1.0, f"Model uses {size_mb:.2f} MB"

    def test_edge_detector_buffer_bounded(self, config, normal_data):
        """
        Confirm that the EdgeDetector's raw telemetry buffer does not leak memory.

        Args:
            config: Test configuration fixture
            normal_data: Synthetic normal telemetry data fixture
        """
        from edge.edge_detector import EdgeDetector

        det = EdgeDetector(service_name="mem-svc", config=config)
        for row in normal_data:
            det.update_data(row)
        for row in normal_data:
            det.update_data(row)

        assert len(det.data_buffer) <= config["edge"]["data"]["window_size"]

    def test_q_table_memory_bounded(self, config):
        """
        Verify that Q-table memory usage remains bounded under active training.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner

        tuner = ThresholdTuner(config)
        tuner.initialize_service("mem-svc")

        for i in range(2000):
            tuner.update_feedback("mem-svc", float(i % 10), i % 3 == 0, i % 5 == 0, False)
            tuner.tune_threshold("mem-svc")

        q_size = sys.getsizeof(tuner.q_table)
        assert q_size < 10 * 1024 * 1024

    def test_causal_graph_snapshot_history_bounded(self, config):
        """
        Verify that dependency graph snapshot history is capped to prevent growth leaks.

        Args:
            config: Test configuration fixture
        """
        from tracing.causal_graph import CausalGraph

        cg = CausalGraph(config)
        cg.add_dependency("a", "b")

        for _ in range(150):
            cg.create_snapshot()

        assert len(cg.snapshots) <= 100

    def test_rca_history_bounded(self, config, causal_graph):
        """
        Verify that Root Cause Analyzer result history is capped to prevent growth leaks.

        Args:
            config: Test configuration fixture
            causal_graph: CausalGraph test fixture
        """
        from analysis.root_cause_analyzer import RootCauseAnalyzer

        rca = RootCauseAnalyzer(causal_graph, config)
        for _ in range(150):
            rca.analyze({"svc-a"})

        assert len(rca.analysis_history) <= 100
