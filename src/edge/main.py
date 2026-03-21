import argparse
import sys
import time
import numpy as np

from edge.edge_detector import EdgeDetector
from edge.models import LSTMAnomalyDetector
from utils.logger import setup_logger, get_logger
from utils.config import init_config
from utils.metrics import get_metrics
from utils.system_metrics_collector import SystemMetricsCollector


def _record_system_metrics_to_exporter(metrics_exporter, service: str, system_metrics: dict):
    """Record collected CPU, Memory, Disk, Network metrics to the Prometheus exporter."""
    try:
        metrics_exporter.set_edge_cpu_usage(service, "all", system_metrics.get("cpu_usage_percent", 0.0))
        metrics_exporter.set_edge_cpu_load(
            service,
            system_metrics.get("cpu_load1", 0.0),
            system_metrics.get("cpu_load5", 0.0),
        )
        metrics_exporter.set_edge_memory_usage(
            service,
            system_metrics.get("memory_usage_percent", 0.0),
            int(system_metrics.get("memory_available_mb", 0) * 1024 * 1024),
        )
        metrics_exporter.record_edge_disk_read_latency(
            service, "default", system_metrics.get("disk_read_latency_ms", 0.0)
        )
        metrics_exporter.record_edge_disk_write_latency(
            service, "default", system_metrics.get("disk_write_latency_ms", 0.0)
        )
        metrics_exporter.set_edge_network_error_rates(
            service,
            "default",
            system_metrics.get("network_tx_error_rate", 0.0),
            system_metrics.get("network_rx_error_rate", 0.0),
            0.0,
        )
        metrics_exporter.record_edge_network_bytes(
            service,
            "default",
            int(system_metrics.get("network_tx_bytes_per_sec", 0)),
            int(system_metrics.get("network_rx_bytes_per_sec", 0)),
        )
    except Exception as e:
        get_logger(__name__).debug("Failed to record system metrics to exporter: %s", e)


def main():
    """Main entry point for edge detector service."""
    parser = argparse.ArgumentParser(description='Edge Anomaly Detector')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--service', required=True, help='Service name')
    parser.add_argument('--model-path', help='Path to pre-trained model')
    args = parser.parse_args()
    
    # Initialize
    config = init_config(args.config)
    setup_logger(
        level=config.get('monitoring.logging.level', 'INFO'),
        log_file=f"logs/{args.service}.log"
    )
    logger = get_logger(__name__)
    
    logger.info(f"Starting edge detector for service: {args.service}")
    
    # Create model
    model_config = config.get('edge.model')
    model = LSTMAnomalyDetector(
        input_size=model_config.get('input_size', 10),
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2)
    )
    
    # Create detector
    detector = EdgeDetector(args.service, model=model, config=config.to_dict())

    # System metrics collector (CPU, Memory, Disk, Network) for anomaly detection and FL
    system_collector = SystemMetricsCollector(service_name=args.service)

    # Load pre-trained model if provided
    if args.model_path:
        from utils.serialization import load_model
        load_model(model, args.model_path)
        logger.info(f"Loaded model from {args.model_path}")

    # Start metrics server
    metrics = get_metrics()
    metrics.start()
    logger.info("Metrics server started on port 9090")

    # Main loop for metrics collection and anomaly detection
    logger.info("Entering detection loop...")

    try:
        while True:
            # Collect system metrics from this edge/node
            system_metrics = system_collector.collect()
            # Record to Prometheus exporter for monitoring
            _record_system_metrics_to_exporter(metrics, args.service, system_metrics)
            # Build feature vector for anomaly detection (includes CPU, Memory, Disk, Network)
            feature_vector = np.array([system_collector.to_vector(system_metrics)], dtype=np.float64)
            result = detector.detect(feature_vector)

            if result['is_anomaly']:
                logger.warning(
                    f"Anomaly detected: score={result['score']:.3f}, "
                    f"threshold={result['threshold']:.3f}"
                )

            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down edge detector...")
        sys.exit(0)


if __name__ == '__main__':
    main()
