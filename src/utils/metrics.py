from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from typing import Dict, Optional
import time


class MetricsExporter:
    """
    Prometheus metrics exporter for the anomaly detection framework.
    """
    
    def __init__(self, port: int = 9090, prefix: str = "adf"):
        """
        Initialize metrics exporter.
        
        Args:
            port: Port for Prometheus metrics endpoint
            prefix: Prefix for all metrics
        """
        self.port = port
        self.prefix = prefix
        self._metrics: Dict[str, any] = {}
        
        # Initialize common metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all framework metrics."""
        
        # Edge Detection Metrics
        self._metrics["anomalies_detected"] = Counter(
            f"{self.prefix}_anomalies_detected_total",
            "Total number of anomalies detected",
            ["service", "severity"]
        )
        
        self._metrics["detection_latency"] = Histogram(
            f"{self.prefix}_detection_latency_seconds",
            "Time from anomaly occurrence to detection",
            ["service"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self._metrics["inference_time"] = Histogram(
            f"{self.prefix}_inference_time_seconds",
            "Model inference time",
            ["model_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        self._metrics["anomaly_score"] = Gauge(
            f"{self.prefix}_anomaly_score",
            "Current anomaly score",
            ["service", "metric"]
        )
        
        # Federated Learning Metrics
        self._metrics["fl_rounds_completed"] = Counter(
            f"{self.prefix}_fl_rounds_completed_total",
            "Total federated learning rounds completed"
        )
        
        self._metrics["fl_clients_participating"] = Gauge(
            f"{self.prefix}_fl_clients_participating",
            "Number of clients participating in current round"
        )
        
        self._metrics["fl_model_accuracy"] = Gauge(
            f"{self.prefix}_fl_model_accuracy",
            "Global model accuracy",
            ["metric_type"]
        )
        
        self._metrics["fl_poisoning_detected"] = Counter(
            f"{self.prefix}_fl_poisoning_detected_total",
            "Number of poisoning attempts detected",
            ["detection_method"]
        )
        
        self._metrics["fl_bandwidth_bytes"] = Counter(
            f"{self.prefix}_fl_bandwidth_bytes_total",
            "Total bandwidth used for federated learning",
            ["direction"]
        )
        
        # Root Cause Analysis Metrics
        self._metrics["root_causes_identified"] = Counter(
            f"{self.prefix}_root_causes_identified_total",
            "Total root causes identified"
        )
        
        self._metrics["rca_accuracy"] = Gauge(
            f"{self.prefix}_rca_accuracy",
            "Root cause analysis accuracy"
        )
        
        self._metrics["rca_latency"] = Histogram(
            f"{self.prefix}_rca_latency_seconds",
            "Root cause analysis time",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self._metrics["dependency_graph_size"] = Gauge(
            f"{self.prefix}_dependency_graph_nodes",
            "Number of nodes in dependency graph"
        )
        
        # Threshold Tuning Metrics
        self._metrics["threshold_adjustments"] = Counter(
            f"{self.prefix}_threshold_adjustments_total",
            "Total threshold adjustments",
            ["service", "direction"]
        )
        
        self._metrics["current_threshold"] = Gauge(
            f"{self.prefix}_current_threshold",
            "Current detection threshold",
            ["service"]
        )
        
        self._metrics["slo_violations"] = Counter(
            f"{self.prefix}_slo_violations_total",
            "Total SLO violations",
            ["service", "slo_type"]
        )
        
        self._metrics["false_positive_rate"] = Gauge(
            f"{self.prefix}_false_positive_rate",
            "False positive rate",
            ["service"]
        )
        
        self._metrics["false_negative_rate"] = Gauge(
            f"{self.prefix}_false_negative_rate",
            "False negative rate",
            ["service"]
        )
        
        # System Metrics
        self._metrics["service_health"] = Gauge(
            f"{self.prefix}_service_health",
            "Service health status (1=healthy, 0=unhealthy)",
            ["service"]
        )
        
        self._metrics["events_processed"] = Counter(
            f"{self.prefix}_events_processed_total",
            "Total events processed",
            ["event_type"]
        )
        
        # Edge System Metrics - CPU
        self._metrics["edge_cpu_usage"] = Gauge(
            f"{self.prefix}_edge_cpu_usage_percent",
            "CPU usage percentage on edge node",
            ["service", "core"]
        )
        
        self._metrics["edge_cpu_load1"] = Gauge(
            f"{self.prefix}_edge_cpu_load1",
            "CPU load average over 1 minute",
            ["service"]
        )
        
        self._metrics["edge_cpu_load5"] = Gauge(
            f"{self.prefix}_edge_cpu_load5",
            "CPU load average over 5 minutes",
            ["service"]
        )
        
        # Edge System Metrics - Memory
        self._metrics["edge_memory_usage"] = Gauge(
            f"{self.prefix}_edge_memory_usage_percent",
            "Memory usage percentage on edge node",
            ["service"]
        )
        
        self._metrics["edge_memory_available_bytes"] = Gauge(
            f"{self.prefix}_edge_memory_available_bytes",
            "Available memory in bytes",
            ["service"]
        )
        
        # Edge System Metrics - Disk
        self._metrics["edge_disk_read_latency"] = Histogram(
            f"{self.prefix}_edge_disk_read_latency_ms",
            "Disk read latency in milliseconds",
            ["service", "device"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        )
        
        self._metrics["edge_disk_write_latency"] = Histogram(
            f"{self.prefix}_edge_disk_write_latency_ms",
            "Disk write latency in milliseconds",
            ["service", "device"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        )
        
        # Edge System Metrics - Network
        self._metrics["edge_network_tx_error_rate"] = Gauge(
            f"{self.prefix}_edge_network_tx_error_rate",
            "Network transmit error rate",
            ["service", "interface"]
        )
        
        self._metrics["edge_network_rx_error_rate"] = Gauge(
            f"{self.prefix}_edge_network_rx_error_rate",
            "Network receive error rate",
            ["service", "interface"]
        )
        
        self._metrics["edge_network_drop_rate"] = Gauge(
            f"{self.prefix}_edge_network_drop_rate",
            "Network packet drop rate",
            ["service", "interface"]
        )
        
        self._metrics["edge_network_tx_bytes"] = Counter(
            f"{self.prefix}_edge_network_tx_bytes_total",
            "Total bytes transmitted",
            ["service", "interface"]
        )
        
        self._metrics["edge_network_rx_bytes"] = Counter(
            f"{self.prefix}_edge_network_rx_bytes_total",
            "Total bytes received",
            ["service", "interface"]
        )
    
    def start(self):
        """Start the metrics HTTP server."""
        start_http_server(self.port)
    
    def record_anomaly(self, service: str, severity: str = "medium"):
        """Record an anomaly detection."""
        self._metrics["anomalies_detected"].labels(
            service=service,
            severity=severity
        ).inc()
    
    def record_detection_latency(self, service: str, latency: float):
        """Record detection latency."""
        self._metrics["detection_latency"].labels(service=service).observe(latency)
    
    def record_inference_time(self, model_type: str, duration: float):
        """Record model inference time."""
        self._metrics["inference_time"].labels(model_type=model_type).observe(duration)
    
    def set_anomaly_score(self, service: str, metric: str, score: float):
        """Set current anomaly score."""
        self._metrics["anomaly_score"].labels(
            service=service,
            metric=metric
        ).set(score)
    
    def record_fl_round(self):
        """Record completion of federated learning round."""
        self._metrics["fl_rounds_completed"].inc()
    
    def set_fl_clients(self, count: int):
        """Set number of participating FL clients."""
        self._metrics["fl_clients_participating"].set(count)
    
    def set_fl_accuracy(self, metric_type: str, accuracy: float):
        """Set global model accuracy."""
        self._metrics["fl_model_accuracy"].labels(metric_type=metric_type).set(accuracy)
    
    def record_poisoning_detection(self, method: str):
        """Record poisoning attempt detection."""
        self._metrics["fl_poisoning_detected"].labels(detection_method=method).inc()
    
    def record_fl_bandwidth(self, direction: str, bytes_count: int):
        """Record federated learning bandwidth usage."""
        self._metrics["fl_bandwidth_bytes"].labels(direction=direction).inc(bytes_count)
    
    def record_root_cause(self):
        """Record root cause identification."""
        self._metrics["root_causes_identified"].inc()
    
    def set_rca_accuracy(self, accuracy: float):
        """Set root cause analysis accuracy."""
        self._metrics["rca_accuracy"].set(accuracy)
    
    def record_rca_latency(self, latency: float):
        """Record root cause analysis latency."""
        self._metrics["rca_latency"].observe(latency)
    
    def set_graph_size(self, size: int):
        """Set dependency graph size."""
        self._metrics["dependency_graph_size"].set(size)
    
    def record_threshold_adjustment(self, service: str, direction: str):
        """Record threshold adjustment."""
        self._metrics["threshold_adjustments"].labels(
            service=service,
            direction=direction
        ).inc()
    
    def set_threshold(self, service: str, threshold: float):
        """Set current threshold value."""
        self._metrics["current_threshold"].labels(service=service).set(threshold)
    
    def record_slo_violation(self, service: str, slo_type: str):
        """Record SLO violation."""
        self._metrics["slo_violations"].labels(
            service=service,
            slo_type=slo_type
        ).inc()
    
    def set_false_positive_rate(self, service: str, rate: float):
        """Set false positive rate."""
        self._metrics["false_positive_rate"].labels(service=service).set(rate)
    
    def set_false_negative_rate(self, service: str, rate: float):
        """Set false negative rate."""
        self._metrics["false_negative_rate"].labels(service=service).set(rate)
    
    def set_service_health(self, service: str, is_healthy: bool):
        """Set service health status."""
        self._metrics["service_health"].labels(service=service).set(1 if is_healthy else 0)
    
    def record_event(self, event_type: str):
        """Record event processing."""
        self._metrics["events_processed"].labels(event_type=event_type).inc()
    
    # Edge System Metrics Recording Methods
    
    def set_edge_cpu_usage(self, service: str, core: str, usage: float):
        """Set CPU usage percentage for a specific core."""
        self._metrics["edge_cpu_usage"].labels(service=service, core=core).set(usage)
    
    def set_edge_cpu_load(self, service: str, load1: float, load5: float):
        """Set CPU load averages."""
        self._metrics["edge_cpu_load1"].labels(service=service).set(load1)
        self._metrics["edge_cpu_load5"].labels(service=service).set(load5)
    
    def set_edge_memory_usage(self, service: str, usage_percent: float, available_bytes: int):
        """Set memory usage metrics."""
        self._metrics["edge_memory_usage"].labels(service=service).set(usage_percent)
        self._metrics["edge_memory_available_bytes"].labels(service=service).set(available_bytes)
    
    def record_edge_disk_read_latency(self, service: str, device: str, latency_ms: float):
        """Record disk read latency."""
        self._metrics["edge_disk_read_latency"].labels(
            service=service,
            device=device
        ).observe(latency_ms)
    
    def record_edge_disk_write_latency(self, service: str, device: str, latency_ms: float):
        """Record disk write latency."""
        self._metrics["edge_disk_write_latency"].labels(
            service=service,
            device=device
        ).observe(latency_ms)
    
    def set_edge_network_error_rates(
        self,
        service: str,
        interface: str,
        tx_error_rate: float,
        rx_error_rate: float,
        drop_rate: float
    ):
        """Set network error and drop rates."""
        self._metrics["edge_network_tx_error_rate"].labels(
            service=service,
            interface=interface
        ).set(tx_error_rate)
        
        self._metrics["edge_network_rx_error_rate"].labels(
            service=service,
            interface=interface
        ).set(rx_error_rate)
        
        self._metrics["edge_network_drop_rate"].labels(
            service=service,
            interface=interface
        ).set(drop_rate)
    
    def record_edge_network_bytes(
        self,
        service: str,
        interface: str,
        tx_bytes: int,
        rx_bytes: int
    ):
        """Record network bytes transmitted and received."""
        self._metrics["edge_network_tx_bytes"].labels(
            service=service,
            interface=interface
        ).inc(tx_bytes)
        
        self._metrics["edge_network_rx_bytes"].labels(
            service=service,
            interface=interface
        ).inc(rx_bytes)



# Global metrics instance
_metrics_instance: Optional[MetricsExporter] = None


def get_metrics() -> MetricsExporter:
    """Get the global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsExporter()
    return _metrics_instance
