from typing import Dict, List, Optional, Any
import time

from utils.logger import get_logger

logger = get_logger(__name__)

# Order of system metric keys for consistent vector representation
SYSTEM_METRIC_KEYS = [
    "cpu_usage_percent",
    "cpu_load1",
    "cpu_load5",
    "memory_usage_percent",
    "memory_available_mb",
    "disk_usage_percent",
    "disk_read_latency_ms",
    "disk_write_latency_ms",
    "network_tx_bytes_per_sec",
    "network_rx_bytes_per_sec",
    "network_tx_error_rate",
    "network_rx_error_rate",
]


def _get_psutil():
    """Lazy import psutil to avoid hard dependency at import time."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


class SystemMetricsCollector:
    """
    Collects CPU, Memory, Disk, and Network metrics from the current node.
    """

    def __init__(self, service_name: str = "default"):
        self.service_name = service_name
        self._psutil = _get_psutil()
        self._last_net_io = None
        self._last_net_io_time = None
        self._last_disk_io = None
        self._last_disk_io_time = None

    def collect(self) -> Dict[str, float]:
        """
        Collect all system metrics (CPU, Memory, Disk, Network).

        Returns:
            Dictionary of metric name -> value. Missing metrics are 0.0.
        """
        out = {k: 0.0 for k in SYSTEM_METRIC_KEYS}
        if self._psutil is None:
            logger.warning("psutil not installed; system metrics will be zero")
            return out

        try:
            # CPU
            out["cpu_usage_percent"] = float(self._psutil.cpu_percent(interval=None))
            load = self._psutil.getloadavg() if hasattr(self._psutil, "getloadavg") else (0.0, 0.0, 0.0)
            out["cpu_load1"] = float(load[0]) if len(load) > 0 else 0.0
            out["cpu_load5"] = float(load[1]) if len(load) > 1 else 0.0
        except Exception as e:
            logger.debug("CPU metrics collection failed: %s", e)

        try:
            # Memory
            mem = self._psutil.virtual_memory()
            out["memory_usage_percent"] = float(mem.percent)
            out["memory_available_mb"] = float(mem.available) / (1024 * 1024)
        except Exception as e:
            logger.debug("Memory metrics collection failed: %s", e)

        try:
            # Disk (usage and IO; latency from read_time/write_time when available)
            disk_usage = self._psutil.disk_usage("/")
            out["disk_usage_percent"] = float(disk_usage.percent)
            disk_io = self._psutil.disk_io_counters()
            now = time.time()
            if disk_io is not None and self._last_disk_io is not None and self._last_disk_io_time is not None:
                read_count_delta = disk_io.read_count - self._last_disk_io.read_count
                write_count_delta = disk_io.write_count - self._last_disk_io.write_count
                if hasattr(disk_io, "read_time") and hasattr(self._last_disk_io, "read_time") and read_count_delta > 0:
                    read_time_delta = disk_io.read_time - self._last_disk_io.read_time
                    out["disk_read_latency_ms"] = read_time_delta / read_count_delta
                if hasattr(disk_io, "write_time") and hasattr(self._last_disk_io, "write_time") and write_count_delta > 0:
                    write_time_delta = disk_io.write_time - self._last_disk_io.write_time
                    out["disk_write_latency_ms"] = write_time_delta / write_count_delta
            self._last_disk_io = disk_io
            self._last_disk_io_time = now
            if out["disk_read_latency_ms"] == 0.0:
                out["disk_read_latency_ms"] = 1.0
            if out["disk_write_latency_ms"] == 0.0:
                out["disk_write_latency_ms"] = 1.0
        except Exception as e:
            logger.debug("Disk metrics collection failed: %s", e)
            out["disk_read_latency_ms"] = 1.0
            out["disk_write_latency_ms"] = 1.0

        try:
            # Network (bytes/sec from counters; error rates often not available)
            net_io = self._psutil.net_io_counters()
            now = time.time()
            if net_io is not None and self._last_net_io is not None and self._last_net_io_time is not None:
                dt = now - self._last_net_io_time
                if dt > 0:
                    out["network_tx_bytes_per_sec"] = (net_io.bytes_sent - self._last_net_io.bytes_sent) / dt
                    out["network_rx_bytes_per_sec"] = (net_io.bytes_recv - self._last_net_io.bytes_recv) / dt
                    if hasattr(net_io, "errout") and net_io.packets_sent:
                        out["network_tx_error_rate"] = float(net_io.errout) / max(net_io.packets_sent, 1)
                    if hasattr(net_io, "errin") and net_io.packets_recv:
                        out["network_rx_error_rate"] = float(net_io.errin) / max(net_io.packets_recv, 1)
            self._last_net_io = net_io
            self._last_net_io_time = now
        except Exception as e:
            logger.debug("Network metrics collection failed: %s", e)

        return out

    def to_vector(self, metrics: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Return metrics as a fixed-order vector for model input.

        Args:
            metrics: Optional pre-collected dict; if None, collect() is called.

        Returns:
            List of floats in SYSTEM_METRIC_KEYS order.
        """
        m = metrics if metrics is not None else self.collect()
        return [float(m.get(k, 0.0)) for k in SYSTEM_METRIC_KEYS]

    @staticmethod
    def vector_size() -> int:
        """Number of system metric dimensions."""
        return len(SYSTEM_METRIC_KEYS)


def collect_system_metrics(service_name: str = "default") -> Dict[str, float]:
    """
    One-shot collection of system metrics for a service.

    Args:
        service_name: Service/node identifier for logging.

    Returns:
        Dictionary of metric name -> value.
    """
    collector = SystemMetricsCollector(service_name=service_name)
    return collector.collect()
