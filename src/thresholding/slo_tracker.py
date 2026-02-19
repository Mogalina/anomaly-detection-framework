from typing import Dict, List, Optional
import time
from collections import defaultdict, deque

from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import get_metrics


logger = get_logger(__name__)


class SLOTracker:
    """
    Tracks Service Level Objectives and correlates with anomalies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SLO tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        slo_config = self.config.get('thresholding', {}).get('slo', {})
        
        # SLO thresholds
        self.latency_p95_threshold = slo_config.get('latency_p95_ms', 200)
        self.latency_p99_threshold = slo_config.get('latency_p99_ms', 500)
        self.error_rate_threshold = slo_config.get('error_rate_threshold', 0.01)
        self.collection_interval = slo_config.get('collection_interval', 60)
        
        # Service SLO data
        self.service_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.service_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.service_requests: Dict[str, int] = defaultdict(int)
        
        # Violation tracking
        self.slo_violations: Dict[str, List] = defaultdict(list)
        self.anomaly_slo_correlation: Dict[str, List] = defaultdict(list)
        
        self.last_collection_time = time.time()
        
        logger.info(
            f"SLOTracker initialized: "
            f"p95={self.latency_p95_threshold}ms, "
            f"p99={self.latency_p99_threshold}ms"
        )
    
    def record_request(
        self,
        service: str,
        latency_ms: float,
        is_error: bool = False
    ) -> None:
        """
        Record a service request.
        
        Args:
            service: Service name
            latency_ms: Request latency in milliseconds
            is_error: Whether request resulted in error
        """
        self.service_latencies[service].append(latency_ms)
        self.service_errors[service].append(1 if is_error else 0)
        self.service_requests[service] += 1
    
    def check_slo_violations(self, service: str) -> Dict:
        """
        Check for SLO violations in a service.
        
        Args:
            service: Service name
        
        Returns:
            Violation status dictionary
        """
        if service not in self.service_latencies or not self.service_latencies[service]:
            return {'violations': []}
        
        latencies = list(self.service_latencies[service])
        errors = list(self.service_errors[service])
        
        violations = []
        
        # Check latency P95
        if len(latencies) >= 20:
            import numpy as np
            p95 = np.percentile(latencies, 95)
            if p95 > self.latency_p95_threshold:
                violations.append({
                    'type': 'latency_p95',
                    'value': p95,
                    'threshold': self.latency_p95_threshold,
                    'severity': 'medium'
                })
                self.metrics.record_slo_violation(service, 'latency_p95')
        
            # Check latency P99
            p99 = np.percentile(latencies, 99)
            if p99 > self.latency_p99_threshold:
                violations.append({
                    'type': 'latency_p99',
                    'value': p99,
                    'threshold': self.latency_p99_threshold,
                    'severity': 'high'
                })
                self.metrics.record_slo_violation(service, 'latency_p99')
        
        # Check error rate
        if len(errors) >= 10:
            error_rate = sum(errors) / len(errors)
            if error_rate > self.error_rate_threshold:
                violations.append({
                    'type': 'error_rate',
                    'value': error_rate,
                    'threshold': self.error_rate_threshold,
                    'severity': 'critical'
                })
                self.metrics.record_slo_violation(service, 'error_rate')
        
        if violations:
            violation_record = {
                'service': service,
                'timestamp': time.time(),
                'violations': violations
            }
            self.slo_violations[service].append(violation_record)
            
            # Keep only recent violations
            if len(self.slo_violations[service]) > 100:
                self.slo_violations[service] = self.slo_violations[service][-100:]
        
        return {'violations': violations}
    
    def correlate_with_anomaly(
        self,
        service: str,
        anomaly_timestamp: float,
        time_window: float = 300
    ) -> Dict:
        """
        Check if SLO violations correlate with an anomaly.
        
        Args:
            service: Service name
            anomaly_timestamp: Timestamp of anomaly
            time_window: Time window to check (seconds)
        
        Returns:
            Correlation data
        """
        if service not in self.slo_violations:
            return {'correlated': False}
        
        # Find violations within time window
        correlated_violations = []
        for violation in self.slo_violations[service]:
            time_diff = abs(violation['timestamp'] - anomaly_timestamp)
            if time_diff <= time_window:
                correlated_violations.append(violation)
        
        correlated = len(correlated_violations) > 0
        
        if correlated:
            correlation_record = {
                'service': service,
                'anomaly_timestamp': anomaly_timestamp,
                'violations': correlated_violations,
                'correlation_strength': len(correlated_violations)
            }
            self.anomaly_slo_correlation[service].append(correlation_record)
        
        return {
            'correlated': correlated,
            'violations': correlated_violations,
            'correlation_strength': len(correlated_violations)
        }
    
    def get_service_slo_status(self, service: str) -> Dict:
        """
        Get current SLO status for a service.
        
        Args:
            service: Service name
        
        Returns:
            SLO status dictionary
        """
        if service not in self.service_latencies:
            return {}
        
        latencies = list(self.service_latencies[service])
        errors = list(self.service_errors[service])
        
        if not latencies:
            return {}
        
        import numpy as np
        
        return {
            'service': service,
            'num_requests': self.service_requests[service],
            'latency_p50': np.percentile(latencies, 50) if len(latencies) >= 2 else 0,
            'latency_p95': np.percentile(latencies, 95) if len(latencies) >= 20 else 0,
            'latency_p99': np.percentile(latencies, 99) if len(latencies) >= 20 else 0,
            'error_rate': sum(errors) / len(errors) if errors else 0,
            'recent_violations': len([
                v for v in self.slo_violations.get(service, [])
                if time.time() - v['timestamp'] < 3600
            ])
        }
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            Statistics dictionary
        """
        total_violations = sum(len(v) for v in self.slo_violations.values())
        total_correlations = sum(len(c) for c in self.anomaly_slo_correlation.values())
        
        return {
            'num_tracked_services': len(self.service_latencies),
            'total_requests': sum(self.service_requests.values()),
            'total_violations': total_violations,
            'total_correlations': total_correlations,
            'services_with_violations': len([
                s for s, v in self.slo_violations.items() if v
            ])
        }
