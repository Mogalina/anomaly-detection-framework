import time
from typing import Dict, Set, Optional

from edge.edge_detector import EdgeDetector
from tracing.trace_collector import TraceCollector
from tracing.causal_graph import CausalGraph
from analysis.root_cause_analyzer import RootCauseAnalyzer
from thresholding.slo_tracker import SLOTracker
from thresholding.threshold_tuner import ThresholdTuner
from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import get_metrics


logger = get_logger(__name__)


class AnomalyPipeline:
    """
    End-to-end anomaly detection and analysis pipeline.
    
    Orchestrates edge detection, root cause analysis, and threshold tuning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize anomaly pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        # Initialize components
        self.trace_collector = TraceCollector(self.config)
        self.causal_graph = CausalGraph(self.config)
        self.root_cause_analyzer = RootCauseAnalyzer(self.causal_graph, self.config)
        self.slo_tracker = SLOTracker(self.config)
        self.threshold_tuner = ThresholdTuner(self.config)
        
        # Edge detectors per service
        self.edge_detectors: Dict[str, EdgeDetector] = {}
        
        # Anomaly tracking
        self.active_anomalies: Set[str] = set()
        self.anomaly_history = []
        
        # Pipeline state
        self.is_running = False
        self.last_update_time = time.time()
        
        logger.info("AnomalyPipeline initialized")
    
    def register_service(self, service_name: str, detector: EdgeDetector) -> None:
        """
        Register a service with its edge detector.
        
        Args:
            service_name: Service name
            detector: Edge detector instance
        """
        self.edge_detectors[service_name] = detector
        self.threshold_tuner.initialize_service(service_name)
        self.causal_graph.add_service(service_name)
        
        logger.info(f"Registered service: {service_name}")
    
    def process_anomaly_event(self, service: str, anomaly_data: Dict) -> Dict:
        """
        Process an anomaly event through the complete pipeline.
        
        Args:
            service: Service name
            anomaly_data: Anomaly detection data
        
        Returns:
            Analysis results
        """
        timestamp = time.time()
        
        logger.info(f"Processing anomaly event for {service}")
        
        # Mark in causal graph
        self.causal_graph.mark_anomaly(service)
        self.active_anomalies.add(service)
        
        # Check SLO correlation
        slo_correlation = self.slo_tracker.correlate_with_anomaly(
            service,
            timestamp
        )
        
        # Perform root cause analysis
        rca_result = self.root_cause_analyzer.analyze(self.active_anomalies)
        
        # Find explanation for this specific service
        explanation = rca_result['explanations'].get(service, {})
        
        # Compile result
        result = {
            'service': service,
            'timestamp': timestamp,
            'anomaly_score': anomaly_data.get('score', 0),
            'threshold': anomaly_data.get('threshold', 0),
            'slo_correlation': slo_correlation,
            'root_cause_analysis': rca_result,
            'explanation': explanation,
            'severity': anomaly_data.get('severity', 'medium')
        }
        
        # Store in history
        self.anomaly_history.append(result)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]
        
        # Record metric
        self.metrics.record_event('anomaly_processed')
        
        return result
    
    def update_threshold(self, service: str) -> None:
        """
        Update detection threshold for a service.
        
        Args:
            service: Service name
        """
        new_threshold = self.threshold_tuner.tune_threshold(service)
        
        if service in self.edge_detectors:
            self.edge_detectors[service].update_threshold(new_threshold)
    
    def provide_feedback(
        self,
        service: str,
        was_detected: bool,
        was_true_anomaly: bool
    ) -> None:
        """
        Provide feedback for threshold learning.
        
        Args:
            service: Service name
            was_detected: Whether anomaly was detected
            was_true_anomaly: Whether it was actually an anomaly
        """
        # Check SLO status
        slo_status = self.slo_tracker.check_slo_violations(service)
        slo_violated = len(slo_status.get('violations', [])) > 0
        
        # Update threshold tuner
        self.threshold_tuner.update_feedback(
            service,
            was_detected,
            was_true_anomaly,
            slo_violated
        )
        
        # Periodically tune threshold
        if len(self.threshold_tuner.service_history[service]) % 10 == 0:
            self.update_threshold(service)
    
    def update_graph_from_traces(self) -> None:
        """Update causal graph with latest traces."""
        traces = self.trace_collector.collect_traces()
        
        if traces:
            filtered_traces = self.trace_collector.filter_noise(traces)
            self.causal_graph.update_from_traces(filtered_traces)
            
            logger.info(f"Updated causal graph with {len(filtered_traces)} traces")
    
    def clear_resolved_anomalies(self, resolved_services: Set[str]) -> None:
        """
        Clear anomalies that have been resolved.
        
        Args:
            resolved_services: Set of services with resolved anomalies
        """
        for service in resolved_services:
            self.active_anomalies.discard(service)
            self.causal_graph.clear_anomaly(service)
        
        logger.info(f"Cleared {len(resolved_services)} resolved anomalies")
    
    def get_pipeline_status(self) -> Dict:
        """
        Get current pipeline status.
        
        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'num_registered_services': len(self.edge_detectors),
            'active_anomalies': len(self.active_anomalies),
            'graph_stats': self.causal_graph.get_statistics(),
            'slo_stats': self.slo_tracker.get_statistics(),
            'tuner_stats': self.threshold_tuner.get_statistics(),
            'total_processed_anomalies': len(self.anomaly_history)
        }
