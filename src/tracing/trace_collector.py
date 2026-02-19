import requests
from typing import Dict, List, Optional
import time
from collections import defaultdict

from utils.logger import get_logger
from utils.config import get_config


logger = get_logger(__name__)


class TraceCollector:
    """
    Collector for distributed traces from Jaeger.
    
    Parses trace data to extract service dependencies and call patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trace collector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        
        trace_config = self.config.get('tracing', {})
        jaeger_config = trace_config.get('jaeger', {})
        
        # Jaeger connection
        self.jaeger_host = jaeger_config.get('host', 'localhost')
        self.jaeger_port = jaeger_config.get('query_port', 16686)
        self.base_url = f"http://{self.jaeger_host}:{self.jaeger_port}"
        
        # Collection configuration
        collection_config = trace_config.get('collection', {})
        self.time_window = collection_config.get('time_window', 300)  # 5 minutes
        self.min_trace_length = collection_config.get('min_trace_length', 2)
        self.max_trace_age = collection_config.get('max_trace_age', 3600)
        
        # Cache
        self.cached_traces = []
        self.last_fetch_time = 0
        
        logger.info(
            f"TraceCollector initialized: jaeger={self.jaeger_host}:{self.jaeger_port}"
        )
    
    def collect_traces(
        self,
        service: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Collect traces from Jaeger.
        
        Args:
            service: Optional service name filter
            start_time: Start timestamp (microseconds)
            end_time: End timestamp (microseconds)
            limit: Maximum number of traces
        
        Returns:
            List of parsed traces
        """
        try:
            # Default time window
            if end_time is None:
                end_time = int(time.time() * 1e6)
            if start_time is None:
                start_time = end_time - (self.time_window * 1e6)
            
            # Build query URL
            url = f"{self.base_url}/api/traces"
            params = {
                'start': start_time,
                'end': end_time,
                'limit': limit
            }
            
            if service:
                params['service'] = service
            
            # Fetch traces
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            traces = data.get('data', [])
            
            # Parse traces
            parsed_traces = []
            for trace in traces:
                parsed = self._parse_trace(trace)
                if parsed:
                    parsed_traces.append(parsed)
            
            self.cached_traces = parsed_traces
            self.last_fetch_time = time.time()
            
            logger.info(f"Collected {len(parsed_traces)} traces from Jaeger")
            
            return parsed_traces
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to collect traces from Jaeger: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing traces: {e}")
            return []
    
    def _parse_trace(self, trace: Dict) -> Optional[Dict]:
        """
        Parse a single trace into structured format.
        
        Args:
            trace: Raw trace from Jaeger
        
        Returns:
            Parsed trace or None if invalid
        """
        try:
            trace_id = trace.get('traceID')
            spans = trace.get('spans', [])
            
            if not spans or len(spans) < self.min_trace_length:
                return None
            
            # Extract service dependencies
            dependencies = []
            service_calls = defaultdict(list)
            
            # Build span lookup
            span_map = {span['spanID']: span for span in spans}
            
            for span in spans:
                service_name = span.get('process', {}).get('serviceName', 'unknown')
                span_id = span.get('spanID')
                parent_span_id = span.get('references', [{}])[0].get('spanID') if span.get('references') else None
                
                start_time = span.get('startTime', 0)
                duration = span.get('duration', 0)
                operation = span.get('operationName', 'unknown')
                
                # Find parent service
                if parent_span_id and parent_span_id in span_map:
                    parent_span = span_map[parent_span_id]
                    parent_service = parent_span.get('process', {}).get('serviceName', 'unknown')
                    
                    # Record dependency
                    dependencies.append({
                        'from': parent_service,
                        'to': service_name,
                        'operation': operation,
                        'duration': duration
                    })
                    
                    service_calls[parent_service].append({
                        'target': service_name,
                        'operation': operation,
                        'duration': duration,
                        'timestamp': start_time
                    })
            
            # Get trace metadata
            trace_start = min(span.get('startTime', 0) for span in spans)
            trace_duration = max(
                span.get('startTime', 0) + span.get('duration', 0)
                for span in spans
            ) - trace_start
            
            return {
                'trace_id': trace_id,
                'start_time': trace_start,
                'duration': trace_duration,
                'num_spans': len(spans),
                'dependencies': dependencies,
                'service_calls': dict(service_calls)
            }
        
        except Exception as e:
            logger.warning(f"Error parsing trace: {e}")
            return None
    
    def get_service_dependencies(
        self,
        traces: Optional[List[Dict]] = None
    ) -> Dict[str, List[str]]:
        """
        Extract service dependency graph from traces.
        
        Args:
            traces: Traces to analyze, uses cached if None
        
        Returns:
            Dictionary mapping service -> list of dependent services
        """
        if traces is None:
            traces = self.cached_traces
        
        dependencies = defaultdict(set)
        
        for trace in traces:
            for dep in trace.get('dependencies', []):
                from_service = dep['from']
                to_service = dep['to']
                dependencies[from_service].add(to_service)
        
        # Convert sets to lists
        return {k: list(v) for k, v in dependencies.items()}
    
    def get_service_metrics(
        self,
        traces: Optional[List[Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Compute service-level metrics from traces.
        
        Args:
            traces: Traces to analyze
        
        Returns:
            Dictionary of service metrics
        """
        if traces is None:
            traces = self.cached_traces
        
        metrics = defaultdict(lambda: {
            'call_count': 0,
            'total_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'errors': 0
        })
        
        for trace in traces:
            for dep in trace.get('dependencies', []):
                service = dep['to']
                duration = dep['duration']
                
                metrics[service]['call_count'] += 1
                metrics[service]['total_duration'] += duration
                metrics[service]['min_duration'] = min(
                    metrics[service]['min_duration'],
                    duration
                )
                metrics[service]['max_duration'] = max(
                    metrics[service]['max_duration'],
                    duration
                )
        
        # Compute averages
        for service, m in metrics.items():
            if m['call_count'] > 0:
                m['avg_duration'] = m['total_duration'] / m['call_count']
            else:
                m['avg_duration'] = 0
        
        return dict(metrics)
    
    def filter_noise(self, traces: List[Dict]) -> List[Dict]:
        """
        Filter noisy or incomplete traces.
        
        Args:
            traces: List of traces
        
        Returns:
            Filtered traces
        """
        filtered = []
        
        for trace in traces:
            # Filter by age
            if time.time() - (trace['start_time'] / 1e6) > self.max_trace_age:
                continue
            
            # Filter by length
            if trace['num_spans'] < self.min_trace_length:
                continue
            
            # Filter by duration (remove extremely fast traces - likely health checks)
            if trace['duration'] < 1000:  # 1ms
                continue
            
            filtered.append(trace)
        
        return filtered
