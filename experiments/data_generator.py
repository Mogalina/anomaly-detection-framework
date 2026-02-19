import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from faker import Faker

from utils.logger import get_logger
from utils.config import get_config


logger = get_logger(__name__)
fake = Faker()


class DataGenerator:
    """
    Generates synthetic system metrics and traces for testing.
    
    Includes various anomaly patterns and realistic microservice behaviors.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        
        gen_config = self.config.get('experiments', {}).get('data_generation', {})
        
        self.num_services = gen_config.get('num_services', 20)
        self.num_metrics_per_service = gen_config.get('num_metrics_per_service', 10)
        self.anomaly_injection_rate = gen_config.get('anomaly_injection_rate', 0.05)
        
        # Generate service names
        self.services = [f"service-{i}" for i in range(self.num_services)]
        
        # Metric types - Comprehensive edge system metrics
        self.metric_types = [
            # Application metrics
            'request_rate',
            'error_rate',
            'latency_p50',
            'latency_p95',
            'latency_p99',
            # CPU metrics
            'cpu_usage_percent',
            'cpu_load1',
            'cpu_load5',
            # Memory metrics
            'memory_usage_percent',
            'memory_available_mb',
            # Disk metrics
            'disk_read_latency_ms',
            'disk_write_latency_ms',
            # Network metrics
            'network_tx_error_rate',
            'network_rx_error_rate',
            'network_drop_rate',
            'network_throughput_mbps'
        ]
        
        logger.info(f"DataGenerator initialized with {self.num_services} services")
    
    def generate_normal_metrics(
        self,
        service: str,
        duration_hours: int = 24,
        sampling_rate: int = 60
    ) -> pd.DataFrame:
        """
        Generate normal (non-anomalous) metrics for a service.
        
        Args:
            service: Service name
            duration_hours: Duration in hours
            sampling_rate: Sampling interval in seconds
        
        Returns:
            DataFrame with timestamps and metric values
        """
        num_samples = (duration_hours * 3600) // sampling_rate
        timestamps = [time.time() + i * sampling_rate for i in range(num_samples)]
        
        data = {'timestamp': timestamps, 'service': [service] * num_samples}
        
        # Generate each metric type with realistic ranges
        for i, metric_type in enumerate(self.metric_types[:self.num_metrics_per_service]):
            # Base pattern with daily seasonality
            t = np.arange(num_samples)
            daily_pattern = 20 * np.sin(2 * np.pi * t / (24 * 3600 / sampling_rate))
            
            # Add noise
            noise = np.random.normal(0, 5, num_samples)
            
            # Different base values and ranges for different metric types
            if metric_type == 'cpu_usage_percent' or metric_type == 'memory_usage_percent':
                base = 50
                values = np.clip(base + daily_pattern + noise, 5, 95)
            
            elif metric_type == 'cpu_load1':
                base = 2.0
                values = np.clip(base + daily_pattern * 0.1 + noise * 0.1, 0.1, 8.0)
            
            elif metric_type == 'cpu_load5':
                base = 2.5
                values = np.clip(base + daily_pattern * 0.08 + noise * 0.08, 0.2, 8.0)
            
            elif metric_type == 'memory_available_mb':
                base = 2048  # 2GB
                # Inverse of usage - more usage = less available
                values = np.clip(base - daily_pattern * 20 + noise * 50, 100, 4096)
            
            elif metric_type == 'disk_read_latency_ms' or metric_type == 'disk_write_latency_ms':
                base = 10  # 10ms baseline
                # Disk latency can spike occasionally
                spikes = np.random.exponential(5, num_samples)
                values = np.clip(base + daily_pattern * 0.5 + noise + spikes, 1, 200)
            
            elif metric_type == 'network_tx_error_rate' or metric_type == 'network_rx_error_rate':
                base = 0.001  # 0.1% error rate
                # Very low baseline with occasional spikes
                values = np.clip(base + noise * 0.0001, 0, 0.05)
            
            elif metric_type == 'network_drop_rate':
                base = 0.0005  # 0.05% drop rate
                # Even lower than error rate
                values = np.clip(base + noise * 0.00005, 0, 0.02)
            
            elif metric_type == 'network_throughput_mbps':
                base = 100
                values = np.clip(base + daily_pattern * 2 + noise * 10, 1, 1000)
            
            elif 'rate' in metric_type and 'error' not in metric_type:
                base = 100
                values = np.clip(base + daily_pattern + noise, 0, 1000)
            
            elif 'latency' in metric_type:
                base = 200
                values = np.clip(base + daily_pattern + noise, 10, 2000)
            
            else:
                base = 50
                values = base + daily_pattern + noise
            
            data[f'metric_{i}'] = values
        
        return pd.DataFrame(data)
    
    def inject_anomaly(
        self,
        data: pd.DataFrame,
        anomaly_type: str,
        start_idx: int,
        duration: int = 20,
        metric_indices: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject an anomaly into the data.
        
        Args:
            data: DataFrame to inject anomaly into
            anomaly_type: Type of anomaly (spike, drift, oscillation, level_shift, missing_data)
            start_idx: Starting index for anomaly
            duration: Duration of anomaly in samples
            metric_indices: Which metrics to affect, if None affects random subset
        
        Returns:
            Tuple of (modified DataFrame, anomaly metadata)
        """
        data = data.copy()
        
        if metric_indices is None:
            # Affect 20-50% of metrics
            num_affected = np.random.randint(
                max(1, self.num_metrics_per_service // 5),
                max(2, self.num_metrics_per_service // 2)
            )
            metric_indices = np.random.choice(
                self.num_metrics_per_service,
                num_affected,
                replace=False
            )
        
        end_idx = min(start_idx + duration, len(data))
        
        metadata = {
            'type': anomaly_type,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'duration': end_idx - start_idx,
            'affected_metrics': [f'metric_{i}' for i in metric_indices]
        }
        
        for i in metric_indices:
            metric_col = f'metric_{i}'
            original_values = data[metric_col][start_idx:end_idx].values
            
            if anomaly_type == 'spike':
                # Sudden spike
                magnitude = np.random.uniform(2, 5)
                anomaly_values = original_values * magnitude
            
            elif anomaly_type == 'drift':
                # Gradual drift
                drift_rate = np.random.uniform(0.5, 2.0)
                drift = np.linspace(0, drift_rate * original_values.mean(), len(original_values))
                anomaly_values = original_values + drift
            
            elif anomaly_type == 'oscillation':
                # High-frequency oscillation
                freq = np.random.uniform(0.5, 2.0)
                amplitude = original_values.std() * 3
                oscillation = amplitude * np.sin(freq * np.arange(len(original_values)))
                anomaly_values = original_values + oscillation
            
            elif anomaly_type == 'level_shift':
                # Sudden level shift
                shift = np.random.uniform(1.5, 3.0) * original_values.mean()
                anomaly_values = original_values + shift
            
            elif anomaly_type == 'missing_data':
                # Missing/null data
                anomaly_values = np.full(len(original_values), np.nan)
            
            else:
                continue
            
            data.loc[start_idx:end_idx-1, metric_col] = anomaly_values
        
        return data, metadata
    
    def generate_dataset_with_anomalies(
        self,
        service: str,
        duration_hours: int = 24,
        sampling_rate: int = 60
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate a complete dataset with injected anomalies.
        
        Args:
            service: Service name
            duration_hours: Duration in hours
            sampling_rate: Sampling rate in seconds
        
        Returns:
            Tuple of (DataFrame, list of anomaly metadata)
        """
        # Generate normal data
        data = self.generate_normal_metrics(service, duration_hours, sampling_rate)
        
        # Determine number of anomalies to inject
        num_samples = len(data)
        num_anomalies = int(num_samples * self.anomaly_injection_rate)
        
        anomaly_types = ['spike', 'drift', 'oscillation', 'level_shift', 'missing_data']
        anomalies = []
        
        # Ensure minimum spacing between anomalies
        min_spacing = 100
        occupied_ranges = []
        
        for _ in range(num_anomalies):
            # Find valid start position
            attempts = 0
            while attempts < 100:
                start_idx = np.random.randint(0, num_samples - min_spacing)
                
                # Check if overlaps with existing anomaly
                overlaps = any(
                    start_idx < end and start_idx + min_spacing > start
                    for start, end in occupied_ranges
                )
                
                if not overlaps:
                    break
                
                attempts += 1
            
            if attempts >= 100:
                continue
            
            # Choose anomaly type
            anomaly_type = np.random.choice(anomaly_types)
            duration = np.random.randint(10, 50)
            
            # Inject anomaly
            data, metadata = self.inject_anomaly(
                data,
                anomaly_type,
                start_idx,
                duration
            )
            
            metadata['timestamp'] = data.iloc[start_idx]['timestamp']
            metadata['service'] = service
            metadata['is_root_cause'] = np.random.random() < 0.3  # 30% are root causes
            
            anomalies.append(metadata)
            occupied_ranges.append((start_idx, start_idx + duration))
        
        logger.info(
            f"Generated dataset for {service}: "
            f"{num_samples} samples, {len(anomalies)} anomalies"
        )
        
        return data, anomalies
    
    def generate_traces(
        self,
        duration_minutes: int = 60,
        avg_traces_per_minute: int = 10
    ) -> List[Dict]:
        """
        Generate synthetic distributed traces.
        
        Args:
            duration_minutes: Duration in minutes
            avg_traces_per_minute: Average number of traces per minute
        
        Returns:
            List of trace dictionaries
        """
        traces = []
        num_traces = duration_minutes * avg_traces_per_minute
        
        # Build simple service dependency graph
        # Entry services
        entry_services = self.services[:3]
        
        for _ in range(num_traces):
            # Start with entry service
            entry = np.random.choice(entry_services)
            
            # Generate call chain
            call_chain = [entry]
            current = entry
            
            # Random walk through services
            max_depth = np.random.randint(2, 6)
            for _ in range(max_depth):
                # Choose next service
                next_service = np.random.choice([s for s in self.services if s != current])
                call_chain.append(next_service)
                current = next_service
            
            # Generate trace
            trace_id = fake.uuid4()
            start_time = int(time.time() * 1e6) + np.random.randint(0, duration_minutes * 60) * int(1e6)
            
            dependencies = []
            for i in range(len(call_chain) - 1):
                from_service = call_chain[i]
                to_service = call_chain[i + 1]
                
                # Generate realistic latency
                latency = int(np.random.lognormal(4, 1) * 1000)  # microseconds
                
                dependencies.append({
                    'from': from_service,
                    'to': to_service,
                    'operation': f'call_{to_service}',
                    'duration': latency
                })
            
            traces.append({
                'trace_id': trace_id,
                'start_time': start_time,
                'duration': sum(d['duration'] for d in dependencies),
                'num_spans': len(call_chain),
                'dependencies': dependencies,
                'service_calls': {}
            })
        
        logger.info(f"Generated {len(traces)} synthetic traces")
        
        return traces
    
    def export_to_alibaba_format(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Export data in Alibaba trace format.
        
        Args:
            data: DataFrame to export
            output_path: Output file path
        """
        data.to_csv(output_path, index=False)
        logger.info(f"Exported data to {output_path}")
