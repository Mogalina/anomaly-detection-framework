import time
import numpy as np
from typing import Dict, List
import json
from pathlib import Path

from edge.edge_detector import EdgeDetector
from edge.models import LSTMAnomalyDetector
from federated.federated_coordinator import FederatedCoordinator
from federated.federated_client import FederatedClient
from coordinator.anomaly_pipeline import AnomalyPipeline
from utils.logger import setup_logger, get_logger
from utils.config import init_config
from utils.preprocessing import sliding_window
from utils.system_metrics_collector import collect_system_metrics
from .data_generator import DataGenerator


setup_logger(level='INFO')
logger = get_logger(__name__)


class Benchmark:
    """
    Comprehensive benchmark suite for the anomaly detection framework.
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize benchmark.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = init_config(config_path)
        self.data_generator = DataGenerator()
        self.results = []
        
        logger.info("Benchmark initialized")
    
    def benchmark_detection_latency(self, num_trials: int = 10) -> Dict:
        """
        Benchmark edge detection latency.
        
        Args:
            num_trials: Number of trials
        
        Returns:
            Latency statistics
        """
        logger.info(f"Running detection latency benchmark ({num_trials} trials)")
        
        latencies = []
        
        # Create detector
        model = LSTMAnomalyDetector(input_size=10, hidden_size=64, num_layers=2)
        detector = EdgeDetector('test-service', model=model)
        
        # Generate test data
        data, _ = self.data_generator.generate_dataset_with_anomalies(
            'test-service',
            duration_hours=1,
            sampling_rate=10
        )
        
        # Prepare windows
        metric_cols = [c for c in data.columns if c.startswith('metric_')]
        metrics_data = data[metric_cols].values
        
        windows, _ = sliding_window(metrics_data, window_size=100, stride=1)
        
        # Train detector
        logger.info("Training detector...")
        detector.train(metrics_data[:500], epochs=10, batch_size=32)
        
        # Benchmark inference
        for i in range(num_trials):
            idx = np.random.randint(0, len(windows))
            window = windows[idx]
            
            start_time = time.time()
            result = detector.detect(window[-1])
            latency = time.time() - start_time
            
            latencies.append(latency * 1000)  # Convert to ms
        
        result = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies)
        }
        
        logger.info(f"Detection latency: {result['mean_latency_ms']:.2f}ms (mean)")
        
        return result
    
    def benchmark_detection_accuracy(self, num_services: int = 5) -> Dict:
        """
        Benchmark detection accuracy.
        
        Args:
            num_services: Number of services to test
        
        Returns:
            Accuracy metrics
        """
        logger.info(f"Running detection accuracy benchmark ({num_services} services)")
        
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for service_idx in range(num_services):
            service = f"service-{service_idx}"
            
            # Generate data with known anomalies
            data, anomalies = self.data_generator.generate_dataset_with_anomalies(
                service,
                duration_hours=2,
                sampling_rate=10
            )
            
            metric_cols = [c for c in data.columns if c.startswith('metric_')]
            metrics_data = data[metric_cols].values
            
            # Create and train detector
            model = LSTMAnomalyDetector(input_size=len(metric_cols), hidden_size=64, num_layers=2)
            detector = EdgeDetector(service, model=model)
            detector.train(metrics_data[:300], epochs=10, batch_size=32)
            
            # Test on remaining data
            test_data = metrics_data[300:]
            test_timestamps = data.iloc[300:]['timestamp'].values
            
            # Determine ground truth labels
            ground_truth = np.zeros(len(test_data), dtype=bool)
            for anomaly in anomalies:
                if anomaly['start_idx'] >= 300:
                    start = anomaly['start_idx'] - 300
                    end = min(anomaly['end_idx'] - 300, len(test_data))
                    ground_truth[start:end] = True
            
            # Detect
            predictions = []
            for i, sample in enumerate(test_data):
                result = detector.detect(sample.reshape(1, -1))
                predictions.append(result['is_anomaly'])
            
            predictions = np.array(predictions)
            
            # Compute confusion matrix
            true_positives += np.sum(predictions & ground_truth)
            false_positives += np.sum(predictions & ~ground_truth)
            true_negatives += np.sum(~predictions & ~ground_truth)
            false_negatives += np.sum(~predictions & ground_truth)
        
        # Compute metrics
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1_score = 2 * precision * recall / max(0.001, precision + recall)
        accuracy = (true_positives + true_negatives) / max(1, true_positives + false_positives + true_negatives + false_negatives)
        
        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
        
        logger.info(
            f"Accuracy: precision={precision:.3f}, recall={recall:.3f}, "
            f"f1={f1_score:.3f}"
        )
        
        return result
    
    def benchmark_federated_bandwidth(self, num_rounds: int = 10) -> Dict:
        """
        Benchmark federated learning bandwidth consumption.
        
        Args:
            num_rounds: Number of FL rounds
        
        Returns:
            Bandwidth statistics
        """
        logger.info(f"Running federated bandwidth benchmark ({num_rounds} rounds)")
        
        # Create coordinator
        coordinator = FederatedCoordinator()
        
        # Create clients
        num_clients = 5
        clients = []
        for i in range(num_clients):
            model = LSTMAnomalyDetector(input_size=10, hidden_size=64, num_layers=2)
            client = FederatedClient(f"client-{i}", model)
            clients.append(client)
        
        # Generate training data for each client
        client_data = []
        for i in range(num_clients):
            data, _ = self.data_generator.generate_dataset_with_anomalies(
                f"service-{i}",
                duration_hours=1,
                sampling_rate=10
            )
            metric_cols = [c for c in data.columns if c.startswith('metric_')]
            metrics_data = data[metric_cols].values
            windows, _ = sliding_window(metrics_data, window_size=100, stride=1)
            client_data.append(windows)
        
        upload_sizes = []
        download_sizes = []
        
        # Run FL rounds
        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")
            
            # Each client trains and uploads
            for client_idx, client in enumerate(clients):
                # Download global model
                global_params = coordinator.get_global_model()
                download_size = len(json.dumps(global_params))
                download_sizes.append(download_size)
                
                # Collect system metrics (CPU, Memory, Disk, Network) for this node
                system_metrics = collect_system_metrics(service_name=client.client_id)

                # Train locally (include system_metrics for root cause at coordinator)
                result = client.train_round(
                    client_data[client_idx],
                    global_model_params=global_params,
                    system_metrics=system_metrics,
                )

                # Upload update
                upload_size = len(json.dumps(result['model_update']))
                upload_sizes.append(upload_size)

                # Send to coordinator (full metrics including CPU, Memory, Disk, Network)
                coordinator.receive_update(
                    client.client_id,
                    result['model_update'],
                    result['num_samples'],
                    result.get('metrics', {'loss': result['loss'], 'system_metrics': system_metrics}),
                )
            
            # Aggregate
            coordinator.aggregate_models()
        
        result = {
            'total_upload_bytes': sum(upload_sizes),
            'total_download_bytes': sum(download_sizes),
            'avg_upload_per_round_bytes': np.mean(upload_sizes),
            'avg_download_per_round_bytes': np.mean(download_sizes),
            'total_bandwidth_mb': (sum(upload_sizes) + sum(download_sizes)) / (1024 * 1024)
        }
        
        logger.info(f"Total bandwidth: {result['total_bandwidth_mb']:.2f} MB")
        
        return result
    
    def run_full_benchmark(self, output_dir: str = 'results') -> Dict:
        """
        Run complete benchmark suite.
        
        Args:
            output_dir: Directory to save results
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting full benchmark suite")
        
        results = {
            'timestamp': time.time(),
            'latency': self.benchmark_detection_latency(num_trials=50),
            'accuracy': self.benchmark_detection_accuracy(num_services=5),
            'bandwidth': self.benchmark_federated_bandwidth(num_rounds=10)
        }
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark completed. Results saved to {results_file}")
        
        return results


def main():
    """Main entry point for benchmark."""
    benchmark = Benchmark()
    results = benchmark.run_full_benchmark()
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nDetection Latency:")
    print(f"  Mean: {results['latency']['mean_latency_ms']:.2f} ms")
    print(f"  P95:  {results['latency']['p95_latency_ms']:.2f} ms")
    print(f"\nDetection Accuracy:")
    print(f"  Precision: {results['accuracy']['precision']:.3f}")
    print(f"  Recall:    {results['accuracy']['recall']:.3f}")
    print(f"  F1 Score:  {results['accuracy']['f1_score']:.3f}")
    print(f"\nFederated Learning Bandwidth:")
    print(f"  Total: {results['bandwidth']['total_bandwidth_mb']:.2f} MB")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
