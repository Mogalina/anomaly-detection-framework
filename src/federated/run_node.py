import os
import sys
import signal
import time
import uuid
import threading
from collections import deque
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config
from utils.logger import get_logger
from utils.system_metrics_collector import SystemMetricsCollector
from edge.models import LSTMAnomalyDetector
from federated.federated_client import FederatedClient


logger = get_logger(__name__)


class BackgroundMetricsBuffer:
    """
    Continuously buffers real system metrics in a background thread to construct 
    training data using sliding windows.
    """

    def __init__(
        self, 
        collector: SystemMetricsCollector, 
        input_size: int, 
        poll_interval: float = 1.0, 
        max_history: int = 10000
    ):
        """
        Initialize the BackgroundMetricsBuffer.

        Args:
            collector: SystemMetricsCollector instance to collect metrics
            input_size: Expected size of the feature vector
            poll_interval: Interval between metric collections in seconds
            max_history: Maximum number of metric vectors to store in the buffer
        """
        self.collector = collector
        self.input_size = input_size
        self.poll_interval = poll_interval
        self.buffer = deque(maxlen=max_history)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self) -> None:
        """Start the background metrics collection thread."""
        self.thread.start()
        
    def stop(self) -> None:
        """Stop the background metrics collection thread."""
        self.thread.join(timeout=2.0)
        
    def _run(self) -> None:
        """Continuously collect and buffer metrics in a background thread."""
        while not self.stop_event.is_set():
            metrics = self.collector.collect()
            vec = self.collector.to_vector(metrics)
            
            # Ensure the feature vector strictly matches the expected input_size
            if len(vec) > self.input_size:
                vec = vec[:self.input_size]
            elif len(vec) < self.input_size:
                vec = vec + [0.0] * (self.input_size - len(vec))
                
            self.buffer.append(vec)
            self.stop_event.wait(self.poll_interval)
            
    def get_training_batch(
        self, 
        num_samples: int, 
        seq_len: int
    ) -> np.ndarray:
        """
        Get a batch of training data using sliding windows.

        Args:
            num_samples: Number of samples to collect
            seq_len: Length of each sequence

        Returns:
            Batch of training data
        """
        # Calculate the required length of the buffer to create the sliding windows
        required_len = num_samples + seq_len - 1
        
        # Block until enough data is collected
        if len(self.buffer) < required_len:
            logger.info(f"Waiting for metrics buffer to fill: {len(self.buffer)}/{required_len}")
            while len(self.buffer) < required_len and not self.stop_event.is_set():
                time.sleep(1.0)
                
        if self.stop_event.is_set():
            return np.array([])
            
        # Extract the sliding window sequences from the end of the buffer
        snapshot = list(self.buffer)[-required_len:]
        
        data = []
        for i in range(num_samples):
            data.append(snapshot[i : i + seq_len])
            
        return np.array(data, dtype='float32')


def main() -> None:
    config = get_config()

    client_id = os.environ.get('CLIENT_ID') or f"node-{uuid.uuid4().hex[:8]}"
    host = os.environ.get('COORDINATOR_HOST') or config.get('federated.coordinator.host', 'localhost')
    port = int(os.environ.get('COORDINATOR_PORT') or config.get('federated.coordinator.port', 50051))
    profile = os.environ.get('NODE_PROFILE', 'lightweight')

    # Propagate profile so Federated Learning client picks it up
    os.environ['NODE_PROFILE'] = profile

    logger.info(
        f"Starting Federated Learning node: id={client_id}, "
        f"coordinator={host}:{port}, profile={profile}"
    )

    # Build the local model
    model_config = config.get('edge.model', {})
    model = LSTMAnomalyDetector(
        input_size=model_config.get('input_size', 17),
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
    )

    client = FederatedClient(client_id=client_id, model=model)
    metrics_collector = SystemMetricsCollector(service_name=client_id)

    # Start background metric buffering
    input_size = model_config.get('input_size', 17)
    metrics_buffer = BackgroundMetricsBuffer(metrics_collector, input_size=input_size)
    metrics_buffer.start()

    # Graceful shutdown
    stop_flag = [False]

    def _shutdown(sig, frame):
        logger.info(f"Signal {sig} received: stopping after current round")
        stop_flag[0] = True
        metrics_buffer.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Connect with automatic retry
    client.connect(host, port)

    # Replay any unsent update from a previous crashed run (standard profile)
    if profile == 'standard':
        client.register()
        client._restore_pending_update()
    else:
        client.register()

    num_rounds = config.get('federated.coordinator.num_rounds', 100)
    retry_delay = config.get('federated.client.retry_delay_seconds', 5)

    for round_idx in range(num_rounds):
        if stop_flag[0]:
            break

        logger.info(f"Completing Federated Learning round {round_idx + 1}/{num_rounds}")

        try:
            seq_len = model_config.get('sequence_length', 100)
            num_samples = 32
            
            train_data = metrics_buffer.get_training_batch(
                num_samples=num_samples, 
                seq_len=seq_len
            )
            
            if len(train_data) == 0:
                logger.warning("Empty training batch received. Skipping round.")
                continue

            result = client.train_round(train_data)
            logger.info(
                f"Completed Federated Learning round {round_idx + 1} : "
                f"loss={result['loss']:.4f}, samples={result['num_samples']}"
            )
        except Exception as exc:
            logger.error(f"Failed to complete Federated Learning round {round_idx + 1}: {exc}")
            
            # Brief pause before next round attempt
            time.sleep(retry_delay)
            
            # Re-connect if channel dropped
            try:
                client.connect(host, port)
                client.register()
            except Exception:
                pass

    logger.info(f"Node {client_id} finished all rounds.")
    
    # Stop background metric buffering and disconnect
    metrics_buffer.stop()
    client.disconnect()


if __name__ == '__main__':
    main()
