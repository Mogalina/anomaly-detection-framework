import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .models import LSTMAnomalyDetector
from utils.preprocessing import sliding_window, normalize_data
from utils.logger import get_logger
from utils.metrics import get_metrics
from utils.config import get_config


logger = get_logger(__name__)


class EdgeDetector:
    """
    Edge-local anomaly detector with real-time inference.
    
    Implements LSTM-based anomaly detection with adaptive thresholding
    and inference optimization for low-latency detection.
    """
    
    def __init__(
        self,
        service_name: str,
        model: Optional[LSTMAnomalyDetector] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize edge detector.
        
        Args:
            service_name: Name of the service being monitored
            model: Pre-trained model, if None creates new one
            config: Configuration dictionary
        """
        self.service_name = service_name
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        # Model configuration
        edge_config = self.config.get('edge', {})
        model_config = edge_config.get('model', {})
        
        self.input_size = model_config.get('input_size', 10)
        self.sequence_length = model_config.get('sequence_length', 100)
        
        # Initialize model
        if model is None:
            self.model = LSTMAnomalyDetector(
                input_size=self.input_size,
                hidden_size=model_config.get('hidden_size', 64),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2)
            )
        else:
            self.model = model
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Quantization for faster inference
        inference_config = edge_config.get('inference', {})
        if inference_config.get('quantization', True):
            self._quantize_model()
        
        self.model.eval()
        
        # Detection configuration
        detection_config = edge_config.get('detection', {})
        self.threshold = detection_config.get('initial_threshold', 3.0)
        self.min_anomaly_duration = detection_config.get('min_anomaly_duration', 3)
        self.smoothing_window = detection_config.get('smoothing_window', 5)
        
        # Data normalization
        self.scaler = None
        
        # Sliding window buffer
        data_config = edge_config.get('data', {})
        self.window_size = data_config.get('window_size', 100)
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Anomaly tracking
        self.anomaly_scores = deque(maxlen=1000)
        self.recent_anomalies = deque(maxlen=100)
        self.consecutive_anomalies = 0
        
        # Batch inference
        self.batch_size = inference_config.get('batch_size', 32)
        self.inference_buffer = []
        
        logger.info(
            f"EdgeDetector initialized for service={service_name}, "
            f"device={self.device}, threshold={self.threshold}"
        )
    
    def _quantize_model(self):
        """Apply dynamic quantization to model for faster inference."""
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.LSTM, torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantized successfully")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, continuing without quantization")
    
    def update_data(self, metrics: np.ndarray) -> None:
        """
        Update data buffer with new metrics.
        
        Args:
            metrics: Array of shape (n_features,) or (1, n_features)
        """
        if metrics.ndim == 1:
            metrics = metrics.reshape(1, -1)
        
        # Ensure correct feature dimension
        if metrics.shape[1] != self.input_size:
            logger.warning(
                f"Feature size mismatch: expected {self.input_size}, "
                f"got {metrics.shape[1]}"
            )
            # Pad or truncate as needed
            if metrics.shape[1] < self.input_size:
                padding = np.zeros((1, self.input_size - metrics.shape[1]))
                metrics = np.concatenate([metrics, padding], axis=1)
            else:
                metrics = metrics[:, :self.input_size]
        
        self.data_buffer.append(metrics[0])
    
    def detect(self, metrics: Optional[np.ndarray] = None) -> Dict:
        """
        Detect anomalies in current data.
        
        Args:
            metrics: Optional new metrics to add before detection
        
        Returns:
            Detection result dictionary
        """
        if metrics is not None:
            self.update_data(metrics)
        
        # Check if we have enough data
        if len(self.data_buffer) < self.sequence_length:
            return {
                'is_anomaly': False,
                'score': 0.0,
                'threshold': self.threshold,
                'message': 'Insufficient data'
            }
        
        # Prepare data
        data = np.array(list(self.data_buffer))
        
        # Normalize
        if self.scaler is None:
            data_normalized, self.scaler = normalize_data(data, method='standard')
        else:
            data_normalized, _ = normalize_data(
                data, method='standard', scaler=self.scaler, fit=False
            )
        
        # Extract window
        window = data_normalized[-self.sequence_length:]
        
        # Inference
        start_time = time.time()
        score = self._compute_anomaly_score(window)
        inference_time = time.time() - start_time
        
        # Record metrics
        self.metrics.record_inference_time('lstm', inference_time)
        self.metrics.set_anomaly_score(
            self.service_name,
            'reconstruction_error',
            float(score)
        )
        
        # Store score
        self.anomaly_scores.append(score)
        
        # Apply smoothing
        if len(self.anomaly_scores) >= self.smoothing_window:
            recent_scores = list(self.anomaly_scores)[-self.smoothing_window:]
            smoothed_score = np.mean(recent_scores)
        else:
            smoothed_score = score
        
        # Detect anomaly
        is_anomaly = smoothed_score > self.threshold
        
        # Track consecutive anomalies
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
        
        # Confirm anomaly only if it persists
        confirmed_anomaly = (
            is_anomaly and
            self.consecutive_anomalies >= self.min_anomaly_duration
        )
        
        if confirmed_anomaly:
            anomaly_event = {
                'service': self.service_name,
                'timestamp': time.time(),
                'score': float(smoothed_score),
                'threshold': self.threshold,
                'severity': self._compute_severity(smoothed_score)
            }
            self.recent_anomalies.append(anomaly_event)
            
            # Record metrics
            self.metrics.record_anomaly(
                self.service_name,
                anomaly_event['severity']
            )
            
            logger.warning(
                f"Anomaly detected in {self.service_name}: "
                f"score={smoothed_score:.3f}, threshold={self.threshold:.3f}"
            )
        
        return {
            'is_anomaly': confirmed_anomaly,
            'score': float(smoothed_score),
            'raw_score': float(score),
            'threshold': self.threshold,
            'consecutive_count': self.consecutive_anomalies,
            'inference_time': inference_time,
            'timestamp': time.time()
        }
    
    def _compute_anomaly_score(self, window: np.ndarray) -> float:
        """
        Compute anomaly score for a window.
        
        Args:
            window: Data window of shape (sequence_length, n_features)
        
        Returns:
            Anomaly score
        """
        # Convert to tensor
        x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
        
        # Compute reconstruction error
        with torch.no_grad():
            error = self.model.compute_reconstruction_error(x, reduction='mean')
        
        return error.cpu().item()
    
    def _compute_severity(self, score: float) -> str:
        """
        Compute anomaly severity based on score.
        
        Args:
            score: Anomaly score
        
        Returns:
            Severity level ('low', 'medium', 'high', 'critical')
        """
        if score < self.threshold * 1.5:
            return 'low'
        elif score < self.threshold * 2.0:
            return 'medium'
        elif score < self.threshold * 3.0:
            return 'high'
        else:
            return 'critical'
    
    def update_threshold(self, new_threshold: float) -> None:
        """
        Update detection threshold.
        
        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.threshold
        self.threshold = new_threshold
        
        logger.info(
            f"Threshold updated for {self.service_name}: "
            f"{old_threshold:.3f} -> {new_threshold:.3f}"
        )
        
        self.metrics.set_threshold(self.service_name, new_threshold)
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.anomaly_scores:
            return {}
        
        scores = list(self.anomaly_scores)
        
        return {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'threshold': self.threshold,
            'recent_anomalies': len(self.recent_anomalies),
            'buffer_size': len(self.data_buffer)
        }
    
    def train(
        self,
        train_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train the anomaly detection model.
        
        Args:
            train_data: Training data of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Training history
        """
        logger.info(f"Training model on {len(train_data)} samples")
        
        # Normalize data
        train_data_normalized, self.scaler = normalize_data(train_data)
        
        # Create windows
        windows, _ = sliding_window(
            train_data_normalized,
            window_size=self.sequence_length,
            stride=1
        )
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(windows)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training setup
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstruction = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        
        # Calibrate threshold on training data
        self._calibrate_threshold(windows)
        
        logger.info("Training completed")
        return history
    
    def _calibrate_threshold(self, data: np.ndarray) -> None:
        """
        Calibrate detection threshold based on training data.
        
        Args:
            data: Training windows
        """
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                x = torch.FloatTensor(batch).to(self.device)
                error = self.model.compute_reconstruction_error(x, reduction='mean')
                scores.extend(error.cpu().numpy())
        
        # Set threshold at 95th percentile
        percentile = self.config.get('edge', {}).get('detection', {}).get(
            'threshold_percentile', 95
        )
        self.threshold = float(np.percentile(scores, percentile))
        
        logger.info(f"Threshold calibrated to {self.threshold:.3f} ({percentile}th percentile)")
