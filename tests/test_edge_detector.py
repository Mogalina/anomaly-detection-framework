import pytest
import numpy as np

from edge.edge_detector import EdgeDetector
from edge.models import LSTMAnomalyDetector


class TestEdgeDetector:
    """Test suite for EdgeDetector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = LSTMAnomalyDetector(input_size=5, hidden_size=32, num_layers=1)
        self.detector = EdgeDetector('test-service', model=self.model)
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.service_name == 'test-service'
        assert self.detector.input_size == 5
        assert self.detector.threshold > 0
    
    def test_update_data(self):
        """Test data buffer update."""
        metrics = np.random.randn(5)
        self.detector.update_data(metrics)
        
        assert len(self.detector.data_buffer) == 1
    
    def test_detect_insufficient_data(self):
        """Test detection with insufficient data."""
        metrics = np.random.randn(5)
        result = self.detector.detect(metrics)
        
        assert result['is_anomaly'] == False
        assert result['message'] == 'Insufficient data'
    
    def test_detect_with_sufficient_data(self):
        """Test detection with sufficient data."""
        # Fill buffer with normal data
        for _ in range(150):
            metrics = np.random.randn(5)
            self.detector.update_data(metrics)
        
        # Detect
        result = self.detector.detect()
        
        assert 'is_anomaly' in result
        assert 'score' in result
        assert 'threshold' in result
    
    def test_update_threshold(self):
        """Test threshold update."""
        old_threshold = self.detector.threshold
        new_threshold = old_threshold * 1.5
        
        self.detector.update_threshold(new_threshold)
        
        assert self.detector.threshold == new_threshold
    
    def test_training(self):
        """Test model training."""
        # Generate training data
        train_data = np.random.randn(1000, 5)
        
        history = self.detector.train(
            train_data,
            epochs=5,
            batch_size=32
        )
        
        assert 'loss' in history
        assert len(history['loss']) == 5
        assert self.detector.threshold > 0
