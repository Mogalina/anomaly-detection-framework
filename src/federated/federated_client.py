import torch
import numpy as np
from typing import Dict, Optional
import time
import zlib

from edge.models import LSTMAnomalyDetector
from utils.logger import get_logger
from utils.metrics import get_metrics
from utils.config import get_config


logger = get_logger(__name__)


class FederatedClient:
    """
    Federated learning client for local model training.
    
    Implements local training, gradient compression, and
    differential privacy.
    """
    
    def __init__(
        self,
        client_id: str,
        model: LSTMAnomalyDetector,
        config: Optional[Dict] = None
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model: Local model instance
            config: Configuration dictionary
        """
        self.client_id = client_id
        self.model = model
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        client_config = self.config.get('federated', {}).get('client', {})
        
        # Training configuration
        self.epochs_per_round = client_config.get('epochs_per_round', 5)
        self.batch_size = client_config.get('batch_size', 64)
        self.learning_rate = client_config.get('learning_rate', 0.001)
        
        # Gradient compression
        compression_config = client_config.get('gradient_compression', {})
        self.compression_enabled = compression_config.get('enabled', True)
        self.compression_method = compression_config.get('method', 'topk')
        self.compression_ratio = compression_config.get('compression_ratio', 0.1)
        
        # Differential privacy
        dp_config = client_config.get('differential_privacy', {})
        self.dp_enabled = dp_config.get('enabled', False)
        self.noise_multiplier = dp_config.get('noise_multiplier', 1.0)
        self.max_grad_norm = dp_config.get('max_grad_norm', 1.0)
        
        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.round_number = 0
        self.initial_params = None
        
        logger.info(
            f"FederatedClient initialized: id={client_id}, "
            f"compression={self.compression_enabled}, dp={self.dp_enabled}"
        )
    
    def train_round(
        self,
        train_data: np.ndarray,
        global_model_params: Optional[Dict] = None,
        system_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Train for one federated round.
        
        Args:
            train_data: Training data windows
            global_model_params: Global model parameters to start from
            system_metrics: Optional CPU, Memory, Disk, Network metrics from this
                edge/node for root cause analysis at the coordinator.
        
        Returns:
            Training results including model update and metrics (with system_metrics).
        """
        start_time = time.time()
        if system_metrics is None:
            system_metrics = {}
        
        # Update model with global parameters
        if global_model_params is not None:
            self._load_global_params(global_model_params)
        
        # Store initial parameters for computing delta
        self.initial_params = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_data)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training setup
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        criterion = torch.nn.MSELoss()
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.epochs_per_round):
            epoch_loss = 0.0
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                optimizer.zero_grad()
                
                reconstruction = self.model(x)
                loss = criterion(reconstruction, x)
                
                loss.backward()
                
                # Gradient clipping for DP
                if self.dp_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(dataloader)
        
        avg_loss = total_loss / self.epochs_per_round
        
        # Get model update (delta from initial params)
        model_update = self._compute_model_update()
        
        # Apply differential privacy noise
        if self.dp_enabled:
            model_update = self._add_dp_noise(model_update)
        
        # Compress update
        if self.compression_enabled:
            model_update, compression_stats = self._compress_update(model_update)
        else:
            compression_stats = {'compression_ratio': 1.0}
        
        training_time = time.time() - start_time

        # Metrics (include system_metrics for root cause analysis at coordinator)
        results = {
            'model_update': model_update,
            'num_samples': len(train_data),
            'loss': avg_loss,
            'training_time': training_time,
            'round': self.round_number,
            'compression_stats': compression_stats,
            'system_metrics': system_metrics,
            'metrics': {
                'loss': avg_loss,
                'training_time': training_time,
                'round': self.round_number,
                'compression_stats': compression_stats,
                'system_metrics': system_metrics,
            },
        }

        self.round_number += 1
        
        logger.info(
            f"Round {self.round_number} completed: "
            f"loss={avg_loss:.4f}, time={training_time:.2f}s, "
            f"compression={compression_stats['compression_ratio']:.2%}"
        )
        
        # Record bandwidth
        update_size = self._estimate_size(model_update)
        self.metrics.record_fl_bandwidth('upload', update_size)
        
        return results
    
    def _load_global_params(self, params: Dict) -> None:
        """
        Load global model parameters.
        
        Args:
            params: Serialized parameters
        """
        state_dict = {}
        for key, value in params.items():
            state_dict[key] = torch.FloatTensor(value)
        
        self.model.load_state_dict(state_dict)
        
        # Record bandwidth
        params_size = self._estimate_size(params)
        self.metrics.record_fl_bandwidth('download', params_size)
    
    def _compute_model_update(self) -> Dict:
        """
        Compute model update (delta from initial parameters).
        
        Returns:
            Model update dictionary
        """
        current_params = self.model.state_dict()
        update = {}
        
        for key in current_params.keys():
            delta = current_params[key] - self.initial_params[key]
            update[key] = delta.cpu().numpy().tolist()
        
        return update
    
    def _add_dp_noise(self, update: Dict) -> Dict:
        """
        Add differential privacy noise to model update.
        
        Args:
            update: Model update
        
        Returns:
            Noisy update
        """
        noisy_update = {}
        
        for key, value in update.items():
            param = np.array(value)
            
            # Add Gaussian noise
            noise = np.random.normal(
                0,
                self.noise_multiplier * self.max_grad_norm,
                size=param.shape
            )
            
            noisy_update[key] = (param + noise).tolist()
        
        return noisy_update
    
    def _compress_update(self, update: Dict) -> tuple:
        """
        Compress model update using top-k sparsification.
        
        Args:
            update: Model update
        
        Returns:
            Tuple of (compressed_update, compression_stats)
        """
        if self.compression_method == 'topk':
            return self._topk_compression(update)
        else:
            return update, {'compression_ratio': 1.0}
    
    def _topk_compression(self, update: Dict) -> tuple:
        """
        Top-k sparsification with error feedback.
        
        Args:
            update: Model update
        
        Returns:
            Tuple of (compressed_update, stats)
        """
        compressed = {}
        total_params = 0
        kept_params = 0
        
        for key, value in update.items():
            param = np.array(value)
            total_params += param.size
            
            # Flatten parameter
            flat_param = param.flatten()
            
            # Compute k
            k = max(1, int(len(flat_param) * self.compression_ratio))
            kept_params += k
            
            # Get top-k indices
            abs_param = np.abs(flat_param)
            topk_indices = np.argpartition(abs_param, -k)[-k:]
            
            # Create sparse representation
            compressed[key] = {
                'indices': topk_indices.tolist(),
                'values': flat_param[topk_indices].tolist(),
                'shape': list(param.shape)
            }
        
        compression_ratio = kept_params / total_params
        
        stats = {
            'compression_ratio': compression_ratio,
            'total_params': total_params,
            'kept_params': kept_params
        }
        
        return compressed, stats
    
    def _estimate_size(self, data: Dict) -> int:
        """
        Estimate size of data in bytes.
        
        Args:
            data: Dictionary to estimate
        
        Returns:
            Estimated size in bytes
        """
        import json
        serialized = json.dumps(data)
        return len(serialized.encode('utf-8'))
    
    def evaluate(self, test_data: np.ndarray) -> Dict:
        """
        Evaluate local model.
        
        Args:
            test_data: Test data
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_data)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstruction = self.model(x)
                loss = criterion(reconstruction, x)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'num_samples': len(test_data)
        }
