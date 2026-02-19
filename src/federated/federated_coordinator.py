import torch
import grpc
from concurrent import futures
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

from edge.models import LSTMAnomalyDetector, AutoEncoder
from utils.logger import get_logger
from utils.metrics import get_metrics
from utils.config import get_config
from utils.serialization import save_model, load_model


logger = get_logger(__name__)


class FederatedCoordinator:
    """
    Central coordinator for federated learning.
    
    Implements FedAvg aggregation with poisoning detection
    and model validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize federated coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        fed_config = self.config.get('federated', {}).get('coordinator', {})
        
        # Server configuration
        self.host = fed_config.get('host', '0.0.0.0')
        self.port = fed_config.get('port', 50051)
        self.num_rounds = fed_config.get('num_rounds', 100)
        self.min_clients_per_round = fed_config.get('min_clients_per_round', 3)
        self.fraction_fit = fed_config.get('fraction_fit', 0.8)
        self.fraction_evaluate = fed_config.get('fraction_evaluate', 0.5)
        self.aggregation_strategy = fed_config.get('aggregation_strategy', 'fedavg')
        self.staleness_tolerance = fed_config.get('staleness_tolerance', 2)
        
        # Global model
        model_config = self.config.get('edge', {}).get('model', {})
        self.global_model = LSTMAnomalyDetector(
            input_size=model_config.get('input_size', 10),
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        
        # Client tracking
        self.registered_clients: Dict[str, Dict] = {}
        self.client_models: Dict[str, Dict] = {}
        self.client_metrics: Dict[str, List] = defaultdict(list)
        
        # Round tracking
        self.current_round = 0
        self.round_history = []
        
        # Poisoning detection
        poison_config = self.config.get('federated', {}).get('poisoning_detection', {})
        self.poisoning_detection_enabled = poison_config.get('enabled', True)
        self.zscore_threshold = poison_config.get('zscore_threshold', 3.0)
        self.autoencoder_threshold = poison_config.get('autoencoder_threshold', 0.8)
        
        # Initialize autoencoder for model validation
        if self.poisoning_detection_enabled:
            param_count = sum(p.numel() for p in self.global_model.parameters())
            self.model_validator = AutoEncoder(
                input_dim=param_count,
                encoding_dim=64
            )
            self.model_validator.to(self.device)
            self.validation_samples = []
        
        logger.info(
            f"FederatedCoordinator initialized: "
            f"rounds={self.num_rounds}, min_clients={self.min_clients_per_round}"
        )
    
    def register_client(self, client_id: str, metadata: Dict) -> Dict:
        """
        Register a new client.
        
        Args:
            client_id: Unique client identifier
            metadata: Client metadata
        
        Returns:
            Registration response
        """
        self.registered_clients[client_id] = {
            'metadata': metadata,
            'registered_at': time.time(),
            'last_seen': time.time(),
            'rounds_participated': 0
        }
        
        logger.info(f"Client registered: {client_id}")
        
        return {
            'status': 'success',
            'global_model': self._serialize_model(self.global_model),
            'current_round': self.current_round
        }
    
    def receive_update(
        self,
        client_id: str,
        model_update: Dict,
        num_samples: int,
        metrics: Dict
    ) -> Dict:
        """
        Receive model update from client.
        
        Args:
            client_id: Client identifier
            model_update: Serialized model parameters
            num_samples: Number of training samples
            metrics: Training metrics
        
        Returns:
            Response dictionary
        """
        if client_id not in self.registered_clients:
            return {'status': 'error', 'message': 'Client not registered'}
        
        # Update client info
        self.registered_clients[client_id]['last_seen'] = time.time()
        self.registered_clients[client_id]['rounds_participated'] += 1
        
        # Store update
        self.client_models[client_id] = {
            'model_update': model_update,
            'num_samples': num_samples,
            'metrics': metrics,
            'round': self.current_round,
            'timestamp': time.time()
        }
        
        # Store metrics
        self.client_metrics[client_id].append(metrics)
        
        logger.info(
            f"Received update from {client_id}: "
            f"samples={num_samples}, round={self.current_round}"
        )
        
        return {
            'status': 'success',
            'message': 'Update received'
        }

    def get_client_metrics(self, client_id: str) -> Optional[Dict]:
        """
        Get the latest metrics (including CPU, Memory, Disk, Network) for a client.
        Used by root cause analysis to correlate anomalies with node-level metrics.
        """
        if not self.client_metrics.get(client_id):
            return None
        return self.client_metrics[client_id][-1]

    def get_all_clients_system_metrics(self) -> Dict[str, Dict]:
        """
        Get the latest system_metrics (CPU, Memory, Disk, Network) for all clients.
        Used by root cause analysis to determine which node contributed to an anomaly.
        """
        out = {}
        for client_id in self.client_metrics:
            latest = self.get_client_metrics(client_id)
            if latest and isinstance(latest.get('system_metrics'), dict):
                out[client_id] = latest['system_metrics']
        return out

    def aggregate_models(self) -> bool:
        """
        Aggregate client models using FedAvg.
        
        Returns:
            True if aggregation successful
        """
        if len(self.client_models) < self.min_clients_per_round:
            logger.warning(
                f"Insufficient clients for aggregation: "
                f"{len(self.client_models)} < {self.min_clients_per_round}"
            )
            return False
        
        start_time = time.time()
        
        # Filter out stale updates
        valid_updates = {}
        for client_id, update in self.client_models.items():
            if self.current_round - update['round'] <= self.staleness_tolerance:
                valid_updates[client_id] = update
        
        if len(valid_updates) < self.min_clients_per_round:
            logger.warning("Insufficient valid (non-stale) updates")
            return False
        
        # Poisoning detection
        if self.poisoning_detection_enabled:
            valid_updates = self._detect_poisoned_updates(valid_updates)
        
        if len(valid_updates) < self.min_clients_per_round:
            logger.warning("Insufficient updates after poisoning detection")
            return False
        
        # Aggregate using FedAvg
        total_samples = sum(u['num_samples'] for u in valid_updates.values())
        
        aggregated_state_dict = {}
        
        for client_id, update in valid_updates.items():
            weight = update['num_samples'] / total_samples
            client_state = update['model_update']
            
            for key, param in client_state.items():
                param_tensor = torch.FloatTensor(param).to(self.device)
                
                if key not in aggregated_state_dict:
                    aggregated_state_dict[key] = weight * param_tensor
                else:
                    aggregated_state_dict[key] += weight * param_tensor
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state_dict)
        
        # Update validation samples for autoencoder
        if self.poisoning_detection_enabled:
            self._update_validator(aggregated_state_dict)
        
        aggregation_time = time.time() - start_time
        
        # Record metrics
        self.metrics.record_fl_round()
        self.metrics.set_fl_clients(len(valid_updates))
        
        # Store round history
        round_info = {
            'round': self.current_round,
            'num_clients': len(valid_updates),
            'total_samples': total_samples,
            'aggregation_time': aggregation_time,
            'timestamp': time.time()
        }
        self.round_history.append(round_info)
        
        logger.info(
            f"Round {self.current_round} aggregation completed: "
            f"clients={len(valid_updates)}, time={aggregation_time:.2f}s"
        )
        
        # Clear client models for next round
        self.client_models.clear()
        
        return True
    
    def _detect_poisoned_updates(self, updates: Dict) -> Dict:
        """
        Detect and filter poisoned model updates.
        
        Args:
            updates: Dictionary of client updates
        
        Returns:
            Filtered updates dictionary
        """
        if len(updates) < 3:
            return updates  # Need minimum clients for statistical detection
        
        # Extract model parameters
        param_vectors = []
        client_ids = []
        
        for client_id, update in updates.items():
            params = []
            for param in update['model_update'].values():
                params.extend(param.flatten())
            param_vectors.append(params)
            client_ids.append(client_id)
        
        param_vectors = np.array(param_vectors)
        
        # Z-score based detection
        mean = np.mean(param_vectors, axis=0)
        std = np.std(param_vectors, axis=0) + 1e-7
        z_scores = np.abs((param_vectors - mean) / std)
        max_z_scores = np.max(z_scores, axis=1)
        
        # Autoencoder-based detection
        if len(self.validation_samples) >= 10:
            autoencoder_scores = self._compute_autoencoder_scores(param_vectors)
        else:
            autoencoder_scores = np.zeros(len(param_vectors))
        
        # Filter poisoned updates
        valid_updates = {}
        for i, client_id in enumerate(client_ids):
            is_poisoned = (
                max_z_scores[i] > self.zscore_threshold or
                autoencoder_scores[i] > self.autoencoder_threshold
            )
            
            if is_poisoned:
                logger.warning(
                    f"Poisoned update detected from {client_id}: "
                    f"zscore={max_z_scores[i]:.2f}, "
                    f"ae_score={autoencoder_scores[i]:.2f}"
                )
                self.metrics.record_poisoning_detection('statistical')
            else:
                valid_updates[client_id] = updates[client_id]
        
        return valid_updates
    
    def _compute_autoencoder_scores(self, param_vectors: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error using autoencoder.
        
        Args:
            param_vectors: Parameter vectors
        
        Returns:
            Anomaly scores
        """
        self.model_validator.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(param_vectors).to(self.device)
            reconstruction, _ = self.model_validator(x)
            errors = torch.mean((x - reconstruction) ** 2, dim=1)
        
        return errors.cpu().numpy()
    
    def _update_validator(self, state_dict: Dict) -> None:
        """
        Update autoencoder validator with new model.
        
        Args:
            state_dict: Model state dictionary
        """
        # Flatten parameters
        params = []
        for param in state_dict.values():
            params.extend(param.cpu().flatten().numpy())
        
        self.validation_samples.append(params)
        
        # Keep only recent samples
        max_samples = self.config.get('federated', {}).get(
            'poisoning_detection', {}
        ).get('validation_samples', 100)
        
        if len(self.validation_samples) > max_samples:
            self.validation_samples = self.validation_samples[-max_samples:]
        
        # Retrain validator periodically
        if len(self.validation_samples) >= 20 and len(self.validation_samples) % 10 == 0:
            self._train_validator()
    
    def _train_validator(self) -> None:
        """Train the autoencoder validator."""
        if len(self.validation_samples) < 20:
            return
        
        data = torch.FloatTensor(self.validation_samples).to(self.device)
        
        optimizer = torch.optim.Adam(self.model_validator.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        self.model_validator.train()
        
        for epoch in range(10):
            optimizer.zero_grad()
            reconstruction, _ = self.model_validator(data)
            loss = criterion(reconstruction, data)
            loss.backward()
            optimizer.step()
        
        self.model_validator.eval()
        
        logger.info("Autoencoder validator retrained")
    
    def _serialize_model(self, model: torch.nn.Module) -> Dict:
        """
        Serialize model parameters.
        
        Args:
            model: PyTorch model
        
        Returns:
            Serialized state dict
        """
        state_dict = model.state_dict()
        serialized = {}
        
        for key, param in state_dict.items():
            serialized[key] = param.cpu().numpy().tolist()
        
        return serialized
    
    def get_global_model(self) -> Dict:
        """
        Get current global model.
        
        Returns:
            Serialized global model
        """
        return self._serialize_model(self.global_model)
    
    def run_round(self) -> None:
        """Execute one federated learning round."""
        logger.info(f"Starting round {self.current_round + 1}/{self.num_rounds}")
        
        # Wait for client updates (in practice, this would be async)
        # For now, we just check if we have enough updates
        
        # Aggregate
        success = self.aggregate_models()
        
        if success:
            self.current_round += 1
        
        return success
    
    def save_global_model(self, path: str) -> None:
        """
        Save global model to disk.
        
        Args:
            path: Save path
        """
        metadata = {
            'round': self.current_round,
            'num_clients': len(self.registered_clients),
            'timestamp': time.time()
        }
        
        save_model(self.global_model, path, metadata=metadata)
        logger.info(f"Global model saved to {path}")
    
    def load_global_model(self, path: str) -> None:
        """
        Load global model from disk.
        
        Args:
            path: Model path
        """
        info = load_model(self.global_model, path, device=str(self.device))
        self.current_round = info.get('metadata', {}).get('round', 0)
        
        logger.info(f"Global model loaded from {path}, round={self.current_round}")
