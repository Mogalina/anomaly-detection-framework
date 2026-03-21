import json
import os
import sqlite3
import time
import grpc
import numpy as np
import torch

from pathlib import Path
from typing import Dict, Optional
from edge.models import LSTMAnomalyDetector
from federated.proto import fl_service_pb2 as pb2
from federated.proto import fl_service_pb2_grpc as pb2_grpc
from utils.logger import get_logger
from utils.metrics import get_metrics
from utils.config import get_config
from utils.system_metrics_collector import SystemMetricsCollector
from utils.compression import CompressionType, unpack_state_dict, pack_state_dict


logger = get_logger(__name__)


class FederatedClient:
    """
    Federated learning client for local model training.
    
    Implements local training, gradient compression, differential privacy, 
    and gRPC-based communication with the coordinator.

    Compression algorithm is selected automatically depending on the CPU load 
    at the time of transmission.
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
        self.metrics_collector = SystemMetricsCollector(service_name=client_id)

        # Node profile: 'lightweight' (default) or 'standard'
        self.profile = os.environ.get('NODE_PROFILE', 'lightweight').lower()

        client_config = self.config.get('federated', {}).get('client', {})

        # Training configuration
        self.epochs_per_round = client_config.get('epochs_per_round', 5)
        self.batch_size = client_config.get('batch_size', 64)
        self.learning_rate = client_config.get('learning_rate', 0.001)

        # Retry / reconnect
        self.max_retries = client_config.get('max_retries', 10)
        self.retry_delay = client_config.get('retry_delay_seconds', 5)

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

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # gRPC channel / stub
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[pb2_grpc.FLServiceStub] = None
        self._coordinator_address: Optional[str] = None

        # Training state
        self.round_number = 0
        self.initial_params: Optional[Dict] = None

        # Local database path for standard profile only
        if self.profile == 'standard':
            # Create local database directory
            local_db_dir = Path(client_config.get('local_db_path', '~/.adf/')).expanduser()
            local_db_dir.mkdir(parents=True, exist_ok=True)

            # Set local database path
            self._local_db_path = str(local_db_dir / f"client_{self.client_id}.db")

            # Initialize local database
            self._init_local_db()
        else:
            # Lightweight profile with no local database
            self._local_db_path = None

        logger.info(
            f"Federated client initialised: id={client_id}, profile={self.profile}, "
            f"compression={self.compression_enabled}, dp={self.dp_enabled}"
        )

    def connect(
        self,
        host: str,
        port: int
    ) -> None:
        """
        Open a gRPC channel to the coordinator. Retries with exponential back-off 
        until the coordinator is reachable or `max_retries` is exceeded.

        Args:
            host: Coordinator hostname or IP
            port: Coordinator gRPC port

        Raises:
            ConnectionError: If the coordinator is not reachable after all retries
        """
        address = f"{host}:{port}"
        self._coordinator_address = address

        for attempt in range(1, self.max_retries + 1):
            try:
                channel = grpc.insecure_channel(address)
                stub = pb2_grpc.FLServiceStub(channel)

                # Lightweight connectivity check
                grpc.channel_ready_future(channel).result(timeout=5)

                self.channel = channel
                self.stub = stub

                logger.info(f"gRPC channel opened to {address} (attempt {attempt})")
                return
            except Exception as exc:
                # Calculate delay with exponential back-off
                delay = self.retry_delay * (2 ** (attempt - 1))

                logger.warning(
                    f"Coordinator not reachable after {attempt}/{self.max_retries} attempts: {exc}"
                )

                if attempt < self.max_retries:
                    logger.warning(
                        f"Retrying in {delay:.0f}s, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(delay)

        raise ConnectionError(
            f"Could not connect to coordinator at {address} after {self.max_retries} attempts"
        )

    def register(self) -> bool:
        """
        Register this client with the coordinator and download the global model.

        Returns:
            True on success, False on failure
        """
        if self.stub is None:
            raise RuntimeError("gRPC channel not opened")

        # Register client with coordinator
        metadata = {
            'device': str(self.device),
            'epochs_per_round': self.epochs_per_round,
            'dp_enabled': self.dp_enabled,
        }
        request = pb2.RegisterRequest(
            client_id=self.client_id,
            metadata_json=json.dumps(metadata),
        )

        try:
            response: pb2.RegisterResponse = self.stub.Register(request)
        except grpc.RpcError as exc:
            logger.error(f"Registration failed: {exc}")
            return False

        if response.status != 'ok':
            logger.error(f"Registration rejected: {response.status}")
            return False

        # Load the global model returned by the coordinator
        algorithm = CompressionType(response.compression)
        state_dict = unpack_state_dict(response.global_model, algorithm)
        self.model.load_state_dict(state_dict)

        # Record downloaded bytes
        self.metrics.record_fl_bandwidth('download', len(response.global_model))

        logger.info(
            f"Registered successfully, global model loaded "
            f"(round={response.current_round}, compression={algorithm.name})"
        )

        return True

    def disconnect(self) -> None:
        """Close the gRPC channel."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None
    
    def train_round(
        self,
        train_data: np.ndarray,
        global_model_params: Optional[Dict] = None,
        system_metrics: Optional[Dict] = None
    ) -> Dict:
        """
        Execute one federated training round and submit the update.
        
        Args:
            train_data: Training data windows of shape (num_samples, sequence_length, features)
            global_model_params: Global model parameters to start from
            system_metrics: Optional pre-collected system metrics dictionary; if None,
                metrics are collected fresh from the node
        
        Returns:
            Training results including model update and metrics
        """
        start_time = time.time()

        # Collect system metrics for adaptive compression
        if system_metrics is None:
            system_metrics = self.metrics_collector.collect()

        cpu_usage_percent = float(system_metrics.get('cpu_usage_percent', 0.0))
        
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
        
        # Serialize model update
        send_state = self._update_dict_to_state_dict(model_update)
        
        # Pack and adaptively compress the state dictionary
        payload_bytes, wire_algorithm = pack_state_dict(send_state, cpu_usage_percent)

        training_time = time.time() - start_time
        self.round_number += 1

        metrics_payload = {
            'loss': avg_loss,
            'training_time': training_time,
            'round': self.round_number,
            'compression_stats': compression_stats,
            'system_metrics': system_metrics,
        }

        # Standard profile does persist pending update to database before sending
        # so it can be replayed if the node crashes before transmitting
        if self.profile == 'standard':
            self._save_pending_update(
                self.round_number,
                payload_bytes,
                int(wire_algorithm),
                len(train_data),
                metrics_payload
            )

        # Send update to coordinator
        if self.stub is not None:
            self._submit_update_grpc(
                payload_bytes,
                wire_algorithm,
                len(train_data),
                metrics_payload
            )
        else:
            logger.debug("gRPC stub does not exist, training update not transmitted")
        
        # Record bandwidth
        update_size = self._estimate_size(model_update)
        self.metrics.record_fl_bandwidth('upload', update_size)

        logger.info(
            f"Round {self.round_number} completed: "
            f"loss={avg_loss:.4f}, time={training_time:.2f}s, "
            f"compression={compression_stats['compression_ratio']:.2%}, "
            f"wire={wire_algorithm.name}, payload={len(payload_bytes)/1024:.1f} KB, "
            f"cpu={cpu_usage_percent:.0f}%"
        )

        # Metrics (include system metrics for root cause analysis at coordinator)
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
        
        return results

    def _submit_update_grpc(
        self,
        payload: bytes,
        algorithm: CompressionType,
        num_samples: int,
        metrics: Dict,
    ) -> None:
        """
        Send the model update to the coordinator using the gRPC stub.
        If the coordinator is not available (possibly restarted), re-registers
        and retries once.

        Args:
            payload: Compressed model update bytes
            algorithm: Compression algorithm used
            num_samples: Number of samples used for training
            metrics: Training metrics
        """
        if self.stub is None:
            logger.error("Cannot submit update: gRPC stub not available")
            return

        request = pb2.UpdateRequest(
            client_id=self.client_id,
            round=self.round_number,
            num_samples=num_samples,
            payload=payload,
            compression=int(algorithm),
            metrics_json=json.dumps(metrics, default=str),
        )

        # Retry once if coordinator is not available
        for attempt in range(2):
            try:
                resp: pb2.UpdateResponse = self.stub.SubmitUpdate(request)
                if resp.status == 'ok':
                    # Mark update as sent in local database if standard profile
                    self._mark_update_sent(self.round_number)
                    return
                if 'not registered' in resp.message.lower() and attempt == 0:
                    logger.warning("Re-registering with coordinator due to lost registration")
                    self.register()
                    continue
                logger.warning(f"Coordinator rejected update: {resp.message}")
                return
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.NOT_FOUND and attempt == 0:
                    logger.warning("Re-registering with coordinator due to not found error")
                    self.register()
                    continue
                logger.error(f"Submission update failed: {exc}")
                return

    def fetch_global_model(self) -> bool:
        """
        Download the latest global model from the coordinator.

        Returns:
            True on success
        """
        if self.stub is None:
            raise RuntimeError("Connect before fetching global model")

        request = pb2.GlobalModelRequest(client_id=self.client_id)
        try:
            resp: pb2.GlobalModelResponse = self.stub.GetGlobalModel(request)
        except grpc.RpcError as exc:
            logger.error(f"Downloading global model failed: {exc}")
            return False

        # Unpack and load global model
        algorithm = CompressionType(resp.compression)
        state_dict = unpack_state_dict(resp.payload, algorithm)
        self.model.load_state_dict(state_dict)

        # Record bandwidth
        self.metrics.record_fl_bandwidth('download', len(resp.payload))

        logger.info(
            f"Global model downloaded: round={resp.current_round}, "
            f"compression={algorithm.name}, size={len(resp.payload)/1024:.1f} KB"
        )

        return True
    
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
        Compress model update.
        
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

    @staticmethod
    def _update_dict_to_state_dict(update: Dict) -> Dict:
        """
        Convert a sparse update dictionary back into a dense state dictionary for
        wire serialisation.

        Args:
            update: Sparse update dictionary

        Returns:
            Dense state dictionary
        """
        state = {}

        for key, val in update.items():
            if isinstance(val, dict) and 'indices' in val:
                flat = np.zeros(int(np.prod(val['shape'])), dtype=np.float32)
                flat[val['indices']] = val['values']
                state[key] = torch.from_numpy(flat.reshape(val['shape']))
            else:
                state[key] = torch.FloatTensor(val)
        
        return state
    
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

    def _init_local_db(self) -> None:
        """Create the local database table for pending updates."""
        if not self._local_db_path:
            return
        
        with sqlite3.connect(self._local_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_updates (
                    round_num INTEGER PRIMARY KEY,
                    payload BLOB NOT NULL,
                    algorithm INTEGER NOT NULL,
                    num_samples INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    sent INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    def _save_pending_update(
        self,
        round_num: int,
        payload: bytes,
        algorithm: int,
        num_samples: int,
        metrics: Dict,
    ) -> None:
        """
        Persist a pending (unsent) update to local database.
        
        Args:
            round_num: Round number
            payload: Compressed model update bytes
            algorithm: Compression algorithm used
            num_samples: Number of samples used for training
            metrics: Training metrics
        """
        if not self._local_db_path:
            return

        try:
            with sqlite3.connect(self._local_db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO pending_updates
                       (round_num, payload, algorithm, num_samples, metrics_json, sent)
                       VALUES (?, ?, ?, ?, ?, 0)""",
                    (
                        round_num,
                        payload,
                        algorithm,
                        num_samples,
                        json.dumps(metrics, default=str)
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"Could not save pending update to database: {exc}")

    def _mark_update_sent(self, round_num: int) -> None:
        """
        Mark a pending update as successfully transmitted.
        
        Args:
            round_num: Round number
        """
        if not self._local_db_path:
            return

        try:
            with sqlite3.connect(self._local_db_path) as conn:
                conn.execute(
                    "UPDATE pending_updates SET sent=1 WHERE round_num=?",
                    (round_num,),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"Could not mark update as sent: {exc}")

    def _restore_pending_update(self) -> None:
        """
        On startup, check local database for any unsent update from a previous
        run and re-transmit it. This is only applicable for standard profile.
        """
        if not self._local_db_path or self.stub is None:
            return

        try:
            with sqlite3.connect(self._local_db_path) as conn:
                row = conn.execute(
                    "SELECT round_num, payload, algorithm, num_samples, metrics_json "
                    "FROM pending_updates WHERE sent=0 ORDER BY round_num DESC LIMIT 1"
                ).fetchone()

            if row is None:
                return

            round_num, payload, algorithm, num_samples, metrics_json = row
            
            metrics = json.loads(metrics_json)
            compression_algorithm = CompressionType(algorithm)

            logger.info(
                f"Replaying unsent update from round {round_num}: "
                f"{len(payload)/1024:.2f} KB, {compression_algorithm.name}"
            )
            
            self._submit_update_grpc(
                payload=payload,
                compression_algorithm=compression_algorithm,
                num_samples=num_samples,
                metrics=metrics
            )

        except Exception as exc:
            logger.warning(f"Could not restore pending update: {exc}")
