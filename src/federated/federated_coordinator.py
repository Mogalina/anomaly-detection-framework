import json
import os
import time
import threading
import grpc
import numpy as np
import torch

from collections import defaultdict
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional
from edge.models import LSTMAnomalyDetector, AutoEncoder
from federated.proto import fl_service_pb2 as pb2
from federated.proto import fl_service_pb2_grpc as pb2_grpc
from utils.logger import get_logger
from utils.metrics import get_metrics
from utils.config import get_config
from utils.serialization import save_model, load_model
from utils.compression import CompressionType, pack_state_dict, unpack_state_dict
from utils.db import get_session, init_db
from utils.db_models import CoordinatorState, RegisteredClient, RoundHistory
from utils.cache import cache_client_update, delete_client_update, list_cached_updates


logger = get_logger(__name__)


class _FLServiceServicer(pb2_grpc.FLServiceServicer):
    """
    gRPC servicer that wires the RPC endpoints to the coordinator's internal state.
    """

    def __init__(self, coordinator: "FederatedCoordinator"):
        self.coordinator = coordinator

    def Register(
        self, 
        request: pb2.RegisterRequest, 
        context: grpc.ServicerContext
    ) -> pb2.RegisterResponse:
        try:
            metadata = json.loads(request.metadata_json) if request.metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}

        logger.info(f"gRPC client {request.client_id} registered")
        result = self.coordinator.register_client(request.client_id, metadata)

        if result['status'] != 'success':
            return pb2.RegisterResponse(status=result['status'])

        # Compress and send the global model
        payload, alg = pack_state_dict(
            self.coordinator.global_model.state_dict(), 0.0
        )

        logger.debug(
            f"gRPC client {request.client_id} registered: "
            f"sending global model ({alg.name}, {len(payload)/1024:.1f} KB)"
        )

        return pb2.RegisterResponse(
            status='ok',
            global_model=payload,
            current_round=self.coordinator.current_round,
            compression=int(alg),
        )

    def SubmitUpdate(
        self, 
        request: pb2.UpdateRequest, 
        context: grpc.ServicerContext
    ) -> pb2.UpdateResponse:
        if request.client_id not in self.coordinator.registered_clients:
            # Client not registered
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Client not registered')

            logger.error(f"gRPC client {request.client_id} not registered")

            return pb2.UpdateResponse(
                status='error', 
                message='Client not registered'
            )

        # Decompress payload
        alg = CompressionType(request.compression)

        try:
            state_dict = unpack_state_dict(request.payload, alg)
        except Exception as exc:
            logger.error(f"Decompression failed for {request.client_id}: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))

            return pb2.UpdateResponse(
                status='error', 
                message=str(exc)
            )

        # Convert state dictionary values to numpy lists expected by aggregation logic
        model_update = {k: v.numpy().tolist() for k, v in state_dict.items()}

        # Parse optional metrics (includes system metrics for root cause analysis)
        try:
            metrics = json.loads(request.metrics_json) if request.metrics_json else {}
        except json.JSONDecodeError:
            metrics = {}

        self.coordinator.receive_update(
            client_id=request.client_id,
            model_update=model_update,
            num_samples=request.num_samples,
            metrics=metrics,
        )

        logger.info(
            f"gRPC update from {request.client_id}: "
            f"round={request.round}, samples={request.num_samples}, "
            f"alg={alg.name}, size={len(request.payload)/1024:.1f} KB"
        )

        return pb2.UpdateResponse(status='ok')

    def GetGlobalModel(
        self, 
        request: pb2.GlobalModelRequest, 
        context: grpc.ServicerContext
    ) -> pb2.GlobalModelResponse:
        # Compress and send the global model
        payload, algorithm = pack_state_dict(
            self.coordinator.global_model.state_dict(), 0.0
        )

        return pb2.GlobalModelResponse(
            payload=payload,
            compression=int(algorithm),
            current_round=self.coordinator.current_round,
        )


class FederatedCoordinator:
    """
    Central coordinator for Federated Learning.
    
    Implements FedAvg aggregation with poisoning detection and model validation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FederatedCoordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        fed_config = self.config.get('federated', {}).get('coordinator', {})
        
        # Server configuration
        self.host = fed_config.get('host', '0.0.0.0')
        self.port = fed_config.get('port', 50051)
        self.max_workers = fed_config.get('max_workers', 10)
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
        self.round_history: List[Dict] = []
        
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

        # gRPC server handle
        self.grpc_server: Optional[grpc.Server] = None

        # Round timeout event
        self._round_ready = threading.Event()

        # Restore persisted state (round, model, clients) from database
        self._restore_state()

    def _restore_state(self) -> None:
        """Restore coordinator state from database on startup."""
        try:
            # Ensure tables exist
            init_db()
        except Exception as exc:
            logger.warning(f"Could not initialise DB schema: {exc}")
            return

        try:
            with get_session() as session:
                state = session.get(CoordinatorState, 1)
                if state is not None:
                    # Restore round number
                    self.current_round = state.current_round

                    # Restore global model
                    if state.global_model_path and Path(state.global_model_path).exists():
                        load_model(
                            self.global_model, 
                            state.global_model_path, 
                            device=str(self.device)
                        )
                        logger.info(
                            f"Restored coordinator state from database: "
                            f"round={self.current_round}, model={state.global_model_path}"
                        )
                    else:
                        logger.info(f"Restored round={self.current_round} (no model checkpoint found)")

                # Restore registered clients
                clients = session.query(RegisteredClient).all()
                for client in clients:
                    self.registered_clients[client.client_id] = {
                        'metadata': json.loads(client.metadata_json) if client.metadata_json else {},
                        'registered_at': client.registered_at,
                        'last_seen': client.last_seen,
                        'rounds_participated': client.rounds_participated,
                    }
                
                if clients:
                    logger.info(f"Reloaded {len(clients)} registered client(s) from database")

                # Merge any in-flight cached updates from Redis for current round
                cached = list_cached_updates(self.current_round)
                for client_id, update in cached.items():
                    if client_id not in self.client_models:
                        self.client_models[client_id] = update

                if cached:
                    logger.info(f"Restored {len(cached)} cached update(s) from Redis for round {self.current_round}")

        except Exception as exc:
            logger.warning(f"State restore failed (continuing with empty state): {exc}")

    def _checkpoint(self) -> None:
        """
        Persist coordinator state to database and model checkpoint file.
        Called after every successful Federated Learning round.
        """
        try:
            checkpoint_dir = Path(self.config.get('federated', {}).get('coordinator', {}).get('checkpoint_dir', 'checkpoints'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save global model checkpoint
            model_path = str(checkpoint_dir / 'global_model.pt')
            save_model(self.global_model, model_path, metadata={'round': self.current_round})

            # Save coordinator state to database
            with get_session() as session:
                state = session.get(CoordinatorState, 1)
                if state is None:
                    state = CoordinatorState(id=1)
                    session.add(state)
                state.current_round = self.current_round
                state.global_model_path = model_path
                state.updated_at = time.time()

            logger.debug(f"Checkpoint saved: round={self.current_round}, model={model_path}")

        except Exception as exc:
            logger.error(f"Checkpoint failed: {exc}")

    def serve(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ) -> None:
        """
        Start the gRPC server using a blocking call (run in a thread for async use).

        Args:
            host: Bind address (defaults to configuration value)
            port: Listen port (defaults to configuration value)
        """
        server_host = host or self.host
        server_port = port or self.port

        # Create gRPC server with thread pool executor
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))
        
        # Add Federated Learning service to gRPC server
        pb2_grpc.add_FLServiceServicer_to_server(_FLServiceServicer(self), self.grpc_server)
        
        # Add insecure port
        self.grpc_server.add_insecure_port(f"{server_host}:{server_port}")
        
        # Start gRPC server
        self.grpc_server.start()
        logger.info(f"gRPC Federated Learning server started on {server_host}:{server_port}")
        
        # Wait for gRPC server to terminate
        self.grpc_server.wait_for_termination()

    def stop(self, grace: float = 5.0) -> None:
        """
        Gracefully stop the gRPC server.

        Args:
            grace: Seconds to wait for in-flight RPCs to complete.
        """
        if self.grpc_server is not None:
            self.grpc_server.stop(grace)
            logger.info("gRPC Federated Learning server stopped")
    
    def register_client(self, client_id: str, metadata: Dict) -> Dict:
        """
        Register a new client and persist it in database so the coordinator can 
        repopulate its registry after a restart.

        Args:
            client_id: Unique client identifier
            metadata: Client metadata

        Returns:
            Registration response
        """
        now = time.time()

        # Get existing client
        existing = self.registered_clients.get(client_id)

        # Update client
        self.registered_clients[client_id] = {
            'metadata': metadata,
            'registered_at': existing['registered_at'] if existing else now,
            'last_seen': now,
            'rounds_participated': existing['rounds_participated'] if existing else 0,
        }

        # Persist in database
        try:
            with get_session() as session:
                client_row = session.get(RegisteredClient, client_id)
                if client_row is None:
                    client_row = RegisteredClient(
                        client_id=client_id,
                        registered_at=now,
                    )
                    session.add(client_row)

                # Update client data
                client_row.metadata_json = json.dumps(metadata)
                client_row.last_seen = now
                client_row.rounds_participated = self.registered_clients[client_id]['rounds_participated']

                # Commit changes
                session.commit()
        except Exception as exc:
            logger.warning(f"Could not persist client registration for {client_id}: {exc}")

        re_registered = "(re-registered)" if existing else ""
        logger.info(f"Client {client_id} {re_registered} registered")

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
        Receive a model update from a client and buffer it in Redis. Falls back 
        to RAM if Redis is unavailable.

        Args:
            client_id: Client identifier
            model_update: Serialized model parameters
            num_samples: Number of training samples
            metrics: Training metrics

        Returns:
            Response dictionary
        """
        # Check if client is registered
        if client_id not in self.registered_clients:
            return {'status': 'error', 'message': 'Client not registered'}

        now = time.time()
        self.registered_clients[client_id]['last_seen'] = now
        self.registered_clients[client_id]['rounds_participated'] += 1

        update = {
            'model_update': model_update,
            'num_samples': num_samples,
            'metrics': metrics,
            'round': self.current_round,
            'timestamp': now,
        }

        # Buffer in RAM
        self.client_models[client_id] = update
        self.client_metrics[client_id].append(metrics)

        # Buffer in Redis for cross-restart durability
        try:
            cache_client_update(self.current_round, client_id, update)
        except Exception as exc:
            logger.debug(f"Redis cache write failed: {exc}")

        logger.info(
            f"Received update from {client_id}: "
            f"samples={num_samples}, round={self.current_round}"
        )

        return {'status': 'success', 'message': 'Update received'}

    def get_client_metrics(self, client_id: str) -> Optional[Dict]:
        """
        Get the latest metrics for a client.
        Used by root cause analysis to correlate anomalies with node-level metrics.
        """
        if not self.client_metrics.get(client_id):
            return None
        return self.client_metrics[client_id][-1]

    def get_all_clients_system_metrics(self) -> Dict[str, Dict]:
        """
        Get the latest system_metrics for all clients.
        Used by root cause analysis to determine which node contributed to an anomaly.
        """
        out = {}
        for client_id in self.client_metrics:
            latest = self.get_client_metrics(client_id)
            if latest and isinstance(latest.get('system_metrics'), dict):
                out[client_id] = latest['system_metrics']
        return out

    def aggregate_models(self, timeout: Optional[float] = None) -> bool:
        """
        Aggregate client models using FedAvg.

        Waits up to 'timeout' seconds for enough client updates to arrive.
        If the timeout expires, aggregates with whatever has been received
        (provided it meets 'min_clients_per_round'). Merges any updates
        recovered from Redis that are not yet in RAM.

        Args:
            timeout: Override the configured round timeout in seconds

        Returns:
            True if aggregation was successful
        """
        fed_config = self.config.get('federated', {}).get('coordinator', {})
        round_timeout = timeout if timeout is not None else fed_config.get('round_timeout_seconds', 120)

        # Set deadline for round timeout
        deadline = time.time() + round_timeout

        # Wait for enough client updates to arrive
        while time.time() < deadline:
            # Merge any Redis cached updates that are not yet in RAM
            try:
                # Get cached updates from Redis
                cached = list_cached_updates(self.current_round)

                # Merge cached updates
                for client_id, update in cached.items():
                    if client_id not in self.client_models:
                        self.client_models[client_id] = update
            except Exception:
                pass

            # Check if we have enough updates
            if len(self.client_models) >= self.min_clients_per_round:
                break

            # Wait before checking again
            time.sleep(1.0)

        # Check if we have enough updates
        if len(self.client_models) < self.min_clients_per_round:
            logger.warning(
                f"Round {self.current_round} timed out after {round_timeout}s: "
                f"{len(self.client_models)}/{self.min_clients_per_round} updates received"
            )
            return False

        start_time = time.time()

        # Filter out stale updates
        valid_updates = {
            cid: upd for cid, upd in self.client_models.items()
            if self.current_round - upd['round'] <= self.staleness_tolerance
        }

        if len(valid_updates) < self.min_clients_per_round:
            logger.warning("Insufficient valid (non-stale) updates")
            return False

        # Poisoning detection
        if self.poisoning_detection_enabled:
            valid_updates = self._detect_poisoned_updates(valid_updates)

        if len(valid_updates) < self.min_clients_per_round:
            logger.warning("Insufficient updates after poisoning detection")
            return False

        # Weighted averaging
        total_samples = sum(u['num_samples'] for u in valid_updates.values())
        aggregated_state_dict: Dict[str, torch.Tensor] = {}

        for client_id, update in valid_updates.items():
            weight = update['num_samples'] / total_samples
            for key, param in update['model_update'].items():
                param_tensor = torch.FloatTensor(param).to(self.device)
                if key not in aggregated_state_dict:
                    aggregated_state_dict[key] = weight * param_tensor
                else:
                    aggregated_state_dict[key] += weight * param_tensor

        # Update global model
        self.global_model.load_state_dict(aggregated_state_dict)

        # Update validator if poisoning detection is enabled
        if self.poisoning_detection_enabled:
            self._update_validator(aggregated_state_dict)

        # Calculate aggregation time
        aggregation_time = time.time() - start_time

        # Update metrics
        self.metrics.record_fl_round()
        self.metrics.set_fl_clients(len(valid_updates))
        round_info = {
            'round': self.current_round,
            'num_clients': len(valid_updates),
            'total_samples': total_samples,
            'aggregation_time': aggregation_time,
            'timestamp': time.time(),
        }
        self.round_history.append(round_info)

        # Persist round history to database
        try:
            with get_session() as session:
                session.add(RoundHistory(
                    round_num=self.current_round,
                    num_clients=len(valid_updates),
                    total_samples=total_samples,
                    aggregation_time=aggregation_time,
                ))
        except Exception as exc:
            logger.warning(f"Could not persist round history: {exc}")

        # Evict consumed updates from Redis
        for client_id in valid_updates:
            try:
                delete_client_update(self.current_round, client_id)
            except Exception:
                pass

        # Clear client models
        self.client_models.clear()

        logger.info(
            f"Round {self.current_round} aggregation completed: "
            f"clients={len(valid_updates)}, time={aggregation_time:.2f}s"
        )

        return True
    
    def run_round(self) -> bool:
        """
        Execute one Federated Learning round.
        
        Returns:
            True if aggregation was successful, False otherwise
        """
        logger.info(f"Starting round {self.current_round + 1}/{self.num_rounds}")   

        # Aggregate client models
        success = self.aggregate_models()

        # Update round number and checkpoint
        if success:
            self.current_round += 1
            self._checkpoint()
    
        return success
    
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
            client_params = np.concatenate([np.array(p).flatten() for p in update['model_update'].values()])
            param_vectors.append(client_params)
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
    
    def save_global_model(self, path: str) -> None:
        """
        Save the global model to a file.
        
        Args:
            path: Path to save the model
        """
        metadata = {
            'round': self.current_round,
            'num_clients': len(self.registered_clients),
            'timestamp': time.time(),
        }
        save_model(self.global_model, path, metadata=metadata)
        logger.info(f"Global model saved to {path}")

    def load_global_model(self, path: str) -> None:
        """
        Load the global model from a file.
        
        Args:
            path: Path to load the model from
        """
        info = load_model(self.global_model, path, device=str(self.device))
        self.current_round = info.get('metadata', {}).get('round', 0)
        logger.info(f"Global model loaded from {path}, round={self.current_round}")
