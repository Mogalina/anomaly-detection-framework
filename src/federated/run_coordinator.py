import os
import sys
import signal
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_config
from utils.logger import get_logger
from utils.db import init_db
from federated.federated_coordinator import FederatedCoordinator
from coordinator.main import app as flask_app
from utils.metrics import get_metrics


logger = get_logger(__name__)


def main() -> None:
    config = get_config()

    host = os.environ.get('COORDINATOR_HOST') or config.get('federated.coordinator.host', '0.0.0.0')
    port = int(os.environ.get('COORDINATOR_PORT') or config.get('federated.coordinator.port', 50051))
    num_rounds = config.get('federated.coordinator.num_rounds', 100)

    logger.info(f"Starting Federated Learning Coordinator on {host}:{port}")

    # Initialize database
    try:
        init_db()
        logger.info("Database schema ready")
    except Exception as exc:
        logger.warning(f"Database initialisation failed (continuing without persistence): {exc}")

    coordinator = FederatedCoordinator()

    # Run the gRPC server in a daemon thread so the main thread can manage the 
    # Federated Learning round loop and handle signals cleanly
    server_thread = threading.Thread(
        target=coordinator.serve,
        kwargs={'host': host, 'port': port},
        daemon=True,
        name='grpc-server',
    )
    server_thread.start()

    # Start Flask API server
    api_port = int(config.get('coordinator.api.port', 8080))
    api_thread = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=api_port, debug=False, use_reloader=False),
        daemon=True,
        name='api-server',
    )
    api_thread.start()

    # Start metrics server
    try:
        get_metrics().start()
        logger.info("Metrics server started")
    except Exception as exc:
        logger.warning(f"Could not start metrics server: {exc}")

    # Graceful shutdown
    stop_event = threading.Event()

    def _shutdown(sig, frame):
        logger.info(f"Signal {sig} received: Coordinator shutting down")
        coordinator.stop(grace=5.0)
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"Running {num_rounds} Federated Learning rounds")

    for round_idx in range(num_rounds):
        if stop_event.is_set():
            break

        logger.info(f"Completing Federated Learning round {round_idx + 1}/{num_rounds}")
        success = coordinator.run_round()

        if not success:
            logger.warning(f"Federated Learning round {round_idx + 1} did not complete successfully")

        if stop_event.is_set():
            break

    logger.info("All Federated Learning rounds completed: Coordinator shutting down")
    coordinator.stop(grace=5.0)


if __name__ == '__main__':
    main()
