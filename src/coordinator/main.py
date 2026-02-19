"""
Coordinator service main entry point.
"""
import argparse
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS

from coordinator.anomaly_pipeline import AnomalyPipeline
from federated.federated_coordinator import FederatedCoordinator
from utils.logger import setup_logger, get_logger
from utils.config import init_config
from utils.metrics import get_metrics


app = Flask(__name__)
CORS(app)

# Global instances
pipeline = None
fed_coordinator = None
logger = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/ready', methods=['GET'])
def ready():
    """Readiness check endpoint."""
    if pipeline and pipeline.is_running:
        return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'not ready'}), 503


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    status = pipeline.get_pipeline_status()
    return jsonify(status), 200


@app.route('/api/anomaly', methods=['POST'])
def submit_anomaly():
    """Submit anomaly event."""
    data = request.json
    service = data.get('service')
    anomaly_data = data.get('anomaly_data', {})
    
    result = pipeline.process_anomaly_event(service, anomaly_data)
    
    return jsonify(result), 200


@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get causal dependency graph."""
    graph_data = pipeline.causal_graph.export_graph()
    return jsonify(graph_data), 200


@app.route('/api/root-causes', methods=['GET'])
def get_root_causes():
    """Get recent root cause analysis."""
    if pipeline.root_cause_analyzer.analysis_history:
        latest = pipeline.root_cause_analyzer.analysis_history[-1]
        return jsonify(latest), 200
    return jsonify({'message': 'No analysis available'}), 404


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit detection feedback for threshold tuning."""
    data = request.json
    service = data.get('service')
    was_detected = data.get('was_detected')
    was_true_anomaly = data.get('was_true_anomaly')
    
    pipeline.provide_feedback(service, was_detected, was_true_anomaly)
    
    return jsonify({'status': 'feedback received'}), 200


def main():
    """Main entry point for coordinator service."""
    global pipeline, fed_coordinator, logger
    
    parser = argparse.ArgumentParser(description='Anomaly Detection Coordinator')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--port', type=int, default=8080, help='API port')
    args = parser.parse_args()
    
    # Initialize
    config = init_config(args.config)
    setup_logger(
        level=config.get('monitoring.logging.level', 'INFO'),
        log_file='logs/coordinator.log'
    )
    logger = get_logger(__name__)
    
    logger.info("Starting coordinator service...")
    
    # Initialize components
    pipeline = AnomalyPipeline(config.to_dict())
    fed_coordinator = FederatedCoordinator(config.to_dict())
    
    pipeline.is_running = True
    
    # Start metrics server
    metrics = get_metrics()
    metrics.start()
    logger.info("Metrics server started on port 9090")
    
    # Start API server
    logger.info(f"Starting API server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
