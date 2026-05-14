"""
Thesis Experiment Logger
========================
Provides structured JSON logging for all benchmark scripts to ensure reproducibility
and easy parsing of empirical results for the thesis manuscript.
"""
import logging
import json
import os
from datetime import datetime

class ThesisLogger:
    """
    Handles file-based JSON logging for thesis empirical evaluations.
    
    Creates timestamped log files in the tests/logs directory and ensures all
    recorded metrics are serialized as JSON strings for downstream processing.
    """
    def __init__(self, experiment_name: str):
        """
        Initializes the logger.
        
        Args:
            experiment_name: Name of the current benchmark or experiment.
        """
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Simple format since we will log raw JSON strings for easy parsing
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_metric(self, step: int, metrics_dict: dict) -> None:
        """
        Logs a dictionary of metrics as a JSON string.
        
        Args:
            step: The current training epoch, evaluation round, or sequence step.
            metrics_dict: Dictionary containing the key-value metric pairs.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics_dict
        }
        self.logger.info(json.dumps(record))