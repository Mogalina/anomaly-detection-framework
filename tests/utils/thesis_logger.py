import logging
import json
import os
from datetime import datetime

class ThesisLogger:
    def __init__(self, experiment_name):
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

    def log_metric(self, step, metrics_dict):
        """Logs a dictionary of metrics as a JSON string."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics_dict
        }
        self.logger.info(json.dumps(record))