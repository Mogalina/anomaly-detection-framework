import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import random

from utils.logger import get_logger
from utils.config import get_config
from utils.metrics import get_metrics


logger = get_logger(__name__)


class ThresholdTuner:
    """
    RL-based adaptive threshold tuner using Q-learning.
    
    Learns optimal detection thresholds by balancing precision,
    recall, and SLO compliance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize threshold tuner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().to_dict()
        self.metrics = get_metrics()
        
        tuner_config = self.config.get('thresholding', {}).get('rl_tuner', {})
        
        # Q-learning parameters
        self.learning_rate = tuner_config.get('learning_rate', 0.1)
        self.discount_factor = tuner_config.get('discount_factor', 0.95)
        self.epsilon = tuner_config.get('epsilon', 0.1)
        self.epsilon_decay = tuner_config.get('epsilon_decay', 0.995)
        self.min_epsilon = tuner_config.get('min_epsilon', 0.01)
        
        # State and action spaces
        self.state_features = tuner_config.get('state_features', [
            'current_threshold',
            'false_positive_rate',
            'false_negative_rate',
            'slo_violation_rate'
        ])
        self.actions = tuner_config.get('actions', [-0.5, -0.2, 0.0, 0.2, 0.5])
        
        # Reward weights
        reward_config = tuner_config.get('reward', {})
        self.precision_weight = reward_config.get('precision_weight', 0.3)
        self.recall_weight = reward_config.get('recall_weight', 0.3)
        self.slo_compliance_weight = reward_config.get('slo_compliance_weight', 0.4)
        self.false_positive_penalty = reward_config.get('false_positive_penalty', -1.0)
        self.false_negative_penalty = reward_config.get('false_negative_penalty', -2.0)
        self.slo_violation_penalty = reward_config.get('slo_violation_penalty', -3.0)
        
        # Q-table (service -> state -> action -> Q-value)
        self.q_table: Dict[str, Dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Service states
        self.service_thresholds: Dict[str, float] = {}
        self.service_states: Dict[str, Dict] = {}
        self.service_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.false_positives: Dict[str, int] = defaultdict(int)
        self.false_negatives: Dict[str, int] = defaultdict(int)
        self.true_positives: Dict[str, int] = defaultdict(int)
        self.true_negatives: Dict[str, int] = defaultdict(int)
        
        logger.info(
            f"ThresholdTuner initialized: "
            f"lr={self.learning_rate}, gamma={self.discount_factor}, "
            f"epsilon={self.epsilon}"
        )
    
    def initialize_service(self, service: str, initial_threshold: float = 3.0) -> None:
        """
        Initialize threshold for a service.
        
        Args:
            service: Service name
            initial_threshold: Initial threshold value
        """
        self.service_thresholds[service] = initial_threshold
        self.service_states[service] = self._compute_state(service)
        
        logger.info(f"Initialized threshold for {service}: {initial_threshold}")
    
    def update_feedback(
        self,
        service: str,
        was_anomaly_detected: bool,
        was_true_anomaly: bool,
        slo_violated: bool
    ) -> None:
        """
        Update with detection feedback.
        
        Args:
            service: Service name
            was_anomaly_detected: Whether anomaly was detected
            was_true_anomaly: Whether it was actually an anomaly
            slo_violated: Whether SLO was violated
        """
        # Update confusion matrix
        if was_anomaly_detected and was_true_anomaly:
            self.true_positives[service] += 1
        elif was_anomaly_detected and not was_true_anomaly:
            self.false_positives[service] += 1
        elif not was_anomaly_detected and was_true_anomaly:
            self.false_negatives[service] += 1
        else:
            self.true_negatives[service] += 1
        
        # Compute reward
        reward = self._compute_reward(
            service,
            was_anomaly_detected,
            was_true_anomaly,
            slo_violated
        )
        
        # Store in history
        self.service_history[service].append({
            'timestamp': __import__('time').time(),
            'detected': was_anomaly_detected,
            'true_anomaly': was_true_anomaly,
            'slo_violated': slo_violated,
            'reward': reward,
            'threshold': self.service_thresholds.get(service, 0)
        })
    
    def tune_threshold(self, service: str) -> float:
        """
        Tune threshold for a service using Q-learning.
        
        Args:
            service: Service name
        
        Returns:
            New threshold value
        """
        if service not in self.service_thresholds:
            self.initialize_service(service)
        
        # Get current state
        current_state = self._compute_state(service)
        state_key = self._state_to_key(current_state)
        
        # Choose action (epsilon-greedy)
        if random.random() < self.epsilon:
            # Explore
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            # Exploit
            q_values = [
                self.q_table[service][state_key][i]
                for i in range(len(self.actions))
            ]
            action_idx = int(np.argmax(q_values))
        
        action = self.actions[action_idx]
        
        # Apply action
        old_threshold = self.service_thresholds[service]
        new_threshold = max(0.1, old_threshold + action)
        self.service_thresholds[service] = new_threshold
        
        # Get reward from recent history
        recent_reward = self._get_recent_reward(service)
        
        # Update Q-value
        next_state = self._compute_state(service)
        next_state_key = self._state_to_key(next_state)
        
        max_next_q = max(
            [self.q_table[service][next_state_key][i] for i in range(len(self.actions))],
            default=0
        )
        
        current_q = self.q_table[service][state_key][action_idx]
        new_q = current_q + self.learning_rate * (
            recent_reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[service][state_key][action_idx] = new_q
        
        # Update state
        self.service_states[service] = next_state
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Record metrics
        direction = 'increase' if action > 0 else 'decrease' if action < 0 else 'unchanged'
        self.metrics.record_threshold_adjustment(service, direction)
        self.metrics.set_threshold(service, new_threshold)
        
        logger.info(
            f"Tuned threshold for {service}: {old_threshold:.3f} -> {new_threshold:.3f} "
            f"(action={action}, reward={recent_reward:.3f})"
        )
        
        return new_threshold
    
    def _compute_state(self, service: str) -> Dict:
        """
        Compute current state for a service.
        
        Args:
            service: Service name
        
        Returns:
            State dictionary
        """
        # Compute metrics
        tp = self.true_positives[service]
        fp = self.false_positives[service]
        fn = self.false_negatives[service]
        tn = self.true_negatives[service]
        
        total = tp + fp + fn + tn
        
        if total > 0:
            fpr = fp / max(1, fp + tn)  # False positive rate
            fnr = fn / max(1, fn + tp)  # False negative rate
        else:
            fpr = 0
            fnr = 0
        
        # SLO violation rate from history
        if self.service_history[service]:
            recent = list(self.service_history[service])[-20:]
            slo_violation_rate = sum(1 for h in recent if h['slo_violated']) / len(recent)
        else:
            slo_violation_rate = 0
        
        return {
            'current_threshold': self.service_thresholds.get(service, 3.0),
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'slo_violation_rate': slo_violation_rate
        }
    
    def _state_to_key(self, state: Dict) -> str:
        """
        Convert state to hashable key.
        
        Args:
            state: State dictionary
        
        Returns:
            State key string
        """
        # Discretize continuous values
        threshold_bucket = int(state['current_threshold'] * 10) / 10
        fpr_bucket = int(state['false_positive_rate'] * 20) / 20
        fnr_bucket = int(state['false_negative_rate'] * 20) / 20
        slo_bucket = int(state['slo_violation_rate'] * 20) / 20
        
        return f"{threshold_bucket:.1f}_{fpr_bucket:.2f}_{fnr_bucket:.2f}_{slo_bucket:.2f}"
    
    def _compute_reward(
        self,
        service: str,
        detected: bool,
        true_anomaly: bool,
        slo_violated: bool
    ) -> float:
        """
        Compute reward for current action.
        
        Args:
            service: Service name
            detected: Whether anomaly was detected
            true_anomaly: Whether it was truly an anomaly
            slo_violated: Whether SLO was violated
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Reward for correct detection
        if detected and true_anomaly:
            reward += self.precision_weight + self.recall_weight
        
        # Penalty for false positive
        if detected and not true_anomaly:
            reward += self.false_positive_penalty
        
        # Penalty for false negative
        if not detected and true_anomaly:
            reward += self.false_negative_penalty
        
        # Reward/penalty for SLO compliance
        if slo_violated:
            reward += self.slo_violation_penalty
        else:
            reward += self.slo_compliance_weight
        
        return reward
    
    def _get_recent_reward(self, service: str) -> float:
        """
        Get average recent reward.
        
        Args:
            service: Service name
        
        Returns:
            Average reward
        """
        if not self.service_history[service]:
            return 0.0
        
        recent = list(self.service_history[service])[-10:]
        avg_reward = sum(h['reward'] for h in recent) / len(recent)
        
        return avg_reward
    
    def get_service_performance(self, service: str) -> Dict:
        """
        Get performance metrics for a service.
        
        Args:
            service: Service name
        
        Returns:
            Performance metrics
        """
        tp = self.true_positives[service]
        fp = self.false_positives[service]
        fn = self.false_negatives[service]
        tn = self.true_negatives[service]
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_score = 2 * precision * recall / max(0.001, precision + recall)
        
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        
        # Update metrics
        self.metrics.set_false_positive_rate(service, fpr)
        self.metrics.set_false_negative_rate(service, fnr)
        
        return {
            'service': service,
            'threshold': self.service_thresholds.get(service, 0),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    def get_statistics(self) -> Dict:
        """
        Get tuner statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'num_services': len(self.service_thresholds),
            'epsilon': self.epsilon,
            'total_updates': sum(len(h) for h in self.service_history.values()),
            'avg_threshold': (
                sum(self.service_thresholds.values()) / len(self.service_thresholds)
                if self.service_thresholds else 0
            )
        }
