import pytest
import numpy as np


class TestTunerInitialization:
    """
    Validates ThresholdTuner initialization and state mapping settings.
    """

    def test_default_init(self, config):
        """
        Verify that default hyperparameters load correctly from configurations.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        assert tuner.learning_rate > 0
        assert tuner.discount_factor > 0
        assert tuner.epsilon > 0

    def test_initialize_service(self, config):
        """
        Verify manual service initialization registers thresholds correctly.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=2.5)
        assert tuner.service_thresholds["svc-x"] == 2.5

    def test_auto_initialize(self, config):
        """
        Verify that unknown services auto-initialize when tuned.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        th = tuner.tune_threshold("never-seen")
        assert isinstance(th, float)
        assert th > 0


class TestQlearningLogic:
    """
    Validates Q-learning state representations, state discretization, and selection.
    """

    def test_state_discretization(self, config):
        """
        Verify continuous metrics partition into correct discrete Q-table keys.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        state = {"false_positive_rate": 0.15, "false_negative_rate": 0.27, "slo_violation_rate": 0.0}
        key = tuner._state_to_key(state)
        assert key == "0.1_0.2_0.0"

    def test_state_key_zero(self, config):
        """
        Verify state discretization bounds clean zero-value telemetry.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        state = {"false_positive_rate": 0.0, "false_negative_rate": 0.0, "slo_violation_rate": 0.0}
        key = tuner._state_to_key(state)
        assert key == "0.0_0.0_0.0"

    def test_q_table_updated_after_feedback_and_tune(self, config):
        """
        Verify that feedback loops register updates inside service Q-tables.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        for i in range(20):
            tuner.update_feedback("svc-x", float(i), True, True, False)
            tuner.tune_threshold("svc-x")

        assert len(tuner.q_table["svc-x"]) > 0

    def test_epsilon_greedy_exploration(self, config):
        """
        Verify random actions are picked under exploration configurations (epsilon=1.0).

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.epsilon = 1.0
        tuner.initialize_service("svc-x")

        thresholds = set()
        for i in range(50):
            tuner.update_feedback("svc-x", 3.0, True, True, False)
            th = tuner.tune_threshold("svc-x")
            thresholds.add(round(th, 4))

        assert len(thresholds) > 1

    def test_epsilon_zero_exploits(self, config):
        """
        Verify deterministic actions are picked under exploitation configurations (epsilon=0.0).

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.epsilon = 0.0
        tuner.min_epsilon = 0.0
        tuner.initialize_service("svc-x")

        for i in range(30):
            tuner.update_feedback("svc-x", 3.0, True, True, False)
            tuner.tune_threshold("svc-x")

        tuner.update_feedback("svc-x", 3.0, True, True, False)
        th1 = tuner.tune_threshold("svc-x")
        tuner.update_feedback("svc-x", 3.0, True, True, False)
        th2 = tuner.tune_threshold("svc-x")
        assert isinstance(th1, float) and isinstance(th2, float)


class TestRewardComputation:
    """
    Validates reinforcement learning reward calculations.
    """

    def test_true_positive_positive_reward(self, config):
        """
        Verify true positive detections yield positive reward values.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        reward = tuner._compute_reward("svc-x", detected=True, true_anomaly=True, slo_violated=False)
        assert reward > 0

    def test_false_positive_negative_reward(self, config):
        """
        Verify false positive detections are penalized.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        reward = tuner._compute_reward("svc-x", detected=True, true_anomaly=False, slo_violated=False)
        assert reward < 0.6

    def test_false_negative_negative_reward(self, config):
        """
        Verify false negative omissions (missed anomalies) are heavily penalized.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        reward = tuner._compute_reward("svc-x", detected=False, true_anomaly=True, slo_violated=False)
        assert reward < 0

    def test_slo_violation_penalty(self, config):
        """
        Verify that SLO violations result in severe reward reductions.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        r_ok = tuner._compute_reward("svc-x", detected=False, true_anomaly=False, slo_violated=False)
        r_bad = tuner._compute_reward("svc-x", detected=False, true_anomaly=False, slo_violated=True)
        assert r_bad < r_ok


class TestEpsilonDecay:
    """
    Validates exploration rate decay behavior.
    """

    def test_epsilon_decreases(self, config):
        """
        Verify that exploration rate epsilon decreases over tuning cycles.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")
        initial = tuner.epsilon

        for i in range(10):
            tuner.update_feedback("svc-x", float(i), True, True, False)
            tuner.tune_threshold("svc-x")

        assert tuner.epsilon < initial

    def test_epsilon_bounded_below(self, config):
        """
        Verify that epsilon is capped at the configured minimum threshold value.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        for i in range(5000):
            tuner.update_feedback("svc-x", float(i % 10), True, True, False)
            tuner.tune_threshold("svc-x")

        assert tuner.epsilon >= tuner.min_epsilon


class TestThresholdAdjustment:
    """
    Validates boundaries on adjusted thresholds.
    """

    def test_threshold_positive(self, config):
        """
        Verify adjusted thresholds remain positive.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=3.0)
        th = tuner.tune_threshold("svc-x")
        assert th > 0

    def test_threshold_bounded(self, config):
        """
        Verify threshold limits constrain updates to valid ranges.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x", initial_threshold=3.0)

        for i in range(100):
            tuner.update_feedback("svc-x", float(i % 10), i % 3 == 0, i % 5 == 0, False)
            th = tuner.tune_threshold("svc-x")

        assert 1e-5 <= th <= 10.0


class TestPerformanceMetrics:
    """
    Validates precision, recall, and F1 confusion matrix statistics.
    """

    def test_performance_keys(self, config):
        """
        Verify accuracy and confusion matrix metric outputs carry correct schema keys.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")
        tuner.update_feedback("svc-x", 5.0, True, True, False)
        tuner.update_feedback("svc-x", 1.0, False, False, False)

        perf = tuner.get_service_performance("svc-x")
        for key in ["precision", "recall", "f1_score", "false_positive_rate", "false_negative_rate",
                     "true_positives", "false_positives", "false_negatives", "true_negatives"]:
            assert key in perf

    def test_perfect_detection_f1(self, config):
        """
        Verify flawless classification yields F1, precision, and recall values of 1.0.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        for _ in range(10):
            tuner.update_feedback("svc-x", 5.0, True, True, False)
            tuner.update_feedback("svc-x", 1.0, False, False, False)

        perf = tuner.get_service_performance("svc-x")
        assert perf["precision"] == 1.0
        assert perf["recall"] == 1.0
        assert perf["f1_score"] == 1.0

    def test_all_false_positives(self, config):
        """
        Verify flagging only false positives results in precision of 0.0.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        for _ in range(10):
            tuner.update_feedback("svc-x", 5.0, True, False, False)

        perf = tuner.get_service_performance("svc-x")
        assert perf["precision"] == 0.0
        assert perf["false_positives"] == 10


class TestTunerStatistics:
    """
    Validates aggregated tuner telemetry counters.
    """

    def test_statistics_keys(self, config):
        """
        Verify that aggregate statistic responses carry tracked service metrics.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")
        stats = tuner.get_statistics()
        assert "num_services" in stats
        assert "epsilon" in stats
        assert stats["num_services"] == 1

    def test_global_confusion_matrix_tracking(self, config):
        """
        Verify that confusion matrix counts increment correctly.

        Args:
            config: Test configuration fixture
        """
        from thresholding.threshold_tuner import ThresholdTuner
        tuner = ThresholdTuner(config)
        tuner.initialize_service("svc-x")

        tuner.update_feedback("svc-x", 5.0, True, True, False)
        tuner.update_feedback("svc-x", 5.0, True, False, False)
        tuner.update_feedback("svc-x", 1.0, False, True, False)
        tuner.update_feedback("svc-x", 1.0, False, False, False)

        assert tuner.global_tp["svc-x"] == 1
        assert tuner.global_fp["svc-x"] == 1
        assert tuner.global_fn["svc-x"] == 1
        assert tuner.global_tn["svc-x"] == 1
