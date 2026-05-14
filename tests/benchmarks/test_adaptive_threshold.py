"""
Benchmark E – Adaptive Thresholding Performance (Thesis §11.5)
==============================================================
Compares static, EWMA, and Q-Learning thresholds over reconstruction
errors from the best pre-trained LSTM-AE on SMD data.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

def _import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

_models = _import_from_file('edge_models', os.path.join(ROOT, 'src', 'edge', 'models.py'))
_logger = _import_from_file('thesis_logger', os.path.join(ROOT, 'tests', 'utils', 'thesis_logger.py'))
_loader = _import_from_file('smd_loader', os.path.join(ROOT, 'tests', 'utils', 'smd_loader.py'))
LSTMAnomalyDetector = _models.LSTMAnomalyDetector
ThesisLogger = _logger.ThesisLogger
load_smd_dataset = _loader.load_smd_dataset

from thresholding.threshold_tuner import ThresholdTuner

np.random.seed(42); torch.manual_seed(42)

INPUT_SIZE  = 38
SEQ_LEN     = 50
BEST_HIDDEN = 128
BEST_LAYERS = 2
TIME_STEPS  = 10080
EWMA_ALPHA  = 0.05

BEST_CKPT = os.path.join(ROOT, 'tests', 'saved_models', 'lstm', 'lstm_best.pt')


def _get_real_reconstruction_errors():
    """Train/load LSTM-AE and compute errors on SMD test data."""
    print("  [E] Loading SMD dataset ...")
    train_data, _, _, test_windows, test_labels = load_smd_dataset(
        seq_len=SEQ_LEN, max_train=2000, max_test=TIME_STEPS,
    )

    # Load best pre-trained model
    model = LSTMAnomalyDetector(INPUT_SIZE, BEST_HIDDEN, BEST_LAYERS)
    if os.path.exists(BEST_CKPT):
        print("  [E] Loading best pre-trained LSTM ...")
        ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("  [E] Training LSTM-AE for 50 epochs ...")
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = torch.nn.MSELoss()
        x_train = torch.FloatTensor(train_data)
        ds = torch.utils.data.TensorDataset(x_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        for _ in range(50):
            model.train()
            for (bx,) in dl:
                opt.zero_grad()
                loss = crit(model(bx), bx); loss.backward(); opt.step()

    model.eval()
    x_test = torch.FloatTensor(test_windows)
    with torch.no_grad():
        errors = model.compute_reconstruction_error(x_test, reduction='mean').numpy()
    return errors, test_labels


def run_adaptive_threshold_benchmark():
    logger = ThesisLogger("benchmark_E_adaptive_threshold")
    metrics, ground_truth = _get_real_reconstruction_errors()

    INITIAL_THRESH = float(np.percentile(metrics, 90))
    STATIC_THRESH  = float(np.percentile(metrics, 95))
    LOG_EVERY = 60

    # ── Static ─────────────────────────────────────────────────────────
    static_tp = static_fp = static_fn = static_tn = 0
    static_alerts = 0

    # ── EWMA ───────────────────────────────────────────────────────────
    ewma_threshold = INITIAL_THRESH
    ewma_tp = ewma_fp = ewma_fn = ewma_tn = 0
    ewma_alerts = 0

    # ── Q-Learning ────────────────────────────────────────────────────
    tuner = ThresholdTuner()
    tuner.initialize_service("bench-svc", initial_threshold=INITIAL_THRESH)
    ql_alerts = 0
    cumulative_reward = 0.0

    for t in range(len(metrics)):
        val = metrics[t]
        is_true_anomaly = bool(ground_truth[t])

        # Static
        sd = val > STATIC_THRESH
        if sd: static_alerts += 1
        if sd and is_true_anomaly:     static_tp += 1
        elif sd and not is_true_anomaly: static_fp += 1
        elif not sd and is_true_anomaly: static_fn += 1
        else:                           static_tn += 1

        # EWMA
        ed = val > ewma_threshold
        if ed: ewma_alerts += 1
        if ed and is_true_anomaly:     ewma_tp += 1
        elif ed and not is_true_anomaly: ewma_fp += 1
        elif not ed and is_true_anomaly: ewma_fn += 1
        else:                           ewma_tn += 1
        ewma_threshold = EWMA_ALPHA * val + (1 - EWMA_ALPHA) * ewma_threshold

        # Q-Learning
        ql_thresh = tuner.service_thresholds.get("bench-svc", INITIAL_THRESH)
        qd = val > ql_thresh
        if qd: ql_alerts += 1

        tuner.update_feedback(
            service="bench-svc",
            metric_value=float(val),
            was_anomaly_detected=qd,
            was_true_anomaly=is_true_anomaly,
            slo_violated=is_true_anomaly and not qd,
        )
        if t % 10 == 0:
            tuner.tune_threshold("bench-svc")

        reward = tuner.service_history["bench-svc"][-1]['reward'] if tuner.service_history["bench-svc"] else 0
        cumulative_reward += reward

        if t % LOG_EVERY == 0:
            hour = t // LOG_EVERY
            ql_perf = tuner.get_service_performance("bench-svc")
            static_fpr = static_fp / max(1, static_fp + static_tn)
            ewma_fpr   = ewma_fp / max(1, ewma_fp + ewma_tn)

            logger.log_metric(f"hour_{hour}", {
                'minute': t, 'metric_value': round(float(val), 4),
                'static_threshold': STATIC_THRESH,
                'static_alerts': static_alerts,
                'static_fpr': round(static_fpr, 4),
                'ewma_threshold': round(ewma_threshold, 4),
                'ewma_alerts': ewma_alerts,
                'ewma_fpr': round(ewma_fpr, 4),
                'ql_threshold': round(ql_perf['threshold'], 4),
                'ql_alerts': ql_perf.get('true_positives', 0) + ql_perf.get('false_positives', 0),
                'ql_fpr': round(ql_perf['false_positive_rate'], 4),
                'ql_f1': round(ql_perf['f1_score'], 4),
                'ql_cumulative_reward': round(cumulative_reward, 4),
                'ql_epsilon': round(tuner.epsilon, 6),
            })

    print("  [E] Complete.")


if __name__ == "__main__":
    run_adaptive_threshold_benchmark()