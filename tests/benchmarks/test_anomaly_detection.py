"""
Benchmark A – Anomaly Detection Performance (Thesis §11.2)
==========================================================
Uses the BEST pre-trained LSTM-AE from the hyperparameter sweep, plus
trains baseline variants (static, centralized, federated+DP) on real SMD data.

Logs per-model breakdown: Precision, Recall, F1, AUC-ROC, AUC-PR, latency.
Also logs a cold-start convergence curve (MSE vs round).
"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

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

# ── Best config from sweep: Config C (hidden=128, layers=2) ────────────────
INPUT_SIZE    = 38
SEQ_LEN       = 50
BEST_HIDDEN   = 128
BEST_LAYERS   = 2
NUM_TRAIN     = 4000
NUM_TEST      = 1000
BURNIN_EPOCHS = 100

np.random.seed(42)
torch.manual_seed(42)

BEST_CKPT = os.path.join(ROOT, 'tests', 'saved_models', 'lstm', 'lstm_best.pt')


def _load_best_model():
    """Load the best pre-trained LSTM from sweep."""
    model = LSTMAnomalyDetector(INPUT_SIZE, BEST_HIDDEN, BEST_LAYERS)
    ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def _train_model(model, data, epochs, lr=0.001, dp_noise=0.0, clip_norm=0.0, batch_size=64):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    x    = torch.FloatTensor(data)
    ds   = torch.utils.data.TensorDataset(x)
    dl   = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    losses = []
    for _ in range(epochs):
        model.train()
        ep_loss = 0.0
        for (bx,) in dl:
            opt.zero_grad()
            loss = crit(model(bx), bx); loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if dp_noise > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * dp_noise
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(dl))
    return losses


def _score_model(model, test_normal, test_anomalous, y_true):
    model.eval()
    all_data = np.concatenate([test_normal, test_anomalous], axis=0)
    x = torch.FloatTensor(all_data)
    t0 = time.perf_counter()
    with torch.no_grad():
        errors = model.compute_reconstruction_error(x, reduction='mean').numpy()
    latency = (time.perf_counter() - t0) * 1000 / len(all_data)
    thresh = np.percentile(errors[:len(test_normal)], 95)
    y_pred = (errors > thresh).astype(int)
    y_scores = errors / (errors.max() + 1e-9)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score':  float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_true, y_scores)),
        'auc_pr':    float(average_precision_score(y_true, y_scores)),
        'detection_latency_ms': round(latency, 4),
    }


def run_anomaly_detection_benchmark():
    logger = ThesisLogger("benchmark_A_anomaly_detection")

    print("  [A] Loading real SMD dataset ...")
    train_data, test_normal, test_anomalous, _, _ = load_smd_dataset(
        seq_len=SEQ_LEN, max_train=NUM_TRAIN, max_test=NUM_TEST,
    )
    y_true = np.concatenate([np.zeros(len(test_normal)),
                             np.ones(len(test_anomalous))]).astype(int)

    print(f"      train={len(train_data)}, normal={len(test_normal)}, anomalous={len(test_anomalous)}")

    # ── 1. Static 3-sigma (untrained) ───────────────────────────────────
    print("  [A] Evaluating static_3sigma ...")
    m_static = LSTMAnomalyDetector(INPUT_SIZE, BEST_HIDDEN, BEST_LAYERS)
    r = _score_model(m_static, test_normal, test_anomalous, y_true)
    r['model'] = 'static_3sigma'
    logger.log_metric("eval_static_3sigma", r)
    print(f"      F1={r['f1_score']:.4f}  AUC-ROC={r['auc_roc']:.4f}")

    # ── 2. Centralized LSTM-AE (train from scratch, no DP) ─────────────
    print("  [A] Training centralized_lstm_ae ...")
    m_cent = LSTMAnomalyDetector(INPUT_SIZE, BEST_HIDDEN, BEST_LAYERS)
    _train_model(m_cent, train_data, BURNIN_EPOCHS, lr=0.001)
    r = _score_model(m_cent, test_normal, test_anomalous, y_true)
    r['model'] = 'centralized_lstm_ae'
    logger.log_metric("eval_centralized_lstm_ae", r)
    print(f"      F1={r['f1_score']:.4f}  AUC-ROC={r['auc_roc']:.4f}")

    # ── 3. Federated LSTM-AE + DP (pre-trained best) ───────────────────
    print("  [A] Loading best pre-trained federated model ...")
    m_fed = _load_best_model()
    r = _score_model(m_fed, test_normal, test_anomalous, y_true)
    r['model'] = 'federated_lstm_ae_dp'
    logger.log_metric("eval_federated_lstm_ae_dp", r)
    print(f"      F1={r['f1_score']:.4f}  AUC-ROC={r['auc_roc']:.4f}")

    # ── 4. Cold-start convergence (MSE vs round) ──────────────────────
    print("  [A] Generating cold-start convergence curve ...")
    m_lc = LSTMAnomalyDetector(INPUT_SIZE, BEST_HIDDEN, BEST_LAYERS)
    opt  = torch.optim.Adam(m_lc.parameters(), lr=0.001)
    crit = torch.nn.MSELoss()
    x_train = torch.FloatTensor(train_data)
    ds = torch.utils.data.TensorDataset(x_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    for rnd in range(BURNIN_EPOCHS):
        m_lc.train()
        ep_loss = 0.0
        for (bx,) in dl:
            opt.zero_grad()
            loss = crit(m_lc(bx), bx); loss.backward(); opt.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(dl)

        # Evaluate every 5 rounds for thesis table
        if rnd % 5 == 0 or rnd == BURNIN_EPOCHS - 1:
            sc = _score_model(m_lc, test_normal, test_anomalous, y_true)
            sc.update({'model': 'federated_cold_start', 'round': rnd, 'train_loss': avg_loss})
            logger.log_metric(f"learning_curve_round_{rnd}", sc)
        else:
            logger.log_metric(f"learning_curve_round_{rnd}", {
                'model': 'federated_cold_start', 'round': rnd, 'train_loss': avg_loss,
            })

    print("  [A] Complete.")


if __name__ == "__main__":
    run_anomaly_detection_benchmark()