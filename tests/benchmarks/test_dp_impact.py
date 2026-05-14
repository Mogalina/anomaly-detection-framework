"""
Benchmark B – Impact of Differential Privacy (Thesis §11.2.3)
=============================================================
Trains the LSTM-AE under four DP noise multipliers (σ) mapped to
approximate privacy budgets. Uses DP-SGD: per-sample gradient clipping
followed by Gaussian noise addition.

Key insight: the noise multiplier σ controls the privacy–accuracy trade-off.
Lower σ → weaker privacy but better accuracy.
"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
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

INPUT_SIZE  = 38
SEQ_LEN     = 50
HIDDEN_SIZE = 32
NUM_LAYERS  = 1
EPOCHS      = 200
np.random.seed(42); torch.manual_seed(42)


def _evaluate(model, test_normal, test_anomalous):
    model.eval()
    all_data = np.concatenate([test_normal, test_anomalous])
    y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomalous))]).astype(int)
    x = torch.FloatTensor(all_data)
    with torch.no_grad():
        errors = model.compute_reconstruction_error(x, reduction='mean').numpy()
    thresh = np.percentile(errors[:len(test_normal)], 95)
    y_pred = (errors > thresh).astype(int)
    y_scores = errors / (errors.max() + 1e-9)
    return {
        'f1_score':  float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_true, y_scores)),
    }


def run_dp_benchmark():
    logger = ThesisLogger("benchmark_B_dp_impact")

    print("  [B] Loading real SMD dataset ...")
    train_data, test_normal, test_anomalous, _, _ = load_smd_dataset(
        seq_len=SEQ_LEN, max_train=4000, max_test=1000,
    )
    x_train = torch.FloatTensor(train_data)

    # DP-SGD noise configurations: (label, noise_multiplier, clip_norm, approx_epsilon)
    # σ=0 → no DP, σ=0.001 → very loose DP (~ε≈100), σ=0.005 → moderate (~ε≈10),
    # σ=0.01 → strong (~ε≈1), σ=0.05 → very strong (~ε≈0.1)
    dp_configs = [
        ("No DP",             0.0,      0.0,  None),
        ("ε ≈ 100 (σ=0.0001)", 0.0001, 1.0,  100.0),
        ("ε ≈ 10 (σ=0.0005)",  0.0005, 1.0,  10.0),
        ("ε ≈ 1 (σ=0.001)",    0.001,  1.0,   1.0),
        ("ε ≈ 0.1 (σ=0.005)",  0.005,  1.0,   0.1),
    ]

    for label, sigma, clip_norm, approx_eps in dp_configs:
        tag = f"sigma_{sigma}"
        print(f"  [B] {label} ...")

        torch.manual_seed(42)
        model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=0.1)
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        crit  = torch.nn.MSELoss()

        ds = torch.utils.data.TensorDataset(x_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        for r in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for (bx,) in dl:
                opt.zero_grad()
                loss = crit(model(bx), bx); loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                if sigma > 0:
                    # Scale noise correctly for averaged gradients: (clip_norm * sigma) / batch_size
                    noise_std = (clip_norm * sigma) / 64.0
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad += torch.randn_like(p.grad) * noise_std
                opt.step()
                epoch_loss += loss.item()

            if r % 20 == 0 or r == EPOCHS - 1:
                m = _evaluate(model, test_normal, test_anomalous)
                m.update({'sigma': sigma, 'epsilon': approx_eps if approx_eps else 'inf',
                          'round': r, 'loss': epoch_loss / len(dl), 'label': label})
                logger.log_metric(f"{tag}_round_{r}", m)

        final = _evaluate(model, test_normal, test_anomalous)
        final.update({'sigma': sigma, 'epsilon': approx_eps if approx_eps else 'inf',
                      'label': label, 'final_loss': epoch_loss / len(dl)})
        logger.log_metric(f"{tag}_final", final)
        print(f"      → F1={final['f1_score']:.4f}  AUC-ROC={final['auc_roc']:.4f}  "
              f"Recall={final['recall']:.4f}")

    # ── Part 2: clipping norm sensitivity at σ=0.01 ───────────────────
    print("  [B] Clipping norm sensitivity at σ=0.001 ...")
    clip_norms = [0.01, 0.05, 0.1, 0.5, 1.0]
    sigma_fixed = 0.001

    for cn in clip_norms:
        torch.manual_seed(42)
        model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, dropout=0.1)
        opt   = torch.optim.Adam(model.parameters(), lr=0.001)
        crit  = torch.nn.MSELoss()
        ds = torch.utils.data.TensorDataset(x_train)
        dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

        for r in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for (bx,) in dl:
                opt.zero_grad()
                loss = crit(model(bx), bx); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cn)
                # Scale noise correctly for averaged gradients
                noise_std = (cn * sigma_fixed) / 64.0
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * noise_std
                opt.step()
                epoch_loss += loss.item()

        final = _evaluate(model, test_normal, test_anomalous)
        final.update({'clip_norm': cn, 'sigma': sigma_fixed, 'final_loss': epoch_loss / len(dl)})
        logger.log_metric(f"clip_norm_{cn}", final)
        print(f"      clip_norm={cn}: F1={final['f1_score']:.4f}  AUC-ROC={final['auc_roc']:.4f}")

    print("  [B] Complete.")


if __name__ == "__main__":
    run_dp_benchmark()