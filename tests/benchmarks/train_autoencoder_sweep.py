"""
AutoEncoder Hyperparameter Sweep — Train on FL Weight Vectors
=============================================================
Trains the AutoEncoder (poisoned-model-update detector) using actual
neural network weight vectors from the LSTM models trained in Phase 2.

Training data:
  - "Normal" samples: flattened weight vectors from LSTM checkpoints +
    small Gaussian perturbations (simulating legitimate FL client updates)
  - "Poisoned" samples: weight vectors with large-scale perturbations,
    sign-flipping, and scaling attacks (simulating Byzantine clients)

For each config:
  1. Train AutoEncoder on normal weight vectors
  2. Evaluate detection of poisoned updates (F1, AUC-ROC, AUC-PR)
  3. Save model checkpoint to tests/saved_models/autoencoder/
"""
import os, sys, time, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from tests.utils.thesis_logger import ThesisLogger
import importlib.util

def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Improt models
_edge_models = _import_from_file('edge_models',
    os.path.join(ROOT, 'src', 'edge', 'models.py'))

np.random.seed(42)
torch.manual_seed(42)

LSTM_DIR = os.path.join(ROOT, 'tests', 'saved_models', 'lstm')
SAVE_DIR = os.path.join(ROOT, 'tests', 'saved_models', 'autoencoder')
os.makedirs(SAVE_DIR, exist_ok=True)

AutoEncoder = _edge_models.AutoEncoder

# ── hyperparameter grid ────────────────────────────────────────────────────
CONFIGS = {
    'A': dict(encoding_dim=32,  hidden_dims=[128, 64],       lr=1e-3,  epochs=100, batch_size=32),
    'B': dict(encoding_dim=64,  hidden_dims=[256, 128],      lr=1e-3,  epochs=100, batch_size=32),
    'C': dict(encoding_dim=128, hidden_dims=[512, 256],      lr=1e-3,  epochs=100, batch_size=32),
    'D': dict(encoding_dim=64,  hidden_dims=[256, 128],      lr=5e-4,  epochs=150, batch_size=32),
    'E': dict(encoding_dim=32,  hidden_dims=[128, 64],       lr=5e-3,  epochs=100, batch_size=32),
    'F': dict(encoding_dim=64,  hidden_dims=[256, 128, 64],  lr=1e-3,  epochs=150, batch_size=32),
}

# Number of synthetic normal/poisoned updates to generate
N_NORMAL_TRAIN  = 500
N_NORMAL_TEST   = 100
N_POISONED_TEST = 50


def _flatten_state_dict(sd):
    """Flatten all parameters of a state dict into a 1‑D numpy array."""
    return np.concatenate([v.cpu().numpy().flatten() for v in sd.values()])


def _generate_weight_data():
    """
    Generate training data from actual LSTM model weights.
    """
    base_weights = []
    
    # 1. Load checkpoints
    if os.path.exists(LSTM_DIR):
        for fname in sorted(os.listdir(LSTM_DIR)):
            if fname.endswith('.pt') and 'config_' in fname:
                try:
                    ckpt = torch.load(os.path.join(LSTM_DIR, fname),
                                      map_location='cpu', weights_only=True)
                    sd = ckpt['model_state_dict']
                    base_weights.append(_flatten_state_dict(sd))
                except Exception as e:
                    print(f"  Skipping {fname}: {e}")

    # FIX: Filter out checkpoints that have different architectures (different vector lengths)
    if base_weights:
        target_dim = len(base_weights[0])
        valid_weights = [w for w in base_weights if len(w) == target_dim]
        if len(valid_weights) < len(base_weights):
            print(f"  Warning: Discarded {len(base_weights) - len(valid_weights)} checkpoints due to mismatched architectures.")
        base_weights = valid_weights

    # 2. Fallback if no valid weights were found
    if not base_weights:
        print("  No matching LSTM checkpoints found, generating fresh weights ...")
        model = LSTMAnomalyDetector(input_size=38, hidden_size=64, num_layers=2)
        base_weights = [_flatten_state_dict(model.state_dict())]
        for _ in range(5):
            m = LSTMAnomalyDetector(input_size=38, hidden_size=64, num_layers=2)
            base_weights.append(_flatten_state_dict(m.state_dict()))

    base_weights = np.array(base_weights)
    mean_w = base_weights.mean(axis=0)
    std_w  = base_weights.std(axis=0) + 1e-8
    input_dim = len(mean_w)

    print(f"  Weight vector dim = {input_dim}")
    print(f"  Base checkpoints  = {len(base_weights)}")

    # --- Normal updates: small Gaussian perturbations around base weights ---
    train_normal = np.array([
        mean_w + np.random.normal(0, 0.01, input_dim) * std_w
        for _ in range(N_NORMAL_TRAIN)
    ], dtype=np.float32)

    test_normal = np.array([
        mean_w + np.random.normal(0, 0.01, input_dim) * std_w
        for _ in range(N_NORMAL_TEST)
    ], dtype=np.float32)

    # --- Poisoned updates: various Byzantine attack patterns ---
    poisoned = []
    for i in range(N_POISONED_TEST):
        attack = i % 4
        w = mean_w.copy()
        if attack == 0:
            scale = np.random.uniform(10, 50)
            w += np.random.normal(0, 0.01, input_dim) * std_w * scale
        elif attack == 1:
            noise = np.random.normal(0, 0.01, input_dim) * std_w
            w -= noise * np.random.uniform(5, 20)
        elif attack == 2:
            w = np.random.normal(0, 1.0, input_dim)
        else:
            n_corrupt = input_dim // 4
            w[:n_corrupt] += np.random.normal(0, 5.0, n_corrupt)
        poisoned.append(w)

    test_poisoned = np.array(poisoned, dtype=np.float32)

    return train_normal, test_normal, test_poisoned, input_dim


def _train_ae(model, data, cfg):
    """Train AutoEncoder and return per-epoch losses."""
    opt  = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    crit = nn.MSELoss()
    x    = torch.FloatTensor(data)
    ds   = torch.utils.data.TensorDataset(x)
    dl   = torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    losses = []
    for ep in range(cfg['epochs']):
        model.train()
        ep_loss = 0.0
        for (batch_x,) in dl:
            opt.zero_grad()
            rec, _ = model(batch_x)
            loss = crit(rec, batch_x)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        losses.append(avg)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"      epoch {ep+1:>4}/{cfg['epochs']}  loss={avg:.6f}")
    return losses


def _evaluate_ae(model, test_normal, test_poisoned):
    """Evaluate AutoEncoder on normal vs poisoned detection."""
    model.eval()
    all_data = np.concatenate([test_normal, test_poisoned], axis=0)
    y_true   = np.concatenate([np.zeros(len(test_normal)),
                               np.ones(len(test_poisoned))]).astype(int)
    x = torch.FloatTensor(all_data)

    with torch.no_grad():
        rec, _ = model(x)
        errors = torch.mean((x - rec) ** 2, dim=1).numpy()

    # Threshold at 95th percentile of normal errors
    thresh = np.percentile(errors[:len(test_normal)], 95)
    y_pred = (errors > thresh).astype(int)
    y_scores = errors / (errors.max() + 1e-9)

    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score':  float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_true, y_scores)),
        'auc_pr':    float(average_precision_score(y_true, y_scores)),
        'threshold': float(thresh),
    }


def main():
    logger = ThesisLogger("autoencoder_hyperparameter_sweep")

    print("Generating weight vector training data from LSTM checkpoints ...")
    train_normal, test_normal, test_poisoned, input_dim = _generate_weight_data()
    print(f"  train_normal={len(train_normal)}, test_normal={len(test_normal)}, "
          f"test_poisoned={len(test_poisoned)}, dim={input_dim}")

    results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Config {name}: encoding={cfg['encoding_dim']}, "
              f"hidden={cfg['hidden_dims']}, lr={cfg['lr']}, epochs={cfg['epochs']}")
        print(f"{'='*60}")

        model = AutoEncoder(
            input_dim=input_dim,
            encoding_dim=cfg['encoding_dim'],
            hidden_dims=cfg['hidden_dims'],
        )

        t0 = time.perf_counter()
        losses = _train_ae(model, train_normal, cfg)
        train_time = time.perf_counter() - t0

        metrics = _evaluate_ae(model, test_normal, test_poisoned)
        metrics.update({
            'config': name,
            'encoding_dim': cfg['encoding_dim'],
            'hidden_dims': str(cfg['hidden_dims']),
            'lr': cfg['lr'],
            'epochs': cfg['epochs'],
            'train_time_s': round(train_time, 2),
            'final_loss': losses[-1],
            'input_dim': input_dim,
        })

        # Save model
        ckpt_path = os.path.join(SAVE_DIR, f"autoencoder_config_{name}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'metrics': metrics,
            'losses': losses,
            'input_dim': input_dim,
        }, ckpt_path)
        print(f"  ✓ Saved → {ckpt_path}")

        for ep_idx, loss_val in enumerate(losses):
            logger.log_metric(f"config_{name}_epoch_{ep_idx}", {
                'config': name, 'epoch': ep_idx, 'train_loss': loss_val,
            })

        logger.log_metric(f"config_{name}_final", metrics)
        results[name] = metrics

        print(f"  F1={metrics['f1_score']:.4f}  AUC-ROC={metrics['auc_roc']:.4f}  "
              f"AUC-PR={metrics['auc_pr']:.4f}  Loss={losses[-1]:.6f}")

    # ── select best ────────────────────────────────────────────────────────
    best = max(results, key=lambda k: results[k]['f1_score'])
    print(f"\n{'='*60}")
    print(f"  BEST CONFIG: {best}  (F1={results[best]['f1_score']:.4f})")
    print(f"{'='*60}")

    import shutil
    src = os.path.join(SAVE_DIR, f"autoencoder_config_{best}.pt")
    dst = os.path.join(SAVE_DIR, "autoencoder_best.pt")
    shutil.copy2(src, dst)
    print(f"  ✓ Best model copied → {dst}")

    summary = {
        'best_config': best,
        'best_f1': results[best]['f1_score'],
        'best_auc_roc': results[best]['auc_roc'],
        'best_auc_pr': results[best]['auc_pr'],
        'input_dim': input_dim,
        'all_results': {k: {
            'f1': v['f1_score'], 'auc_roc': v['auc_roc'],
            'precision': v['precision'], 'recall': v['recall'],
        } for k, v in results.items()},
    }
    logger.log_metric("sweep_summary", summary)

    with open(os.path.join(SAVE_DIR, "sweep_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary → {os.path.join(SAVE_DIR, 'sweep_summary.json')}")


if __name__ == "__main__":
    main()
