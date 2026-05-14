"""
LSTM Hyperparameter Sweep — Train & Evaluate on SMD
=====================================================
Trains LSTMAnomalyDetector with 6 hyperparameter configurations on the
real Server Machine Dataset (SMD).  For each config:
  1. Train on SMD train split
  2. Evaluate F1/Precision/Recall/AUC-ROC/AUC-PR on SMD test split
  3. Save model checkpoint to tests/saved_models/lstm/
  4. Log all metrics for thesis plots

Selects and marks the best configuration by F1-score.
"""
import os, sys, time, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import test utilities directly (avoid path collision with src/utils/)
import importlib.util

def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_thesis_logger = _import_from_file('thesis_logger',
    os.path.join(ROOT, 'tests', 'utils', 'thesis_logger.py'))
_smd_loader = _import_from_file('smd_loader',
    os.path.join(ROOT, 'tests', 'utils', 'smd_loader.py'))
ThesisLogger = _thesis_logger.ThesisLogger
load_smd_dataset = _smd_loader.load_smd_dataset

# Import model class directly from its file
_edge_models = _import_from_file('edge_models',
    os.path.join(ROOT, 'src', 'edge', 'models.py'))
LSTMAnomalyDetector = _edge_models.LSTMAnomalyDetector

# ── constants ───────────────────────────────────────────────────────────────
INPUT_SIZE = 38
SEQ_LEN    = 50
MAX_TRAIN  = 4000   # use more data for better training
MAX_TEST   = 1000

np.random.seed(42)
torch.manual_seed(42)

SAVE_DIR = os.path.join(ROOT, 'tests', 'saved_models', 'lstm')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── hyperparameter grid ────────────────────────────────────────────────────
CONFIGS = {
    'A': dict(hidden_size=32,  num_layers=1, lr=1e-3,  dropout=0.1, epochs=100, batch_size=64),
    'B': dict(hidden_size=64,  num_layers=2, lr=1e-3,  dropout=0.2, epochs=100, batch_size=64),
    'C': dict(hidden_size=128, num_layers=2, lr=1e-3,  dropout=0.2, epochs=100, batch_size=64),
    'D': dict(hidden_size=64,  num_layers=3, lr=5e-4,  dropout=0.3, epochs=150, batch_size=64),
    'E': dict(hidden_size=128, num_layers=3, lr=5e-4,  dropout=0.2, epochs=150, batch_size=64),
    'F': dict(hidden_size=64,  num_layers=2, lr=1e-4,  dropout=0.2, epochs=200, batch_size=64),
}


def _train(model, data, cfg):
    """Train and return per-epoch losses."""
    opt  = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    crit = torch.nn.MSELoss()
    x    = torch.FloatTensor(data)
    ds   = torch.utils.data.TensorDataset(x)
    dl   = torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    losses = []
    for ep in range(cfg['epochs']):
        model.train()
        ep_loss = 0.0
        for (batch_x,) in dl:
            opt.zero_grad()
            rec  = model(batch_x)
            loss = crit(rec, batch_x)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        losses.append(avg)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"      epoch {ep+1:>4}/{cfg['epochs']}  loss={avg:.6f}")
    return losses


def _evaluate(model, test_normal, test_anomalous):
    """Return metrics dict."""
    model.eval()
    all_data = np.concatenate([test_normal, test_anomalous], axis=0)
    y_true   = np.concatenate([np.zeros(len(test_normal)),
                               np.ones(len(test_anomalous))]).astype(int)
    x = torch.FloatTensor(all_data)
    t0 = time.perf_counter()
    with torch.no_grad():
        errors = model.compute_reconstruction_error(x, reduction='mean').numpy()
    latency = (time.perf_counter() - t0) * 1000 / len(all_data)

    # Use percentile threshold on normal errors
    thresh = np.percentile(errors[:len(test_normal)], 95)
    y_pred = (errors > thresh).astype(int)
    y_scores = errors / (errors.max() + 1e-9)

    return {
        'precision':     float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':        float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score':      float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc':       float(roc_auc_score(y_true, y_scores)),
        'auc_pr':        float(average_precision_score(y_true, y_scores)),
        'latency_ms':    round(latency, 4),
        'threshold':     float(thresh),
    }


def main():
    logger = ThesisLogger("lstm_hyperparameter_sweep")

    print("Loading SMD dataset ...")
    train_data, test_normal, test_anomalous, _, _ = load_smd_dataset(
        seq_len=SEQ_LEN, max_train=MAX_TRAIN, max_test=MAX_TEST,
    )
    print(f"  train={len(train_data)}, test_normal={len(test_normal)}, "
          f"test_anomalous={len(test_anomalous)}")

    results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Config {name}: hidden={cfg['hidden_size']}, "
              f"layers={cfg['num_layers']}, lr={cfg['lr']}, "
              f"dropout={cfg['dropout']}, epochs={cfg['epochs']}")
        print(f"{'='*60}")

        model = LSTMAnomalyDetector(
            input_size=INPUT_SIZE,
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
        )

        t0 = time.perf_counter()
        losses = _train(model, train_data, cfg)
        train_time = time.perf_counter() - t0

        metrics = _evaluate(model, test_normal, test_anomalous)
        metrics.update({
            'config': name,
            'hidden_size': cfg['hidden_size'],
            'num_layers': cfg['num_layers'],
            'lr': cfg['lr'],
            'dropout': cfg['dropout'],
            'epochs': cfg['epochs'],
            'train_time_s': round(train_time, 2),
            'final_loss': losses[-1],
        })

        # Save model
        ckpt_path = os.path.join(SAVE_DIR, f"lstm_config_{name}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'metrics': metrics,
            'losses': losses,
        }, ckpt_path)
        print(f"  ✓ Saved → {ckpt_path}")

        # Save losses per epoch for learning curve
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

    # Copy best model
    import shutil
    src = os.path.join(SAVE_DIR, f"lstm_config_{best}.pt")
    dst = os.path.join(SAVE_DIR, "lstm_best.pt")
    shutil.copy2(src, dst)
    print(f"  ✓ Best model copied → {dst}")

    # Log summary
    summary = {
        'best_config': best,
        'best_f1': results[best]['f1_score'],
        'best_auc_roc': results[best]['auc_roc'],
        'best_auc_pr': results[best]['auc_pr'],
        'all_results': {k: {
            'f1': v['f1_score'], 'auc_roc': v['auc_roc'],
            'precision': v['precision'], 'recall': v['recall'],
        } for k, v in results.items()},
    }
    logger.log_metric("sweep_summary", summary)

    # Save summary JSON for easy parsing
    with open(os.path.join(SAVE_DIR, "sweep_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary → {os.path.join(SAVE_DIR, 'sweep_summary.json')}")


if __name__ == "__main__":
    main()
