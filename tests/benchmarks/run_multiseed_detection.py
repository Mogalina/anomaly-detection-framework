"""
Multi-Seed Detection Benchmark (C3 — Statistical Variance)
==========================================================
Retrains the LSTM-AE (Config C: h=128, L=2, 100 epochs) with different
random seeds to quantify the variance in detection metrics.

This matches the thesis sweep methodology: centralized training on the
full SMD training set, evaluated with 95th-percentile threshold on the
test set. The pre-trained model (seed 42) is also re-evaluated for
consistency checking.

Usage:
    python tests/benchmarks/run_multiseed_detection.py [--seeds 42 123 7]

Output: prints mean ± std and LaTeX-ready table rows.
"""
import os, sys, time, argparse
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
_loader = _import_from_file('smd_loader', os.path.join(ROOT, 'tests', 'utils', 'smd_loader.py'))

LSTMAnomalyDetector = _models.LSTMAnomalyDetector
load_smd_dataset = _loader.load_smd_dataset

# Config C — matches thesis sweep
INPUT_SIZE    = 38
SEQ_LEN       = 50
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 0.001
NUM_TRAIN     = 4000
NUM_TEST      = 1000


def _train(model, data, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train model on data — matches sweep methodology exactly."""
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    x    = torch.FloatTensor(data)
    ds   = torch.utils.data.TensorDataset(x)
    dl   = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for (bx,) in dl:
            opt.zero_grad()
            loss = crit(model(bx), bx); loss.backward(); opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(dl))
    return losses


def _evaluate(model, test_normal, test_anomalous, y_true):
    """Evaluate with 95th-percentile threshold — matches sweep methodology."""
    model.eval()
    all_data = np.concatenate([test_normal, test_anomalous], axis=0)
    x = torch.FloatTensor(all_data)
    with torch.no_grad():
        errors = model.compute_reconstruction_error(x, reduction='mean').numpy()
    thresh = np.percentile(errors[:len(test_normal)], 95)
    y_pred = (errors > thresh).astype(int)
    y_scores = errors / (errors.max() + 1e-9)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'auc_roc':   float(roc_auc_score(y_true, y_scores)),
        'auc_pr':    float(average_precision_score(y_true, y_scores)),
        'final_loss': 0.0,  # filled below
    }


def run_single_seed(seed, train_data, test_normal, test_anomalous, y_true):
    """Train and evaluate one model from scratch with the given seed."""
    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = LSTMAnomalyDetector(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                                dropout=DROPOUT)
    losses = _train(model, train_data)
    results = _evaluate(model, test_normal, test_anomalous, y_true)
    results['final_loss'] = losses[-1]
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-seed detection benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7],
                        help='Random seeds to evaluate (default: 42 123 7)')
    args = parser.parse_args()

    print(f"Multi-seed LSTM-AE Detection Benchmark")
    print(f"Config C: h={HIDDEN_SIZE}, L={NUM_LAYERS}, dropout={DROPOUT}, "
          f"epochs={EPOCHS}, lr={LR}")
    print(f"Data:     train={NUM_TRAIN}, test={NUM_TEST}")
    print(f"Seeds:    {args.seeds}\n")

    # Load data ONCE (data is deterministic from files, only model init varies)
    # Note: we seed before loading to handle the internal shuffle in smd_loader
    np.random.seed(args.seeds[0])
    train_data, test_normal, test_anomalous, _, _ = load_smd_dataset(
        seq_len=SEQ_LEN, max_train=NUM_TRAIN, max_test=NUM_TEST,
    )
    y_true = np.concatenate([np.zeros(len(test_normal)),
                             np.ones(len(test_anomalous))]).astype(int)
    print(f"Loaded:   train={len(train_data)}, normal_test={len(test_normal)}, "
          f"anom_test={len(test_anomalous)}")
    print(f"{'='*72}\n")

    all_results = []

    for seed in args.seeds:
        print(f"  Seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        results = run_single_seed(seed, train_data, test_normal,
                                  test_anomalous, y_true)
        elapsed = time.time() - t0
        all_results.append(results)
        print(f"done ({elapsed:.1f}s)  "
              f"F1={results['f1']:.4f}  AUC={results['auc_roc']:.4f}  "
              f"P={results['precision']:.4f}  R={results['recall']:.4f}  "
              f"loss={results['final_loss']:.2e}")

    # ── Aggregate ──────────────────────────────────────────────────────
    metrics = ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']
    print(f"\n{'='*72}")
    print("RESULTS: mean ± std across seeds")
    print(f"{'='*72}\n")

    for m in metrics:
        vals = [r[m] for r in all_results]
        mean, std = np.mean(vals), np.std(vals)
        print(f"  {m:>12s}: {mean:.4f} ± {std:.4f}  "
              f"[{', '.join(f'{v:.4f}' for v in vals)}]")

    # ── LaTeX row ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("LaTeX row for Table 6.2:")
    print(f"{'='*72}\n")
    row = ["LSTM-AE (Config C)"]
    for m in metrics:
        vals = [r[m] for r in all_results]
        mean, std = np.mean(vals), np.std(vals)
        row.append(f"${mean:.3f} \\pm {std:.3f}$")
    print(" & ".join(row) + " \\\\")

    # ── Reference comparison ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Comparison to thesis Table 6.2 (seed 42 pre-trained):")
    print(f"{'='*72}")
    print(f"  Thesis:    F1=0.839  AUC-ROC=0.950  P=0.813  R=0.868")
    if any(s == 42 for s in args.seeds):
        idx = args.seeds.index(42)
        r = all_results[idx]
        print(f"  This run:  F1={r['f1']:.3f}  AUC-ROC={r['auc_roc']:.3f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}")
        delta = abs(r['f1'] - 0.839)
        print(f"  F1 delta:  {delta:.4f} "
              f"({'MATCH' if delta < 0.01 else 'DRIFT — check data loading'})")


if __name__ == "__main__":
    main()
