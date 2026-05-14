"""
Generate Thesis Plots — Publication-quality figures for Chapter 11
=================================================================
Parses all benchmark logs and sweep summaries to produce PDF/PNG plots.
"""
import os, sys, json, glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif', 'axes.titlesize': 14, 'axes.labelsize': 12,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_DIR   = os.path.join(ROOT, 'tests', 'logs')
ASSET_DIR = os.path.join(ROOT, 'docs', 'thesis-final', 'chapters', '11_evaluation', 'assets')
LSTM_DIR  = os.path.join(ROOT, 'tests', 'saved_models', 'lstm')
AE_DIR    = os.path.join(ROOT, 'tests', 'saved_models', 'autoencoder')
os.makedirs(ASSET_DIR, exist_ok=True)


def _latest_log(prefix):
    """Find the most recent log file matching prefix."""
    pattern = os.path.join(LOG_DIR, f"{prefix}_*.log")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _parse_log(path):
    """Parse a JSON-lines log file."""
    entries = []
    if not path: return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: entries.append(json.loads(line))
                except json.JSONDecodeError: pass
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: LSTM Hyperparameter Comparison (bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def plot_lstm_sweep():
    summary_path = os.path.join(LSTM_DIR, 'sweep_summary.json')
    if not os.path.exists(summary_path): return
    with open(summary_path) as f: data = json.load(f)

    configs = sorted(data['all_results'].keys())
    f1s     = [data['all_results'][c]['f1'] for c in configs]
    aucs    = [data['all_results'][c]['auc_roc'] for c in configs]
    precs   = [data['all_results'][c]['precision'] for c in configs]
    recs    = [data['all_results'][c]['recall'] for c in configs]

    x = np.arange(len(configs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, f1s, w, label='F1-Score', color='#2196F3')
    ax.bar(x - 0.5*w, aucs, w, label='AUC-ROC', color='#4CAF50')
    ax.bar(x + 0.5*w, precs, w, label='Precision', color='#FF9800')
    ax.bar(x + 1.5*w, recs, w, label='Recall', color='#9C27B0')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('LSTM-AE Hyperparameter Sweep — SMD Dataset')
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    best = data['best_config']
    best_idx = configs.index(best)
    ax.annotate(f'Best: {best}\nF1={data["best_f1"]:.3f}',
                xy=(best_idx, f1s[best_idx]), xytext=(best_idx+0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10,
                color='red', fontweight='bold')
    fig.savefig(os.path.join(ASSET_DIR, 'lstm_sweep_comparison.pdf'))
    fig.savefig(os.path.join(ASSET_DIR, 'lstm_sweep_comparison.png'))
    plt.close(fig)
    print("  ✓ lstm_sweep_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: LSTM Best Model Learning Curve
# ═══════════════════════════════════════════════════════════════════════════
def plot_lstm_learning_curve():
    import torch
    best_path = os.path.join(LSTM_DIR, 'lstm_best.pt')
    if not os.path.exists(best_path): return
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    losses = ckpt.get('losses', [])
    if not losses: return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(losses, color='#2196F3', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('LSTM-AE Training Convergence — Best Configuration (C)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(ASSET_DIR, 'lstm_learning_curve.pdf'))
    fig.savefig(os.path.join(ASSET_DIR, 'lstm_learning_curve.png'))
    plt.close(fig)
    print("  ✓ lstm_learning_curve")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: AutoEncoder Sweep Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_ae_sweep():
    summary_path = os.path.join(AE_DIR, 'sweep_summary.json')
    if not os.path.exists(summary_path): return
    with open(summary_path) as f: data = json.load(f)

    configs = sorted(data['all_results'].keys())
    f1s   = [data['all_results'][c]['f1'] for c in configs]
    aucs  = [data['all_results'][c]['auc_roc'] for c in configs]
    precs = [data['all_results'][c]['precision'] for c in configs]
    recs  = [data['all_results'][c]['recall'] for c in configs]

    x = np.arange(len(configs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, f1s, w, label='F1-Score', color='#E91E63')
    ax.bar(x - 0.5*w, aucs, w, label='AUC-ROC', color='#00BCD4')
    ax.bar(x + 0.5*w, precs, w, label='Precision', color='#FFC107')
    ax.bar(x + 1.5*w, recs, w, label='Recall', color='#8BC34A')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('AutoEncoder Poisoned Model Detection — FL Weight Vectors')
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    best = data['best_config']
    best_idx = configs.index(best)
    ax.annotate(f'Best: {best}\nF1={data["best_f1"]:.3f}',
                xy=(best_idx, f1s[best_idx]), xytext=(best_idx+0.8, 0.6),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10,
                color='red', fontweight='bold')
    fig.savefig(os.path.join(ASSET_DIR, 'ae_sweep_comparison.pdf'))
    fig.savefig(os.path.join(ASSET_DIR, 'ae_sweep_comparison.png'))
    plt.close(fig)
    print("  ✓ ae_sweep_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: DP Privacy–Accuracy Tradeoff
# ═══════════════════════════════════════════════════════════════════════════
def plot_dp_tradeoff():
    log = _latest_log("benchmark_B_dp_impact")
    entries = _parse_log(log)
    if not entries: return

    # Extract final results per sigma
    finals = [e for e in entries if '_final' in e.get('step', '')]
    if not finals:
        finals = [e for e in entries if 'sigma' in e and 'round' not in e]

    # Also extract convergence curves per sigma
    sigmas_seen = {}
    for e in entries:
        s = e.get('sigma')
        r = e.get('round')
        if s is not None and r is not None:
            sigmas_seen.setdefault(s, []).append((r, e.get('auc_roc', 0), e.get('loss', 0)))

    if sigmas_seen:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        for i, (sigma, pts) in enumerate(sorted(sigmas_seen.items())):
            pts.sort(key=lambda x: x[0])
            rounds = [p[0] for p in pts]
            auc_vals = [p[1] for p in pts]
            loss_vals = [p[2] for p in pts]
            lbl = f'σ={sigma}' if sigma > 0 else 'No DP'
            ax1.plot(rounds, auc_vals, marker='o', markersize=3, label=lbl,
                     color=colors[i % len(colors)], linewidth=1.5)
            ax2.semilogy(rounds, loss_vals, marker='o', markersize=3, label=lbl,
                         color=colors[i % len(colors)], linewidth=1.5)

        ax1.set_xlabel('Training Round'); ax1.set_ylabel('AUC-ROC')
        ax1.set_title('Privacy–Accuracy Tradeoff')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('Training Round'); ax2.set_ylabel('Loss (log)')
        ax2.set_title('Training Convergence Under DP')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSET_DIR, 'dp_tradeoff.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'dp_tradeoff.png'))
        plt.close(fig)
        print("  ✓ dp_tradeoff")
    else:
        print("  ✗ dp_tradeoff (no convergence data)")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 5: Benchmark A — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════
def plot_anomaly_detection():
    log = _latest_log("benchmark_A_anomaly_detection")
    entries = _parse_log(log)
    if not entries: return

    models = {}
    convergence = []
    for e in entries:
        step = e.get('step', '')
        if step.startswith('eval_'):
            models[e.get('model', step)] = e
        elif 'learning_curve' in step and 'f1_score' in e:
            convergence.append(e)

    if models:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = list(models.keys())
        pretty = [n.replace('_', ' ').title() for n in names]
        metrics = ['f1_score', 'auc_roc', 'precision', 'recall']
        labels  = ['F1-Score', 'AUC-ROC', 'Precision', 'Recall']
        x = np.arange(len(names))
        w = 0.18
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
        for i, (met, lbl) in enumerate(zip(metrics, labels)):
            vals = [models[n].get(met, 0) for n in names]
            ax.bar(x + (i-1.5)*w, vals, w, label=lbl, color=colors[i])
        ax.set_xticks(x); ax.set_xticklabels(pretty, fontsize=9)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
        ax.set_title('Anomaly Detection Performance — SMD Dataset')
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(ASSET_DIR, 'anomaly_detection_comparison.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'anomaly_detection_comparison.png'))
        plt.close(fig)
        print("  ✓ anomaly_detection_comparison")

    if convergence:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        rounds = [e['round'] for e in convergence]
        losses = [e['train_loss'] for e in convergence]
        f1s    = [e.get('f1_score', 0) for e in convergence]
        ax1.semilogy(rounds, losses, color='#2196F3', linewidth=1.5, marker='o', markersize=3)
        ax1.set_xlabel('Round'); ax1.set_ylabel('MSE Loss (log)')
        ax1.set_title('Cold-Start Convergence')
        ax2.plot(rounds, f1s, color='#4CAF50', linewidth=1.5, marker='s', markersize=3)
        ax2.set_xlabel('Round'); ax2.set_ylabel('F1-Score')
        ax2.set_title('F1 vs Training Round')
        ax2.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSET_DIR, 'cold_start_convergence.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'cold_start_convergence.png'))
        plt.close(fig)
        print("  ✓ cold_start_convergence")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 6: FL Overhead — Payload Matrix
# ═══════════════════════════════════════════════════════════════════════════
def plot_fl_overhead():
    log = _latest_log("benchmark_C_fl_overhead")
    entries = _parse_log(log)
    if not entries: return

    payloads = [e for e in entries if e.get('step') == 'payload_eval']
    adaptive = [e for e in entries if e.get('step') == 'adaptive_compression']
    rounds   = [e for e in entries if 'fl_round' in e.get('step', '')]

    if payloads:
        sers = sorted(set(e['serialization'] for e in payloads))
        comps = sorted(set(e['compression'] for e in payloads))
        matrix = np.zeros((len(sers), len(comps)))
        for e in payloads:
            si = sers.index(e['serialization'])
            ci = comps.index(e['compression'])
            matrix[si, ci] = e['payload_size_kb']

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(comps))); ax.set_xticklabels(comps, fontsize=9)
        ax.set_yticks(range(len(sers))); ax.set_yticklabels(sers, fontsize=9)
        for i in range(len(sers)):
            for j in range(len(comps)):
                ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=9)
        ax.set_title('Payload Size (KB) — Serialization × Compression')
        fig.colorbar(im, label='Size (KB)')
        fig.savefig(os.path.join(ASSET_DIR, 'payload_size_matrix.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'payload_size_matrix.png'))
        plt.close(fig)
        print("  ✓ payload_size_matrix")

    if rounds:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        rr = sorted(rounds, key=lambda e: e.get('round', 0))
        rnums = [e['round'] for e in rr]
        train_ms = [e['local_train_ms'] for e in rr]
        comp_ms  = [e['compression_ms'] for e in rr]
        agg_ms   = [e['aggregation_ms'] for e in rr]
        ax.bar(rnums, train_ms, label='Local Training', color='#2196F3')
        ax.bar(rnums, comp_ms, bottom=train_ms, label='Compression', color='#FF9800')
        bottoms = [t+c for t,c in zip(train_ms, comp_ms)]
        ax.bar(rnums, agg_ms, bottom=bottoms, label='Aggregation', color='#4CAF50')
        ax.set_xlabel('Round'); ax.set_ylabel('Time (ms)')
        ax.set_title('FL Round Latency Breakdown')
        ax.legend()
        fig.savefig(os.path.join(ASSET_DIR, 'fl_round_latency.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'fl_round_latency.png'))
        plt.close(fig)
        print("  ✓ fl_round_latency")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 7: RCA Accuracy
# ═══════════════════════════════════════════════════════════════════════════
def plot_rca():
    log = _latest_log("benchmark_D_rca_accuracy")
    entries = _parse_log(log)
    if not entries: return

    summaries = {}
    for e in entries:
        step = e.get('step', '')
        if 'summary_' in step:
            summaries[step] = e

    fw_tt = summaries.get('trainticket_summary_framework', {})
    rand_tt = summaries.get('trainticket_summary_random', {})
    fw_al = summaries.get('alibaba_summary_framework', {})
    rand_al = summaries.get('alibaba_summary_random', {})

    if fw_tt:
        fig, ax = plt.subplots(figsize=(9, 5))
        groups = ['Train-Ticket\n(11 svc)', 'Alibaba-Scale\n(50 svc)']
        metrics = ['top_1_accuracy', 'top_3_accuracy', 'mrr']
        labels = ['Top-1 Acc', 'Top-3 Acc', 'MRR']
        x = np.arange(len(groups))
        w = 0.12
        colors_fw = ['#2196F3', '#1976D2', '#0D47A1']
        colors_rand = ['#FF9800', '#F57C00', '#E65100']
        for i, (met, lbl) in enumerate(zip(metrics, labels)):
            fw_vals = [fw_tt.get(met, 0), fw_al.get(met, 0)]
            rand_vals = [rand_tt.get(met, 0), rand_al.get(met, 0)]
            ax.bar(x + (i-1)*w*2 - w/2, fw_vals, w, label=f'Framework {lbl}', color=colors_fw[i])
            ax.bar(x + (i-1)*w*2 + w/2, rand_vals, w, label=f'Random {lbl}', color=colors_rand[i])
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
        ax.set_title('Root Cause Analysis — Framework vs Random Baseline')
        ax.legend(ncol=2, fontsize=8, loc='upper right')
        fig.savefig(os.path.join(ASSET_DIR, 'rca_accuracy.pdf'))
        fig.savefig(os.path.join(ASSET_DIR, 'rca_accuracy.png'))
        plt.close(fig)
        print("  ✓ rca_accuracy")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 8: Adaptive Threshold Trajectory
# ═══════════════════════════════════════════════════════════════════════════
def plot_adaptive_threshold():
    log = _latest_log("benchmark_E_adaptive_threshold")
    entries = _parse_log(log)
    if not entries: return

    hours = sorted(entries, key=lambda e: e.get('minute', 0))
    if not hours: return

    minutes = [e['minute'] for e in hours]
    static  = [e.get('static_threshold', 0) for e in hours]
    ewma    = [e.get('ewma_threshold', 0) for e in hours]
    ql      = [e.get('ql_threshold', 0) for e in hours]
    vals    = [e.get('metric_value', 0) for e in hours]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(minutes, vals, alpha=0.4, color='gray', linewidth=0.5, label='Metric Value')
    ax1.plot(minutes, static, '--', color='#F44336', linewidth=1.5, label='Static')
    ax1.plot(minutes, ewma, '-', color='#FF9800', linewidth=1.5, label='EWMA')
    ax1.plot(minutes, ql, '-', color='#2196F3', linewidth=1.5, label='Q-Learning')
    ax1.set_ylabel('Threshold / Score')
    ax1.set_title('Adaptive Threshold Strategies Over Time')
    ax1.legend(loc='upper right')

    s_alerts = [e.get('static_alerts', 0) for e in hours]
    e_alerts = [e.get('ewma_alerts', 0) for e in hours]
    q_alerts = [e.get('ql_alerts', 0) for e in hours]
    ax2.plot(minutes, s_alerts, '--', color='#F44336', linewidth=1.5, label='Static')
    ax2.plot(minutes, e_alerts, '-', color='#FF9800', linewidth=1.5, label='EWMA')
    ax2.plot(minutes, q_alerts, '-', color='#2196F3', linewidth=1.5, label='Q-Learning')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Alerts')
    ax2.set_title('Alert Volume Comparison')
    ax2.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, 'adaptive_threshold.pdf'))
    fig.savefig(os.path.join(ASSET_DIR, 'adaptive_threshold.png'))
    plt.close(fig)
    print("  ✓ adaptive_threshold")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 9: Scalability
# ═══════════════════════════════════════════════════════════════════════════
def plot_scalability():
    log = _latest_log("benchmark_F_scalability")
    entries = _parse_log(log)
    if not entries: return

    entries.sort(key=lambda e: e.get('num_nodes', 0))
    nodes = [e['num_nodes'] for e in entries]
    times = [e['aggregation_time_ms'] for e in entries]
    mem   = [e.get('memory_delta_mb', 0) for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(nodes, times, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax1.plot(nodes, [nodes[0]/nodes[0]*times[0]*n/nodes[0] for n in nodes],
             '--', color='gray', alpha=0.5, label='Linear scaling')
    ax1.set_xlabel('Number of Edge Nodes')
    ax1.set_ylabel('Aggregation Time (ms)')
    ax1.set_title('FedAvg Aggregation Scalability')
    ax1.legend()
    ax1.set_xscale('log'); ax1.set_yscale('log')

    ax2.bar(range(len(nodes)), mem, color='#FF9800')
    ax2.set_xticks(range(len(nodes))); ax2.set_xticklabels(nodes, fontsize=8, rotation=45)
    ax2.set_xlabel('Number of Edge Nodes')
    ax2.set_ylabel('Memory Delta (MB)')
    ax2.set_title('Memory Usage per Aggregation')
    fig.tight_layout()
    fig.savefig(os.path.join(ASSET_DIR, 'scalability.pdf'))
    fig.savefig(os.path.join(ASSET_DIR, 'scalability.png'))
    plt.close(fig)
    print("  ✓ scalability")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Generating thesis plots ...")
    plot_lstm_sweep()
    plot_lstm_learning_curve()
    plot_ae_sweep()
    plot_dp_tradeoff()
    plot_anomaly_detection()
    plot_fl_overhead()
    plot_rca()
    plot_adaptive_threshold()
    plot_scalability()
    print(f"\nAll plots saved to: {ASSET_DIR}")

if __name__ == "__main__":
    main()
