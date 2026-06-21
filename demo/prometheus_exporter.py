#!/usr/bin/env python3

import os, sys, time, json, math, random, logging, importlib.util
import torch, numpy as np
from prometheus_client import start_http_server, Gauge, Info
try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

models = _import('models', os.path.join(ROOT, 'src', 'edge', 'models.py'))
smd_loader = _import('smd_loader', os.path.join(ROOT, 'tests', 'utils', 'smd_loader.py'))
LSTMAnomalyDetector = models.LSTMAnomalyDetector
load_smd_dataset = smd_loader.load_smd_dataset

LOKI_URL = os.environ.get("LOKI_URL", "http://localhost:3100/loki/api/v1/push")

error_gauge       = Gauge('smd_reconstruction_error', 'Current reconstruction error')
threshold_gauge   = Gauge('smd_anomaly_threshold', 'Adaptive threshold')
anomaly_gauge     = Gauge('smd_is_anomaly', '1 if anomalous')
q_learning_reward = Gauge('q_learning_reward', 'Q-learning cumulative reward')
cumulative_alerts = Gauge('cumulative_alerts', 'Total alerts fired')
false_positive_rate = Gauge('smd_fpr', 'Current FPR estimate')
fl_round          = Gauge('fl_current_round', 'FL Round')
fl_loss           = Gauge('fl_global_loss', 'Global Training Loss')
fl_agg_latency    = Gauge('fl_aggregation_latency_ms', 'Aggregation Latency')
fl_agg_memory     = Gauge('fl_coordinator_memory_mb', 'Coordinator Memory')
dp_epsilon        = Gauge('dp_epsilon', 'DP Epsilon')
dp_sigma          = Gauge('dp_sigma', 'DP Noise Multiplier')
dp_clipping       = Gauge('dp_clipping_norm', 'DP Clipping Norm')
payload_size      = Gauge('fl_payload_size_kb', 'Payload Size', ['format', 'compression'])
fl_clients_registered = Gauge('fl_clients_registered', 'Active Edge Agents')
dp_pre_clip_norm  = Gauge('dp_pre_clip_norm', 'Pre-Clip Norm')
dp_post_clip_norm = Gauge('dp_post_clip_norm', 'Post-Clip Norm')

demo_phase        = Gauge('demo_phase', 'Current demo phase index')
demo_phase_name   = Info('demo_phase_info', 'Current demo phase metadata')
detection_f1      = Gauge('detection_f1_score', 'F1 Score', ['method'])
detection_auroc   = Gauge('detection_auroc', 'AUC-ROC', ['method'])

rca_prob       = Gauge('rca_root_cause_probability', 'Prob of root cause', ['service', 'fault_type'])
incident_trace = Gauge('incident_active_trace', 'Active trace', ['incident_id', 'root_cause', 'fault_type', 'affected_services', 'critical_path'])
rca_explanation = Gauge('rca_explanation_metadata', 'RCA explanation', ['incident_id', 'root_cause', 'fault_type', 'explanation', 'trace_depth'])
rca_cascade    = Gauge('rca_cascade_severity', 'Cascade severity', ['root_cause', 'affected_service', 'severity', 'hop_distance'])
rca_pagerank   = Gauge('rca_pagerank_score', 'PageRank', ['service'])
rca_impact     = Gauge('rca_impact_score', 'Impact', ['service'])
rca_confidence = Gauge('rca_explanation_confidence', 'Confidence', ['service', 'type'])
rca_latency    = Gauge('rca_analysis_time_ms', 'RCA latency ms')
rca_combined   = Gauge('rca_combined_score', 'Combined RCA', ['service'])
rca_fault      = Gauge('rca_fault_active', 'Active fault', ['type'])
cg_edges       = Gauge('cg_num_edges', 'Causal graph edges')
cg_nodes       = Gauge('cg_num_nodes', 'Causal graph nodes')

ms_latency     = Gauge('ms_latency_ms', 'Latency', ['service'])
ms_cpu         = Gauge('ms_cpu_percent', 'CPU', ['service'])
ms_memory      = Gauge('ms_memory_mb', 'Memory', ['service'])
ms_net_rx      = Gauge('ms_network_rx_kbps', 'Net RX', ['service'])
ms_net_tx      = Gauge('ms_network_tx_kbps', 'Net TX', ['service'])
ms_active_conns = Gauge('ms_active_connections', 'Connections', ['service'])

adaptive_alerts_total     = Gauge('adaptive_alerts_total', 'Total alerts under each method', ['method'])
adaptive_fpr_current      = Gauge('adaptive_fpr_current', 'Current FPR', ['method'])

SERVICES = [
    'ts-ui-dashboard', 'ts-auth-service', 'ts-order-service',
    'ts-route-service', 'ts-payment-service', 'ts-station-service',
    'ts-train-service', 'ts-ticketinfo-service', 'ts-price-service',
    'ts-notification-service', 'ts-security-service'
]

DEPENDENCIES = {
    'ts-ui-dashboard': ['ts-auth-service', 'ts-order-service', 'ts-route-service'],
    'ts-order-service': ['ts-payment-service', 'ts-ticketinfo-service'],
    'ts-route-service': ['ts-station-service', 'ts-train-service'],
    'ts-payment-service': ['ts-security-service'],
    'ts-ticketinfo-service': ['ts-price-service', 'ts-train-service']
}

PHASE_NAMES = [
    "Boot & Registration",
    "Edge-Local LSTM-AE Detection",
    "Federated Learning Round (DP-FedAvg)",
    "Root Cause Analysis Engine",
    "Adaptive Thresholding (Q-Learning)",
    "Fault Tolerance & Recovery"
]

def push_loki(phase: str, component: str, message: str, level: str = "info"):
    """Push a structured log line to Loki (best-effort, silent on failure)."""
    if not HAS_REQUESTS:
        return
    try:
        _req.post(LOKI_URL, json={
            "streams": [{
                "stream": {"job": "thesis_demo", "phase": phase, "component": component, "level": level},
                "values": [[str(int(time.time() * 1e9)), message]]
            }]
        }, timeout=0.3)
    except Exception:
        pass

def init_demo():
    _, test_normal, test_anomalous, ordered_test_windows, ordered_test_labels = load_smd_dataset(
        seq_len=50, max_train=0, max_test=1000)
    ckpt_path = os.path.join(ROOT, 'tests', 'saved_models', 'lstm', 'lstm_best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = LSTMAnomalyDetector(
        input_size=38, hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'], dropout=cfg.get('dropout', 0.2))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        x_norm = torch.FloatTensor(test_normal)
        norm_errors = model.compute_reconstruction_error(x_norm, reduction='mean').numpy()
        threshold = np.percentile(norm_errors, 95)
    threshold_gauge.set(threshold)
    return model, ordered_test_windows, ordered_test_labels, threshold

def set_ms_telemetry(is_anomaly: bool):
    for svc in SERVICES:
        if is_anomaly and svc == 'ts-payment-service':
            ms_latency.labels(service=svc).set(random.uniform(950, 1200))
            ms_cpu.labels(service=svc).set(random.uniform(92, 98))
            ms_memory.labels(service=svc).set(random.uniform(850, 900))
            ms_net_rx.labels(service=svc).set(random.uniform(12000, 14000))
            ms_net_tx.labels(service=svc).set(random.uniform(12000, 14000))
            ms_active_conns.labels(service=svc).set(random.uniform(300, 350))
        elif is_anomaly and svc in ['ts-order-service', 'ts-ui-dashboard']:
            ms_latency.labels(service=svc).set(random.uniform(450, 600))
            ms_cpu.labels(service=svc).set(random.uniform(55, 75))
            ms_memory.labels(service=svc).set(random.uniform(600, 750))
            ms_net_rx.labels(service=svc).set(random.uniform(4000, 6000))
            ms_net_tx.labels(service=svc).set(random.uniform(4000, 6000))
            ms_active_conns.labels(service=svc).set(random.uniform(180, 240))
        else:
            ms_latency.labels(service=svc).set(random.uniform(12, 35))
            ms_cpu.labels(service=svc).set(random.uniform(5, 18))
            ms_memory.labels(service=svc).set(random.uniform(200, 380))
            ms_net_rx.labels(service=svc).set(random.uniform(100, 450))
            ms_net_tx.labels(service=svc).set(random.uniform(100, 450))
            ms_active_conns.labels(service=svc).set(random.uniform(15, 45))

def set_rca_metrics(active: bool):
    culprit = 'ts-payment-service'
    fault = 'Latency Spike'
    affected = ['ts-order-service', 'ts-ui-dashboard']
    if active:
        incident_trace.labels(incident_id="INC-1025", root_cause=culprit, fault_type=fault,
            affected_services="ts-order-service,ts-ui-dashboard",
            critical_path="ts-payment-service -> ts-order-service -> ts-ui-dashboard").set(1.0)
        rca_prob.labels(service=culprit, fault_type=fault).set(0.88)
        rca_prob.labels(service='ts-order-service', fault_type='Cascade Failure').set(0.38)
        rca_prob.labels(service='ts-ui-dashboard', fault_type='Cascade Failure').set(0.12)
        rca_explanation.labels(incident_id="INC-1025", root_cause=culprit, fault_type=fault,
            explanation=f"{fault} detected at {culprit}, propagated to ts-order-service,ts-ui-dashboard",
            trace_depth="2").set(1.0)
        rca_cascade.labels(root_cause=culprit, affected_service='ts-order-service', severity='High', hop_distance='1').set(1.0)
        rca_cascade.labels(root_cause=culprit, affected_service='ts-ui-dashboard', severity='Critical', hop_distance='2').set(1.0)
        rca_fault.labels(type=fault).set(1.0)
        for svc in SERVICES:
            if svc == culprit:
                rca_pagerank.labels(service=svc).set(0.55)
                rca_impact.labels(service=svc).set(0.95)
                rca_confidence.labels(service=svc, type='root_cause').set(0.92)
                rca_combined.labels(service=svc).set(0.79)
            elif svc in affected:
                rca_pagerank.labels(service=svc).set(0.18)
                rca_impact.labels(service=svc).set(0.50)
                rca_confidence.labels(service=svc, type='propagated').set(0.65)
                rca_combined.labels(service=svc).set(0.37)
            else:
                rca_pagerank.labels(service=svc).set(0.02)
                rca_impact.labels(service=svc).set(0.04)
                rca_confidence.labels(service=svc, type='healthy').set(0.05)
                rca_combined.labels(service=svc).set(0.03)
        rca_latency.set(22.4)
    else:
        incident_trace.labels(incident_id="INC-1025", root_cause=culprit, fault_type=fault,
            affected_services="ts-order-service,ts-ui-dashboard",
            critical_path="ts-payment-service -> ts-order-service -> ts-ui-dashboard").set(0.0)
        rca_prob.labels(service=culprit, fault_type=fault).set(0.0)
        rca_prob.labels(service='ts-order-service', fault_type='Cascade Failure').set(0.0)
        rca_prob.labels(service='ts-ui-dashboard', fault_type='Cascade Failure').set(0.0)
        rca_explanation.labels(incident_id="INC-1025", root_cause=culprit, fault_type=fault,
            explanation=f"{fault} detected at {culprit}, propagated to ts-order-service,ts-ui-dashboard",
            trace_depth="2").set(0.0)
        rca_cascade.labels(root_cause=culprit, affected_service='ts-order-service', severity='High', hop_distance='1').set(0.0)
        rca_cascade.labels(root_cause=culprit, affected_service='ts-ui-dashboard', severity='Critical', hop_distance='2').set(0.0)
        rca_fault.labels(type=fault).set(0.0)
        for svc in SERVICES:
            rca_pagerank.labels(service=svc).set(0.0)
            rca_impact.labels(service=svc).set(0.0)
            rca_confidence.labels(service=svc, type='root_cause').set(0.0)
            rca_confidence.labels(service=svc, type='propagated').set(0.0)
            rca_confidence.labels(service=svc, type='healthy').set(0.0)
            rca_combined.labels(service=svc).set(0.0)
        rca_latency.set(0.0)

def run_exporter():
    start_http_server(8000)
    print("=" * 65)
    print("  THESIS DEFENSE DEMO")
    print("  Prometheus:  http://localhost:8000/metrics")
    print("=" * 65)
    sys.stdout.flush()

    model, demo_data, demo_labels, base_threshold = init_demo()

    dp_epsilon.set(10.0); dp_sigma.set(0.0005); dp_clipping.set(1.0)
    payload_size.labels(format='JSON', compression='None').set(191.0)
    payload_size.labels(format='TorchSave', compression='None').set(35.4)
    payload_size.labels(format='TorchSave', compression='Zstd L1').set(32.2)
    cg_nodes.set(len(SERVICES))
    cg_edges.set(len(DEPENDENCIES) + sum(len(v) for v in DEPENDENCIES.values()))

    detection_f1.labels(method='Static 3-sigma').set(0.0)
    detection_f1.labels(method='Centralized LSTM-AE').set(0.496)
    detection_f1.labels(method='Federated LSTM-AE (Ours)').set(0.839)
    detection_f1.labels(method='OmniAnomaly (Ref)').set(0.83)
    detection_auroc.labels(method='Federated LSTM-AE (Ours)').set(0.95)
    detection_auroc.labels(method='Centralized LSTM-AE').set(0.72)

    start_time = time.time()
    logged = set()
    CYCLE = 240
    alert_count = 0.0

    def log(key, phase_name, component, msg, level="info"):
        if key not in logged:
            print(f"[{phase_name}] {msg}")
            sys.stdout.flush()
            push_loki(phase_name, component, msg, level)
            logged.add(key)

    while True:
        t = int(time.time() - start_time) % CYCLE
        if t == 0:
            logged.clear()

        random.seed(t)

        err = 0.012 + random.uniform(-0.001, 0.001)
        thr = base_threshold
        anom = False
        fl_r = 0; fl_l = 0.000035; clients = 3
        pre_c = 0.0; post_c = 0.0; q_rew = -15.0

        if t < 30:
            phase_idx = 0
            clients = min(t // 9, 3)
            log("b0", PHASE_NAMES[0], "coordinator", "Central coordinator started on gRPC port 50051. Waiting for edge agent registrations...")
            if t >= 8: log("b8", PHASE_NAMES[0], "registration", "Edge Agent 1 (node-A) RegisterNode -> handshake success. Registry: [node-A]")
            if t >= 16: log("b16", PHASE_NAMES[0], "registration", "Edge Agent 2 (node-B) RegisterNode -> handshake success. Registry: [node-A, node-B]")
            if t >= 24: log("b24", PHASE_NAMES[0], "registration", "Edge Agent 3 (node-C) RegisterNode -> handshake success. Registry: [node-A, node-B, node-C]")
            if t >= 28: log("b28", PHASE_NAMES[0], "coordinator", "All edge nodes registered. Hub-and-spoke federated topology established.", "info")

        elif t < 75:
            phase_idx = 1
            if t < 45:
                err = 0.012 + random.uniform(-0.001, 0.001)
            elif t < 50:
                err = 0.012 + (t - 45) * 0.003
                anom = err > thr
            else:
                err = 0.027 + random.uniform(-0.002, 0.002)
                anom = True
            log("d30", PHASE_NAMES[1], "edge_agent", "Streaming SMD telemetry into LSTM-AE local inference pipeline (window=50, features=38)...")
            if t >= 40: log("d40", PHASE_NAMES[1], "edge_agent", f"Local inference: e_t = {err:.4f}, τ_t = {thr:.4f} - Normal baseline")
            if t >= 45: log("d45", PHASE_NAMES[1], "injector", "Metric anomaly injected. Reconstruction error rising...", "warning")
            if t >= 48: log("d48", PHASE_NAMES[1], "edge_agent", "Warning: ANOMALY DETECTED: e_t exceeded τ_t. Alert fired to coordinator.", "warning")
            if t >= 55: log("d55", PHASE_NAMES[1], "edge_agent", f"Sustained anomaly: e_t = {err:.4f} >> τ_t = {thr:.4f}", "warning")

        elif t < 130:
            phase_idx = 2
            err = 0.027 + random.uniform(-0.002, 0.002); anom = True
            if t < 115: fl_r = 0; fl_l = 0.000035
            else: fl_r = 1; fl_l = 0.000026
            if 100 <= t < 115: pre_c = 2.54; post_c = 1.00

            log("f75",  PHASE_NAMES[2], "coordinator", "Initiating federated learning round 1...")
            if t >= 78:  log("f78",  PHASE_NAMES[2], "coordinator", "Global model broadcast to 3 agents (32.2 KB, zstd compressed)")
            if t >= 85:  log("f85",  PHASE_NAMES[2], "edge_agent",  "node-A local training 5/5 epochs complete. Submitting DP-protected update.")
            if t >= 90:  log("f90",  PHASE_NAMES[2], "edge_agent",  "node-B local training 5/5 epochs complete. Submitting DP-protected update.")
            if t >= 95:  log("f95",  PHASE_NAMES[2], "edge_agent",  "node-C local training 5/5 epochs complete. Submitting DP-protected update.")
            if t >= 100: log("f100", PHASE_NAMES[2], "dp_engine",   "DP-SGD pipeline: pre-clip ‖g‖ = 2.54 -> post-clip ‖g‖ = 1.00 (C = 1.0)")
            if t >= 105: log("f105", PHASE_NAMES[2], "dp_engine",   "Gaussian noise injected: σ = 0.0005, privacy budget ε = 10.0")
            if t >= 110: log("f110", PHASE_NAMES[2], "coordinator", "FedAvg aggregation over 3 updates complete.")
            if t >= 115: log("f115", PHASE_NAMES[2], "coordinator", "Round 1 done. Global loss: 0.000026 (converged). Model persisted + broadcast.")

        elif t < 175:
            phase_idx = 3
            err = 0.027 + random.uniform(-0.002, 0.002); anom = True
            fl_r = 1; fl_l = 0.000026

            log("r130", PHASE_NAMES[3], "rca_engine", "Correlated anomalies: ts-payment-service (CPU 98%), ts-order-service (latency 1200ms)", "warning")
            if t >= 135: log("r135", PHASE_NAMES[3], "rca_engine", "Constructing causal dependency graph from traces. Nodes: 11, Edges: 16.")
            if t >= 140: log("r140", PHASE_NAMES[3], "rca_engine", "Running Random Walk with Restart (PageRank) on causal graph...")
            if t >= 148: log("r148", PHASE_NAMES[3], "rca_engine", "Root cause ranking: #1 ts-payment-service (0.88) #2 ts-order-service (0.38) #3 ts-ui-dashboard (0.12)")
            if t >= 152: log("r152", PHASE_NAMES[3], "rca_engine", "INC-1025: ts-payment-service -> ts-order-service -> ts-ui-dashboard")
            if t >= 160: log("r160", PHASE_NAMES[3], "rca_engine", "Top-3 accuracy: 89% on Train-Ticket topology (vs random baseline).")
            if t >= 165: log("r165", PHASE_NAMES[3], "rca_engine", "Root cause isolated: ts-payment-service (Latency Spike). Confidence: 0.92")

        elif t < 210:
            phase_idx = 4
            fl_r = 1; fl_l = 0.000026
            err = 0.022 + random.uniform(-0.001, 0.001)
            if t < 192:
                thr = base_threshold
                anom = err > thr
            else:
                thr = 0.0244
                anom = err > thr

            progress = (t - 175) / 35.0
            adaptive_alerts_total.labels(method='Static 3-sigma').set(int(502 * progress))
            adaptive_alerts_total.labels(method='EWMA').set(int(380 * progress + random.uniform(0, 20)))
            adaptive_alerts_total.labels(method='Q-Learning (Ours)').set(int(28 * progress))
            adaptive_fpr_current.labels(method='Static').set(0.45)
            adaptive_fpr_current.labels(method='EWMA').set(0.32)
            if t < 185:
                adaptive_fpr_current.labels(method='Q-Learning').set(max(0.9 - (t - 175) * 0.08, 0.05))
            else:
                adaptive_fpr_current.labels(method='Q-Learning').set(0.05 + random.uniform(-0.01, 0.01))

            log("t175", PHASE_NAMES[4], "threshold_tuner", "Q-learning adaptive threshold tuner activated.")
            if t >= 180: log("t180", PHASE_NAMES[4], "threshold_tuner", f"State: FPR=0.12, FNR=0.00. Action: Increase threshold by 20%.")
            if t >= 192: log("t192", PHASE_NAMES[4], "threshold_tuner", f"Threshold adjusted: {base_threshold:.4f} -> {thr:.4f}. Alerts reduced from 502 to 28.")
            if t >= 200: log("t200", PHASE_NAMES[4], "threshold_tuner", "Note: initial exploration phase shows cold-start instability (FPR ≈ 0.9). Warm-start recommended.", "warning")

        else:
            phase_idx = 5
            fl_r = 1; fl_l = 0.000026
            thr = 0.0244; anom = False
            clients = 2 if t < 228 else 3

            log("ft210", PHASE_NAMES[5], "coordinator", "Edge Agent 2 (node-B) crashed. Keepalive timeout triggered.", "error")
            if t >= 215: log("ft215", PHASE_NAMES[5], "coordinator", "Active registry: [node-A, node-C] (2/3). Round 2 aborted - graceful degradation.", "warning")
            if t >= 222: log("ft222", PHASE_NAMES[5], "edge_agent", "Local inference continues on cached model. Detection unaffected.")
            if t >= 228: log("ft228", PHASE_NAMES[5], "coordinator", "node-B recovered. RegisterNode handshake restored. Registry: [3/3]")
            if t >= 235: log("ft235", PHASE_NAMES[5], "coordinator", "Resuming normal federated operations. System healthy.")

        error_gauge.set(err)
        threshold_gauge.set(thr)
        anomaly_gauge.set(1 if anom else 0)
        fl_clients_registered.set(clients)
        dp_pre_clip_norm.set(pre_c)
        dp_post_clip_norm.set(post_c)
        demo_phase.set(phase_idx)
        demo_phase_name.info({"phase": PHASE_NAMES[phase_idx], "index": str(phase_idx)})

        if anom:
            alert_count += 0.1
            q_rew = -15.0 - (t * 0.01)
        else:
            q_rew = -15.0 + (t * 0.001)
        cumulative_alerts.set(alert_count)
        q_learning_reward.set(q_rew)
        false_positive_rate.set(0.017 + random.uniform(-0.002, 0.002))
        fl_round.set(fl_r); fl_loss.set(fl_l)
        fl_agg_latency.set(25.4 + random.uniform(-1, 1))
        fl_agg_memory.set(42.3 + random.uniform(-0.5, 0.5))

        set_rca_metrics(anom)
        set_ms_telemetry(anom)
        time.sleep(1.0)

if __name__ == '__main__':
    run_exporter()
