"""
Benchmark D – Root Cause Analysis Accuracy (Thesis §11.4)
=========================================================
Runs 100 fault-injection scenarios with Train-Ticket-inspired topology
(11 services, 14 edges) plus an Alibaba-inspired large-scale topology
(50 services). Compares framework RCA against a random baseline.

Logs: Top-1, Top-3, MRR per event, per-anomaly-type breakdown, and summary.
"""
import os, sys, random, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import importlib.util

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

def _import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

_logger = _import_from_file('thesis_logger', os.path.join(ROOT, 'tests', 'utils', 'thesis_logger.py'))
ThesisLogger = _logger.ThesisLogger

from analysis.root_cause_analyzer import RootCauseAnalyzer
from tracing.causal_graph import CausalGraph

random.seed(42); np.random.seed(42)

# ── Train-Ticket topology ─────────────────────────────────────────────────
SERVICES = [
    "api-gateway", "auth-service", "ticket-service",
    "order-service", "payment-service", "db-service",
    "notification-service", "user-service", "route-service",
    "station-service", "config-service",
]

BASE_EDGES = [
    ("api-gateway", "auth-service"),
    ("api-gateway", "ticket-service"),
    ("api-gateway", "order-service"),
    ("api-gateway", "route-service"),
    ("ticket-service", "db-service"),
    ("ticket-service", "station-service"),
    ("order-service", "payment-service"),
    ("order-service", "db-service"),
    ("payment-service", "notification-service"),
    ("auth-service", "user-service"),
    ("user-service", "db-service"),
    ("route-service", "station-service"),
    ("station-service", "db-service"),
    ("config-service", "db-service"),
]

SCENARIOS = [
    ("db-service",       {"db-service", "ticket-service", "order-service", "user-service"}),
    ("ticket-service",   {"ticket-service", "db-service"}),
    ("payment-service",  {"payment-service", "notification-service", "order-service"}),
    ("auth-service",     {"auth-service", "user-service", "api-gateway"}),
    ("route-service",    {"route-service", "station-service"}),
    ("api-gateway",      {"api-gateway", "ticket-service", "order-service", "auth-service"}),
    ("station-service",  {"station-service", "db-service", "route-service"}),
    ("config-service",   {"config-service", "db-service"}),
]

# ── Alibaba-inspired large-scale topology (50 services) ───────────────────
ALIBABA_SERVICES = [f"svc-{i:03d}" for i in range(50)]
ALIBABA_EDGES = []
# Create a realistic microservice DAG
for i in range(50):
    n_deps = random.randint(1, 4)
    targets = random.sample(range(max(1, i-10), min(50, i+15)), min(n_deps, 5))
    for t in targets:
        if t != i:
            ALIBABA_EDGES.append((ALIBABA_SERVICES[i], ALIBABA_SERVICES[t]))

ALIBABA_SCENARIOS = [
    (ALIBABA_SERVICES[i], {ALIBABA_SERVICES[i]} |
     {ALIBABA_SERVICES[j] for j in random.sample(range(50), random.randint(2, 6)) if j != i})
    for i in random.sample(range(50), 10)
]


def _build_graph(services, edges, extra_edges=None, removed_edges=None):
    g = CausalGraph()
    for s in services:
        g.add_service(s)
    edge_set = set(edges)
    if extra_edges:
        edge_set |= set(extra_edges)
    if removed_edges:
        edge_set -= set(removed_edges)
    for src, dst in edge_set:
        g.add_dependency(src, dst, call_count=random.randint(50, 500),
                         latency=random.uniform(1, 50))
    return g


def _random_baseline_rank(true_rc, anomalous_set):
    shuffled = list(anomalous_set)
    random.shuffle(shuffled)
    try:
        return shuffled.index(true_rc) + 1
    except ValueError:
        return len(shuffled) + 1


def _run_scenarios(logger, services, edges, scenarios, total, prefix):
    fw_top1 = fw_top3 = fw_mrr = 0
    rand_top1 = rand_top3 = rand_mrr = 0

    for i in range(total):
        true_rc, base_anomalous = scenarios[i % len(scenarios)]
        anomalous = set(base_anomalous)
        if random.random() < 0.3 and len(anomalous) > 1:
            extra = random.choice([s for s in services if s not in anomalous])
            anomalous.add(extra)

        extra_edges = None; removed_edges = None
        if i % 20 == 0 and i > 0:
            extra_edges = [(random.choice(services), random.choice(services))]
        if i % 30 == 0 and i > 0:
            removed_edges = [random.choice(edges)] if edges else None

        graph = _build_graph(services, edges, extra_edges, removed_edges)
        analyzer = RootCauseAnalyzer(causal_graph=graph)
        result = analyzer.analyze(anomalous)
        ranked = [rc['service'] for rc in result.get('root_causes', [])]

        try:
            fw_rank = ranked.index(true_rc) + 1
        except ValueError:
            fw_rank = len(ranked) + 1
        if fw_rank == 1: fw_top1 += 1
        if fw_rank <= 3: fw_top3 += 1
        fw_mrr += 1.0 / fw_rank

        r_rank = _random_baseline_rank(true_rc, anomalous)
        if r_rank == 1: rand_top1 += 1
        if r_rank <= 3: rand_top3 += 1
        rand_mrr += 1.0 / r_rank

        logger.log_metric(f"{prefix}_event_{i}", {
            'event_id': i, 'true_root_cause': true_rc,
            'anomalous_services': list(anomalous),
            'fw_predicted_rank': fw_rank,
            'fw_top_1': int(fw_rank == 1), 'fw_top_3': int(fw_rank <= 3),
            'rand_rank': r_rank,
            'analysis_time_ms': round(result.get('analysis_time', 0) * 1000, 4),
        })

    logger.log_metric(f"{prefix}_summary_framework", {
        'top_1_accuracy': round(fw_top1 / total, 4),
        'top_3_accuracy': round(fw_top3 / total, 4),
        'mrr': round(fw_mrr / total, 4),
    })
    logger.log_metric(f"{prefix}_summary_random", {
        'top_1_accuracy': round(rand_top1 / total, 4),
        'top_3_accuracy': round(rand_top3 / total, 4),
        'mrr': round(rand_mrr / total, 4),
    })
    return fw_top1 / total, fw_top3 / total, fw_mrr / total


def run_rca_benchmark():
    logger = ThesisLogger("benchmark_D_rca_accuracy")

    # ── Train-Ticket topology (100 events) ────────────────────────────
    print("  [D] Train-Ticket RCA (100 events) ...")
    t1, t3, mrr = _run_scenarios(logger, SERVICES, BASE_EDGES, SCENARIOS, 100, "trainticket")
    print(f"      Top-1={t1:.2%}  Top-3={t3:.2%}  MRR={mrr:.4f}")

    # ── Alibaba large-scale topology (50 events) ─────────────────────
    print("  [D] Alibaba-scale RCA (50 events, 50 services) ...")
    t1, t3, mrr = _run_scenarios(logger, ALIBABA_SERVICES, ALIBABA_EDGES,
                                  ALIBABA_SCENARIOS, 50, "alibaba")
    print(f"      Top-1={t1:.2%}  Top-3={t3:.2%}  MRR={mrr:.4f}")

    print("  [D] Complete.")


if __name__ == "__main__":
    run_rca_benchmark()