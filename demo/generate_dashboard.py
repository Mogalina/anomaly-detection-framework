"""
Grafana Dashboard Generator
===========================
Generates multiple massive JSON representations of Grafana dashboards
used to visualize the Anomaly Detection and Root Cause Analysis metrics.
Outputs directly to the Grafana provisioning folder.
"""
import json
import os

def create_base_dash(title: str, uid: str) -> dict:
    """
    Creates the base skeleton for a Grafana dashboard.
    
    Args:
        title: The display title of the dashboard.
        uid: A unique string identifier for the dashboard.
        
    Returns:
        A dictionary representing the dashboard base JSON structure.
    """
    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": True,
        "refresh": "1s",
        "schemaVersion": 38,
        "tags": ["thesis", "demo"],
        "time": {"from": "now-5m", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": title,
        "uid": uid,
        "version": 1,
        "panels": []
    }

def build_command_center() -> dict:
    """
    Builds the 'Command Center' dashboard.
    
    Provides high-level executive insights into system health, Differential Privacy
    guarantees, Federated Learning rounds, and global anomaly thresholds.
    
    Returns:
        JSON-serializable dict of the dashboard.
    """
    dash = create_base_dash("1. Thesis Command Center", "dash-command-center")
    dash["panels"] = [
        {"type": "stat", "title": "System Status", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 5, "w": 6, "x": 0, "y": 0}, "id": 1, "options": {"colorMode": "background"}, "fieldConfig": {"defaults": {"mappings": [{"options": {"0": {"color": "green", "text": "NORMAL"}, "1": {"color": "red", "text": "ANOMALY DETECTED"}}, "type": "value"}]}}, "targets": [{"expr": "smd_is_anomaly"}]},
        {"type": "stat", "title": "DP Guarantee (ε)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 5, "w": 6, "x": 6, "y": 0}, "id": 2, "options": {"colorMode": "value"}, "targets": [{"expr": "dp_epsilon"}]},
        {"type": "stat", "title": "Federated Round", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 5, "w": 6, "x": 12, "y": 0}, "id": 3, "targets": [{"expr": "fl_current_round"}]},
        {"type": "stat", "title": "Active Incidents", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 5, "w": 6, "x": 18, "y": 0}, "id": 4, "options": {"colorMode": "value"}, "targets": [{"expr": "cumulative_alerts"}]},
        
        {"type": "timeseries", "title": "Global Anomaly Detection", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 16, "x": 0, "y": 5}, "id": 5, "targets": [{"expr": "smd_reconstruction_error", "legendFormat": "Error"}, {"expr": "smd_anomaly_threshold", "legendFormat": "Threshold"}]},
        {"type": "bargauge", "title": "Top-3 Root Causes", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 8, "x": 16, "y": 5}, "id": 6, "options": {"orientation": "horizontal", "displayMode": "gradient"}, "targets": [{"expr": "topk(3, rca_root_cause_probability)", "legendFormat": "{{service}} - {{fault_type}}"}]},
        
        {"type": "state-timeline", "title": "Microservice Latency Heatmap", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 24, "x": 0, "y": 15}, "id": 7, "options": {"mergeValues": False}, "fieldConfig": {"defaults": {"color": {"mode": "continuous-GrYlRd"}}}, "targets": [{"expr": "ms_latency_ms", "legendFormat": "{{service}}"}]}
    ]
    return dash

def build_fl_dp():
    dash = create_base_dash("2. Federated Learning & Privacy", "dash-fl-dp")
    dash["panels"] = [
        {"type": "stat", "title": "Noise Multiplier (σ)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 4, "w": 8, "x": 0, "y": 0}, "id": 1, "targets": [{"expr": "dp_sigma"}]},
        {"type": "stat", "title": "Clipping Norm (C)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 4, "w": 8, "x": 8, "y": 0}, "id": 2, "targets": [{"expr": "dp_clipping_norm"}]},
        {"type": "stat", "title": "Privacy Budget (ε)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 4, "w": 8, "x": 16, "y": 0}, "id": 3, "targets": [{"expr": "dp_epsilon"}]},
        
        {"type": "timeseries", "title": "Global Loss Convergence", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 0, "y": 4}, "id": 4, "targets": [{"expr": "fl_global_loss", "legendFormat": "MSE"}]},
        {"type": "timeseries", "title": "Aggregation Overhead", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 12, "y": 4}, "id": 5, "targets": [{"expr": "fl_aggregation_latency_ms", "legendFormat": "Latency (ms)"}, {"expr": "fl_coordinator_memory_mb", "legendFormat": "Memory (MB)"}]},
        
        {"type": "barchart", "title": "Payload Size Matrix (KB)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 24, "x": 0, "y": 14}, "id": 6, "options": {"orientation": "horizontal"}, "targets": [{"expr": "fl_payload_size_kb", "legendFormat": "{{format}} ({{compression}})"}]}
    ]
    return dash

def build_rca() -> dict:
    """
    Builds the 'Root Cause Analysis (Deep Dive)' dashboard.
    
    Visualizes detailed RCA tables, tracking exact incident traces, fault
    type probabilities, and cascade propagation networks generated by the
    centralized causal graph analyzer.
    
    Returns:
        JSON-serializable dict of the dashboard.
    """
    dash = create_base_dash("3. Root Cause Analysis Tracing", "dash-rca")
    dash["panels"] = [
        {"type": "table", "title": "Active Incident Trace Paths", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 6, "w": 24, "x": 0, "y": 0}, "id": 1, 
         "targets": [{"expr": "incident_active_trace", "format": "table", "instant": True}],
         "transformations": [{"id": "organize", "options": {"excludeByName": {"Time": True, "Value": True, "__name__": True, "instance": True, "job": True}}}]
        },
        
        {"type": "bargauge", "title": "Fault Type Probabilities", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 0, "y": 6}, "id": 2, "options": {"orientation": "vertical", "displayMode": "gradient"}, "targets": [{"expr": "rca_root_cause_probability", "legendFormat": "{{fault_type}}"}]},
        {"type": "bargauge", "title": "Service Culprit Probabilities", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 12, "y": 6}, "id": 3, "options": {"orientation": "horizontal", "displayMode": "gradient"}, "targets": [{"expr": "rca_root_cause_probability", "legendFormat": "{{service}}"}]},
        
        {"type": "state-timeline", "title": "Trace Propagation (Latency spikes)", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 24, "x": 0, "y": 16}, "id": 4, "targets": [{"expr": "ms_latency_ms", "legendFormat": "{{service}}"}]}
    ]
    return dash

def build_microservice() -> dict:
    """
    Builds the 'Microservice Deep Drilldown' dashboard.
    
    Uses Grafana template variables ($service) to allow dynamic filtering of
    the 11 Train-Ticket microservices, displaying RED (Rate, Error, Duration)
    and USE (Utilization, Saturation, Error) metrics per node.
    
    Returns:
        JSON-serializable dict of the dashboard.
    """
    dash = create_base_dash("4. Microservice Drilldown", "dash-ms")
    # Add template variable
    dash["templating"] = {
        "list": [
            {
                "name": "service",
                "type": "query",
                "datasource": {"uid": "Prometheus"},
                "query": "label_values(ms_latency_ms, service)",
                "refresh": 1,
                "includeAll": True,
                "multi": True,
                "current": {"value": ["$__all"]}
            }
        ]
    }
    
    dash["panels"] = [
        {"type": "timeseries", "title": "Latency (ms) - $service", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 0, "y": 0}, "id": 1, "targets": [{"expr": "ms_latency_ms{service=~\"$service\"}", "legendFormat": "{{service}}"}]},
        {"type": "timeseries", "title": "CPU (%) - $service", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 12, "y": 0}, "id": 2, "targets": [{"expr": "ms_cpu_percent{service=~\"$service\"}", "legendFormat": "{{service}}"}]},
        {"type": "timeseries", "title": "Memory (MB) - $service", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 0, "y": 10}, "id": 3, "targets": [{"expr": "ms_memory_mb{service=~\"$service\"}", "legendFormat": "{{service}}"}]},
        {"type": "timeseries", "title": "Active Connections - $service", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 12, "x": 12, "y": 10}, "id": 4, "targets": [{"expr": "ms_active_connections{service=~\"$service\"}", "legendFormat": "{{service}}"}]},
        {"type": "timeseries", "title": "Network RX/TX (KB/s) - $service", "datasource": {"uid": "Prometheus"}, "gridPos": {"h": 10, "w": 24, "x": 0, "y": 20}, "id": 5, "targets": [{"expr": "ms_network_rx_kbps{service=~\"$service\"}", "legendFormat": "RX {{service}}"}, {"expr": "ms_network_tx_kbps{service=~\"$service\"}", "legendFormat": "TX {{service}}"}]}
    ]
    return dash

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), 'grafana/dashboards')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, '01_command_center.json'), 'w') as f:
        json.dump(build_command_center(), f, indent=2)
    with open(os.path.join(out_dir, '02_federated_privacy.json'), 'w') as f:
        json.dump(build_fl_dp(), f, indent=2)
    with open(os.path.join(out_dir, '03_rca.json'), 'w') as f:
        json.dump(build_rca(), f, indent=2)
    with open(os.path.join(out_dir, '04_microservice.json'), 'w') as f:
        json.dump(build_microservice(), f, indent=2)
        
    print("5 Dashboards generated successfully.")
