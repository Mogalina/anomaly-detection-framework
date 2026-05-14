# Federated Anomaly Detection: Demonstration Guide

This document outlines the procedures for initializing, interacting with, and interpreting the live Grafana and Prometheus demonstration environment developed for the thesis defense.

## 1. Overview
The demonstration provides a real-time visualization of the proposed framework, simulating a production-grade Site Reliability Engineering (SRE) environment.

It consists of three components:
1. **The Exporter (`demo/prometheus_exporter.py`)**: A Python engine that actively loads your pre-trained `lstm_best.pt` model, streams the SMD test dataset through it, and exposes real-time telemetry, Federated Learning states, Differential Privacy bounds, and simulated RCA traces.
2. **Prometheus**: A time-series database running in Docker that scrapes the Python exporter every second.
3. **Grafana**: The industry-standard visualization layer running in Docker, pre-provisioned with 4 massive dashboards.

---

## 2. Initialization Procedures

To launch the demonstration environment, execute the following commands within the project root directory:

```bash
# 1. Start the Grafana and Prometheus containers
cd demo
docker-compose up -d

# 2. Activate your python virtual environment
source ../venv/bin/activate

# 3. Start the Live Exporter
python3 prometheus_exporter.py
```

*Note: The exporter will output `Prometheus Exporter running on http://localhost:8000/metrics`. This process must remain active in the background during the demonstration.*

To terminate the demonstration environment:
```bash
# Press Ctrl+C in the exporter terminal to kill the python script, then:
docker-compose down
```

---

## 3. Dashboard Interpretation

Upon successful initialization of the exporter, the Grafana interface can be accessed via:  
**`http://localhost:3000`**

*(Authentication credentials: username `admin`, password `admin`)*.

To access the specific dashboards, navigate to the **Dashboards** menu in the left-hand navigation pane and select **Browse**. Four distinct analytical dashboards are available.

### Dashboard 1: Thesis Command Center
**Description:** The executive overview of the system architecture.
**Interpretation:** 
- The top row shows your KPIs: Current system status (Normal vs Anomaly), active FL round, and your Differential Privacy guarantee ($\varepsilon$). 
- The central graph tracks the live LSTM-AE reconstruction error against your dynamically computed 95th-percentile threshold.
- The bottom heatmap gives you a massive, at-a-glance view of latency across all microservices.

### Dashboard 2: Federated Learning & Privacy
**Description:** A detailed analysis of the DP-FedAvg architecture.
**Interpretation:**
- Observe the **Global Loss Convergence** graph, which simulates the model training trajectory over successive federated rounds.
- Analyze the **Communication Payload Size** bar chart to quantify the bandwidth reduction achieved through binary compression (TorchSave) compared to standard JSON serialization.
- Look at the top stats to point out your specific DP bounds ($C$ and $\sigma$).

### Dashboard 3: Root Cause Analysis Engine (Deep Dive)
**Description:** The incident response and localization monitor.
**Interpretation:**
- Wait for an anomaly to trigger in the Command Center, then quickly switch to this dashboard.
- The **Active Incident Trace Paths** table will populate with the exact Incident ID, the Root Cause Service, the Fault Type (e.g., Latency Spike), and the entire downstream dependency trace (e.g., `ts-order-service -> ts-payment-service -> ts-security-service`).
- The **Probability Gauges** will instantly isolate the highest-scoring culprit service based on your framework's RCA engine.

### Dashboard 4: Microservice Deep Drilldown
**Description:** Independent node-level telemetry displaying RED (Rate, Errors, Duration) and USE (Utilization, Saturation, Errors) metrics.
**Interpretation:**
- This dashboard proves your framework can monitor deep, complex topologies. It tracks CPU, Memory, Latency, Network RX/TX, and Connections.
- **Crucial Interaction:** At the very top-left of this dashboard, there is a dropdown menu labeled `service`. Click it to select specific microservices (e.g., `ts-route-service`). The entire dashboard will instantly filter to show *only* that service's metrics. You can select one, or multiple to compare them side-by-side!

---

## 4. Presentation Strategy

1. **System Overview:** Initiate the presentation on Dashboard 1 to introduce the architecture and demonstrate the active monitoring state.
2. **Anomaly Propagation:** Upon the detection of an anomaly (indicated by a critical state change), transition to the **Root Cause Analysis** dashboard to illustrate the automated tracing of the fault across the dependency graph.
3. **Telemetry Analysis:** Navigate to Dashboard 4, utilize the `$service` variable to isolate the identified anomalous service, and analyze its CPU utilization or latency spikes corresponding to the LSTM-AE trigger event.
