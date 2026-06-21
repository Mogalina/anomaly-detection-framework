#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEMO_DIR="$SCRIPT_DIR/demo"
VENV_DIR="$SCRIPT_DIR/venv"
PID_FILE="$DEMO_DIR/exporter.pid"
LOG_FILE="$DEMO_DIR/exporter.log"

RST="\033[0m"; CY="\033[36m"; GR="\033[32m"; YL="\033[33m"; RD="\033[31m"; MG="\033[35m"; DK="\033[90m"
info()    { echo -e "${CY}[INFO]${RST} $1"; }
ok()      { echo -e "${GR}[OK]${RST} $1"; }
warn()    { echo -e "${YL}[WARN]${RST} $1"; }
err()     { echo -e "${RD}[ERR]${RST} $1" >&2; }
header()  { echo -e "\n${MG}══════ $1 ══════${RST}"; }

is_port_in_use() { timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$1" 2>/dev/null; }

header "Thesis Defense Demo"

info "Checking prerequisites..."
docker info >/dev/null 2>&1 || { err "Docker not running."; exit 1; }
ok "Docker daemon active."

DC=""
docker compose version >/dev/null 2>&1 && DC="docker compose" || {
    docker-compose --version >/dev/null 2>&1 && DC="docker-compose" || { err "No Docker Compose found."; exit 1; }
}
ok "Compose: $DC"

[ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ] || { err "venv not found at $VENV_DIR"; exit 1; }
ok "Virtual environment verified."

for p in 3000 3100 9090 8000; do
    is_port_in_use "$p" && warn "Port $p already in use."
done

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        warn "Exporter already running (PID $PID). Run ./stop_demo.sh first."
        exit 1
    fi
fi

header "Starting Docker Services"
info "$DC -f $DEMO_DIR/docker-compose.yml up -d"
$DC -f "$DEMO_DIR/docker-compose.yml" up -d
ok "Containers started."

header "Launching Live Exporter Engine"
nohup "$VENV_DIR/bin/python3" -u "$DEMO_DIR/prometheus_exporter.py" > "$LOG_FILE" 2>&1 &
EX_PID=$!
echo "$EX_PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$EX_PID" 2>/dev/null; then
    err "Exporter crashed on start. Check $LOG_FILE:"
    tail -10 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi
ok "Exporter started (PID $EX_PID). Logs: $LOG_FILE"

header "Health Checks"
wait_svc() {
    local name=$1 port=$2 max=$3 n=0
    info "Waiting for $name (port $port)..."
    while [ "$n" -lt "$max" ]; do
        is_port_in_use "$port" && { ok "$name ready."; return 0; }
        sleep 1; n=$((n+1))
    done
    warn "$name not ready after ${max}s."
}
wait_svc "Exporter"   8000 20
wait_svc "Prometheus"  9090 20
wait_svc "Loki"        3100 20
wait_svc "Grafana"     3000 20

header "Demo Ready!"
echo -e "
  ${CY}Grafana:${RST}        http://localhost:3000  ${DK}(admin/admin)${RST}
  ${CY}Prometheus:${RST}     http://localhost:9090
  ${CY}Loki:${RST}           http://localhost:3100
  ${CY}Exporter:${RST}       http://localhost:8000/metrics

  ${DK}Home dashboard shows the presentation flow.${RST}
  ${DK}Use the top-right links to navigate to detail dashboards.${RST}

  ${CY}Follow logs:${RST}    tail -f $LOG_FILE
  ${CY}Stop:${RST}           ./stop_demo.sh
"
