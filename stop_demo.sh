#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEMO_DIR="$SCRIPT_DIR/demo"
PID_FILE="$DEMO_DIR/exporter.pid"

RST="\033[0m"; CY="\033[36m"; GR="\033[32m"; YL="\033[33m"; RD="\033[31m"; MG="\033[35m"
info()   { echo -e "${CY}[INFO]${RST} $1"; }
ok()     { echo -e "${GR}[OK]${RST} $1"; }
warn()   { echo -e "${YL}[WARN]${RST} $1"; }
header() { echo -e "\n${MG}══════ $1 ══════${RST}"; }

is_port_in_use() { timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$1" 2>/dev/null; }

header "Stopping Thesis Defense Demo"

info "Stopping exporter..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"; sleep 2
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
        ok "Exporter stopped."
    else
        info "PID $PID not running."
    fi
    rm -f "$PID_FILE"
fi

PIDS=$(pgrep -f "demo/prometheus_exporter.py" || true)
for p in $PIDS; do
    kill "$p" 2>/dev/null || true; sleep 1
    kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true
    ok "Killed leftover process $p."
done

DC=""
docker compose version >/dev/null 2>&1 && DC="docker compose" || {
    docker-compose --version >/dev/null 2>&1 && DC="docker-compose" || true
}
if [ -n "$DC" ]; then
    header "Stopping Docker Services"
    $DC -f "$DEMO_DIR/docker-compose.yml" down && ok "Containers removed." || warn "Docker teardown issue."
fi

header "Port Verification"
ALL_CLEAR=1
for p in 3000 3100 9090 8000; do
    is_port_in_use "$p" && { warn "Port $p still occupied."; ALL_CLEAR=0; } || true
done
[ "$ALL_CLEAR" -eq 1 ] && ok "All ports released. Demo destroyed gracefully." || warn "Some ports still in use."
