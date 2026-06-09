#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure src/ is on PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Determine python binary
if [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

SUITE="${1:-all}"
PYTEST_ARGS="-v --tb=short --color=yes"

header() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════════"
}

run_unit() {
    header "UNIT TESTS (per-module)"
    $PYTHON_BIN -m pytest tests/unit/ $PYTEST_ARGS "$@"
}

run_integration() {
    header "INTEGRATION TESTS"
    $PYTHON_BIN -m pytest tests/integration/ $PYTEST_ARGS "$@"
}

run_fault() {
    header "FAULT TOLERANCE TESTS"
    $PYTHON_BIN -m pytest tests/test_fault_tolerance.py $PYTEST_ARGS "$@"
}

run_performance() {
    header "PERFORMANCE TESTS"
    $PYTHON_BIN -m pytest tests/test_performance.py $PYTEST_ARGS "$@"
}

run_privacy() {
    header "PRIVACY-UTILITY TESTS"
    $PYTHON_BIN -m pytest tests/test_privacy.py $PYTEST_ARGS "$@"
}

run_e2e() {
    header "END-TO-END TESTS"
    $PYTHON_BIN -m pytest tests/test_e2e.py $PYTEST_ARGS "$@"
}

case "$SUITE" in
    unit)
        run_unit "${@:2}"
        ;;
    integration)
        run_integration "${@:2}"
        ;;
    fault)
        run_fault "${@:2}"
        ;;
    performance)
        run_performance "${@:2}"
        ;;
    privacy)
        run_privacy "${@:2}"
        ;;
    e2e)
        run_e2e "${@:2}"
        ;;
    quick)
        run_unit "${@:2}"
        run_integration "${@:2}"
        ;;
    all)
        run_unit "${@:2}"
        run_integration "${@:2}"
        run_fault "${@:2}"
        run_performance "${@:2}"
        run_privacy "${@:2}"
        run_e2e "${@:2}"
        header "ALL TESTS COMPLETE"
        ;;
    *)
        echo "Usage: $0 {all|unit|integration|fault|performance|privacy|e2e|quick}"
        exit 1
        ;;
esac
