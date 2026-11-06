#!/usr/bin/env bash
# Unified Launcher for Berry Good Listener (GUI + Voice Agent)
# Exports Wayland / Xwayland display environment and coordinates both processes.

export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GUI_SCRIPT="${SCRIPT_DIR}/berry_gui.py"
AGENT_SCRIPT="${BASE_DIR}/run_agent.py"
VENV_PYTHON="/home/misael/AI_231_venv/bin/python3"

START_AGENT=true
GUI_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --gui-only)
            START_AGENT=false
            ;;
        *)
            GUI_ARGS+=("$arg")
            ;;
    esac
done

AGENT_PID=""
cleanup() {
    echo ""
    echo "[LAUNCHER]: Shutting down..."
    if [ -n "$AGENT_PID" ] && kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "[LAUNCHER]: Stopping voice agent (PID ${AGENT_PID})..."
        kill -SIGINT "$AGENT_PID" 2>/dev/null
        wait "$AGENT_PID" 2>/dev/null
    fi
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Voice Agent if requested and not already running
if [ "$START_AGENT" = true ]; then
    if pgrep -f "python.*run_agent.py" >/dev/null 2>&1; then
        echo "[LAUNCHER]: Voice agent is already running."
    else
        echo "[LAUNCHER]: Starting Voice Agent in background..."
        cd "${BASE_DIR}"
        "${VENV_PYTHON}" "${AGENT_SCRIPT}" &
        AGENT_PID=$!
        echo "[LAUNCHER]: Voice Agent running with PID ${AGENT_PID}."
        sleep 2
    fi
fi

# 2. Start GUI process (foreground)
echo "[LAUNCHER]: Starting Fullscreen GUI..."
/usr/bin/python3 "${GUI_SCRIPT}" "${GUI_ARGS[@]}"
