#!/usr/bin/env bash
# Root shortcut to start both Berry Good Listener Voice Agent and Fullscreen GUI
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/gui/start_gui.sh" "$@"

