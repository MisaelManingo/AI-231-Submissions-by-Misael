#!/usr/bin/env bash
# Desktop-shortcut entry point: runs start_all.sh and keeps the output in ~/.local/state/berry/last_run.log
# (a double-clicked launcher has no terminal, so this is where to look if something goes wrong).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${HOME}/.local/state/berry"
mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}" || exit 1
exec "${SCRIPT_DIR}/start_all.sh" "$@" > "${LOG_DIR}/last_run.log" 2>&1
