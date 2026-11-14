#!/usr/bin/env bash
# Same simulator/seeds/tasks as the shared benchmark; workers share one model.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${repo_root}/evaluation/run_client.sh" \
    --endpoint "${ENDPOINT:-tcp://127.0.0.1:9100}" \
    --timeout-ms "${TIMEOUT_MS:-300000}" --workers "${WORKERS:-1}" \
    --replan-steps "${REPLAN_STEPS:-32}" \
    --action-bounds "${ACTION_BOUNDS:-controller}" "$@"
