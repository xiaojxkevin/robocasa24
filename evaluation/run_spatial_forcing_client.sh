#!/usr/bin/env bash
# Shared seeds, state, cameras, raw actions and controller bounds.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${repo_root}/evaluation/run_pi05_client.sh" "$@"
