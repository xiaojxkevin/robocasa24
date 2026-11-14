#!/usr/bin/env bash
# Native RoboCasa pi0 (with or without D4RT), using the shared OpenPI service.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_PROJECT="${PI0_PROJECT:-${PI05_PROJECT:-$(dirname "${repo_root}")/pi05-experiments}}"
if [[ -n "${PI0_PYTHON:-}" ]]; then export PI05_PYTHON="${PI0_PYTHON}"; fi
export PI05_CONFIG="${PI0_CONFIG:-${repo_root}/evaluation/configs/pi0.example.json}"
exec bash "${repo_root}/evaluation/run_pi05_server.sh" "$@"
