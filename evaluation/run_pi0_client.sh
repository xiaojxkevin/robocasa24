#!/usr/bin/env bash
# Same benchmark, action bounds and simulator as pi05.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${repo_root}/evaluation/run_pi05_client.sh" "$@"
