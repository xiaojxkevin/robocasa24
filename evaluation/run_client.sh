#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/evaluation/egl_env.sh"
cd "$repo_root"
exec "$repo_root/.venv/bin/python" -m evaluation.client "$@"
