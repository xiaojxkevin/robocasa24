#!/usr/bin/env bash
# Native RoboCasa Spatial Forcing; runs only when explicitly invoked.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PI05_PROJECT="${SF_PROJECT:-$(dirname "${repo_root}")/spatial-forcing/policy/Spatial_Forcing/open_sf}"
if [[ -n "${SF_PYTHON:-}" ]]; then export PI05_PYTHON="${SF_PYTHON}"; else unset PI05_PYTHON; fi
export PI05_CONFIG="${SF_CONFIG:-${repo_root}/evaluation/configs/spatial_forcing.example.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$(dirname "${repo_root}")/spatial-forcing/outputs/cache/openpi}"
if [[ ! -s "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" ]]; then
    echo "Missing offline tokenizer: ${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model" >&2; exit 1
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec bash "${repo_root}/evaluation/run_pi05_server.sh" "$@"
