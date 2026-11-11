#!/usr/bin/env bash
# Launch explicitly after training; this script is not called by preparation.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project="${PI05_PROJECT:-$(dirname "${repo_root}")/pi05-experiments}"
project="$(cd -- "${project}" && pwd)"
model_python="${PI05_PYTHON:-${project}/.venv/bin/python}"
: "${CHECKPOINT:?Set CHECKPOINT to a completed checkpoint step directory containing params/ and assets/}"
if [[ ! -x "${model_python}" ]]; then
    echo "Model Python missing: ${model_python}" >&2; exit 1
fi
if [[ ! -d "${CHECKPOINT}/params" || ! -d "${CHECKPOINT}/assets" ]]; then
    echo "CHECKPOINT must contain params/ and assets/: ${CHECKPOINT}" >&2; exit 1
fi
checkpoint="$(cd -- "${CHECKPOINT}" && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${repo_root}"
exec "${model_python}" -m evaluation.server \
    --config "${PI05_CONFIG:-${repo_root}/evaluation/configs/pi05.example.json}" \
    --project "${project}" --checkpoint "${checkpoint}" \
    --bind "${BIND:-tcp://127.0.0.1:9100}" "$@"
