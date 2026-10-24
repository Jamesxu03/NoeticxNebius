#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec .venv/bin/python -m uvicorn backend.palatable:app --host 0.0.0.0 --port "${PORT:-8080}"
