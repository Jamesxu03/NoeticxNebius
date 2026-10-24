#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m pip install --no-cache-dir 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-cache-dir -r requirements-inference.txt
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
.venv/bin/python scripts/fetch_tastenet.py
