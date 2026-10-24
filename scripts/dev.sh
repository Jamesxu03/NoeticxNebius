#!/usr/bin/env bash
# Start the whole app for a demo: TasteNet (optional), backend API, Palatable frontend.
#
#   scripts/dev.sh                 # backend + frontend (menus use the local taste estimate)
#   TASTENET=1 scripts/dev.sh      # also start Arda's TasteNet on :8002 first, so menus get TasteNet scores
#
# Needs: .venv with requirements.txt, frontend/node_modules (npm ci). For TASTENET=1 also
# tastelayer/.venv (pip install -e tastelayer) and `cd tastelayer && make fetch-models` once.
# Ctrl+C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT

if [[ "${TASTENET:-0}" == "1" ]]; then
  (cd tastelayer && .venv/bin/uvicorn tastelayer.api:app --port 8002) &
  echo "waiting for TasteNet on :8002 ..."
  until curl -sf -o /dev/null http://127.0.0.1:8002/health; do sleep 1; done
  export TASTELAYER_URL=http://127.0.0.1:8002  # the backend warms it up at startup
fi

.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
(cd frontend && npm run dev -- --host 127.0.0.1 --port 5173) &
echo "Palatable: http://127.0.0.1:5173   API docs: http://127.0.0.1:8000/docs"
wait
