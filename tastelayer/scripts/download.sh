#!/usr/bin/env bash
# Download Kaggle datasets into data/raw/<name>/ (idempotent). Failures are logged, not fatal.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
LOG="$ROOT/reports/download_log.md"
KAGGLE="$ROOT/.venv/bin/kaggle"
mkdir -p "$RAW" "$ROOT/reports"

# Export only the Kaggle token from .env (never echoed).
KAGGLE_API_TOKEN="$(grep -E '^KAGGLE_API_TOKEN=' "$ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'')"
export KAGGLE_API_TOKEN

echo "# Download log ($(date -u +%FT%TZ))" > "$LOG"

dl_dataset () {  # $1 = owner/slug, $2 = local dir name
  local dest="$RAW/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "- $1: already present" >> "$LOG"; return 0
  fi
  mkdir -p "$dest"
  if "$KAGGLE" datasets download -d "$1" -p "$dest" --unzip -q >/dev/null 2>"$dest.err"; then
    echo "- $1: OK" >> "$LOG"
  else
    echo "- $1: FAILED ($(head -c 200 "$dest.err" | tr '\n' ' '))" >> "$LOG"
    rmdir "$dest" 2>/dev/null; return 1
  fi
  rm -f "$dest.err"
}

dl_dataset nehaprabhavalkar/indian-food-101 indian_food_101
dl_dataset kanishk307/6000-indian-food-recipes-dataset archana \
  || dl_dataset sooryaprakash12/cleaned-indian-recipes-dataset archana
dl_dataset shuyangli94/food-com-recipes-and-user-interactions foodcom
dl_dataset ahmedshahriarsakib/uber-eats-usa-restaurants-menus uber_eats

# What's Cooking (competition; needs rules accepted)
WC="$RAW/whats_cooking"
if [ -f "$WC/train.json" ]; then
  echo "- whats-cooking: already present" >> "$LOG"
else
  mkdir -p "$WC"
  if "$KAGGLE" competitions download -c whats-cooking -p "$WC" -q >/dev/null 2>"$WC.err"; then
    (cd "$WC" && for z in *.zip; do unzip -oq "$z" && rm -f "$z"; done; for z in *.zip; do [ -f "$z" ] && unzip -oq "$z"; done) 2>/dev/null
    echo "- whats-cooking: OK" >> "$LOG"; rm -f "$WC.err"
  else
    echo "- whats-cooking: FAILED/skipped ($(head -c 200 "$WC.err" | tr '\n' ' '))" >> "$LOG"
    rmdir "$WC" 2>/dev/null
  fi
fi
cat "$LOG"
