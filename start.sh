#!/usr/bin/env bash
# MindSense — start FastAPI backend + React frontend

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Colours ────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
DIM='\033[2m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}  MindSense — Counselor Support Tool${RESET}"
echo -e "${DIM}  ─────────────────────────────────────${RESET}"
echo ""

# ── Python virtual environment ─────────────────────────────────────────────
if [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
  echo -e "${DIM}  venv activated${RESET}"
fi

# ── Check model exists ─────────────────────────────────────────────────────
if [ ! -d "$ROOT/models/saved" ]; then
  echo -e "${YELLOW}  ⚠  No trained model found at models/saved/${RESET}"
  echo -e "${DIM}     Run: python -m models.train --output-dir models/saved --epochs 5${RESET}"
  echo ""
fi

# ── Install backend deps if missing ───────────────────────────────────────
python -c "import fastapi, uvicorn" 2>/dev/null || {
  echo -e "${DIM}  Installing backend dependencies…${RESET}"
  pip install fastapi "uvicorn[standard]" python-multipart --quiet
}

# ── Install frontend deps if missing ──────────────────────────────────────
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo -e "${DIM}  Installing frontend dependencies…${RESET}"
  (cd "$ROOT/frontend" && npm install --silent)
fi

echo ""
echo -e "${GREEN}  ▶  Backend${RESET}   http://127.0.0.1:8000"
echo -e "${GREEN}  ▶  Frontend${RESET}  http://localhost:5173"
echo ""
echo -e "${DIM}  Press Ctrl+C to stop both servers.${RESET}"
echo ""

# ── Start servers ──────────────────────────────────────────────────────────
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload \
  --log-level warning &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev -- --port 5173) &
FRONTEND_PID=$!

# ── Graceful shutdown ──────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo -e "${DIM}  Stopping servers…${RESET}"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  echo -e "${DIM}  Done.${RESET}"
}
trap cleanup EXIT INT TERM

wait
