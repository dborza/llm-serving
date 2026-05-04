#!/usr/bin/env bash
# =============================================================================
# setup.sh — Validate local model and create .env
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

command -v docker >/dev/null 2>&1 || error "Docker not found."

# ── Get local model path ──────────────────────────────────────────────────────
if [ -z "${LOCAL_MODEL_PATH:-}" ]; then
  read -rp "Enter the absolute path to your local Llama 3 model directory: " LOCAL_MODEL_PATH
fi

[ -d "$LOCAL_MODEL_PATH" ] || error "Directory not found: $LOCAL_MODEL_PATH"

# Validate it looks like a HuggingFace model dir
[ -f "$LOCAL_MODEL_PATH/config.json" ]     || error "config.json not found in $LOCAL_MODEL_PATH — is this a valid HuggingFace model directory?"
[ -f "$LOCAL_MODEL_PATH/tokenizer.json" ]  || warn  "tokenizer.json not found — double-check your model directory."

# Check for weights (safetensors preferred, pytorch fallback)
if ls "$LOCAL_MODEL_PATH"/*.safetensors &>/dev/null; then
  info "Found safetensors weights ✓"
elif ls "$LOCAL_MODEL_PATH"/*.bin &>/dev/null; then
  warn "Found .bin weights. Safetensors preferred but .bin will work."
else
  error "No model weights (.safetensors or .bin) found in $LOCAL_MODEL_PATH"
fi

# ── Create .env ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  info "Creating .env..."
  cp .env.example .env
  sed -i "s|/path/to/your/llama3/model|${LOCAL_MODEL_PATH}|" .env
  info "Set LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH} in .env"
  warn "Remember to set a strong API_KEY in .env before deploying!"
else
  info ".env already exists — skipping creation. Make sure LOCAL_MODEL_PATH is set correctly."
fi

info ""
info "Setup complete! Start the stack with:"
info "  docker-compose up -d"
info ""
info "Then run the smoke test:"
info "  ./scripts/smoke-test.sh"
