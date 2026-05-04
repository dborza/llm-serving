#!/usr/bin/env bash
# =============================================================================
# smoke-test.sh — Quick sanity check against the running API
# =============================================================================
set -euo pipefail

API_URL="${API_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-changeme}"

echo "🔥 Smoke-testing LLM API at ${API_URL}"

# Health check
echo -n "  /health ... "
STATUS=$(curl -sf "${API_URL}/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
[ "$STATUS" = "ok" ] && echo "✓" || { echo "FAIL"; exit 1; }

# List models
echo -n "  /v1/models ... "
MODEL=$(curl -sf -H "X-API-Key: ${API_KEY}" "${API_URL}/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "✓ (${MODEL})"

# Chat completion
echo -n "  /v1/chat/completions ... "
REPLY=$(curl -sf -X POST "${API_URL}/v1/chat/completions" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
    "max_tokens": 20,
    "temperature": 0.1
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:60])")
echo "✓"
echo "  Response: ${REPLY}"

echo ""
echo "✅ All smoke tests passed!"
