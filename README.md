# 🦙 Triton LLM Server — Llama 3 on Docker

Serve **Meta Llama 3 8B Instruct** (from a **local model directory**) via NVIDIA Triton Inference Server with a clean **OpenAI-compatible FastAPI gateway**, all wired up with a GitHub Actions CI/CD pipeline.

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                   │
│          (OpenAI-compatible REST client)             │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP :8080
┌──────────────────────▼──────────────────────────────┐
│              FastAPI Gateway (client-api)            │
│  • Auth (X-API-Key)                                  │
│  • Llama 3 prompt formatting                         │
│  • Streaming support                                 │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP :8000 / gRPC :8001
┌──────────────────────▼──────────────────────────────┐
│         Triton Inference Server + vLLM backend       │
│  • Loads weights from /local_model (host mount)      │
│  • GPU-accelerated inference                         │
│  • Continuous batching via vLLM                      │
│  • Metrics on :8002                                  │
└──────────────────────┬──────────────────────────────┘
                       │  bind mount (read-only)
            ┌──────────▼──────────┐
            │  Host filesystem    │
            │  /your/local/model  │
            └─────────────────────┘
```

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker + Docker Compose | 24+ |
| NVIDIA GPU | Ampere (A10, A100, 3090, 4090…) or newer recommended |
| NVIDIA Driver | 525+ |
| nvidia-container-toolkit | Latest |
| Local Llama 3 weights | Downloaded in HuggingFace format |

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/YOUR_ORG/triton-llm-server.git
cd triton-llm-server
cp .env.example .env
```

Edit `.env` and set:
```bash
LOCAL_MODEL_PATH=/absolute/path/to/Meta-Llama-3-8B-Instruct
API_KEY=your-secret-key
```

Your model directory must contain at minimum:
```
Meta-Llama-3-8B-Instruct/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  model-00001-of-00004.safetensors   # (or pytorch .bin files)
  ...
```

> **Don't have the weights yet?** Run `./scripts/setup.sh` — it will prompt you for a path and validate the directory. To download from HuggingFace first: `huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /your/path`

### 2. Validate your setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This checks the model directory looks correct and writes your `.env`.

### 3. Start the stack

```bash
docker-compose up -d
```

Triton takes **2–5 minutes** to load the model. Watch progress:

```bash
docker logs -f triton-llm
```

Wait until you see: `Started GRPCInferenceService at 0.0.0.0:8001`

### 4. Smoke test

```bash
chmod +x scripts/smoke-test.sh
./scripts/smoke-test.sh
```

### 5. Make your first request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what Triton Inference Server does in 2 sentences."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

## How local model loading works

The key pieces:

**`docker-compose.yml`** — bind-mounts your host model directory into the container as read-only:
```yaml
volumes:
  - ${LOCAL_MODEL_PATH}:/local_model:ro
```

**`model_repository/llama3/1/model.json`** — tells vLLM to load from that in-container path:
```json
{ "model": "/local_model" }
```

No HuggingFace token required at runtime. The model never leaves your machine.

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible endpoint.

**Headers**
- `X-API-Key: <your-key>` (required)
- `Content-Type: application/json`

**Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | required | Chat history (role + content) |
| `max_tokens` | int | 512 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0–2) |
| `top_p` | float | 0.95 | Nucleus sampling |
| `stream` | bool | false | Enable SSE streaming |

**Streaming example (Python)**

```python
import httpx

with httpx.stream("POST", "http://localhost:8080/v1/chat/completions",
    headers={"X-API-Key": "your-secret-key"},
    json={
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "stream": True,
        "max_tokens": 300,
    }
) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            import json
            chunk = json.loads(line[6:])
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

### Other endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe (no auth) |
| `GET /v1/models` | List available models |
| `GET :8002/metrics` | Prometheus metrics (Triton) |

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push:

```
push/PR → Lint & Test → Validate Triton Config
                      ↓
              Build & Push to GHCR  →  Security Scan (Trivy)
                      ↓
              Release Summary (on release tag)
```

No secrets needed beyond the auto-provided `GITHUB_TOKEN` for GHCR.

## Project Structure

```
triton-llm-server/
├── model_repository/
│   └── llama3/
│       ├── config.pbtxt          # Triton model config (vLLM backend)
│       └── 1/
│           └── model.json        # vLLM engine params — points to /local_model
├── client-api/
│   ├── main.py                   # FastAPI gateway (OpenAI-compatible)
│   ├── requirements.txt
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions pipeline
├── scripts/
│   ├── setup.sh                  # Validate model dir + create .env
│   └── smoke-test.sh             # Post-deploy sanity check
├── docker-compose.yml            # LOCAL_MODEL_PATH bind-mount
├── .env.example
└── README.md
```

## Tuning

### GPU memory

Edit `model_repository/llama3/1/model.json`:

```json
{
  "gpu_memory_utilization": 0.85,   // increase for more throughput
  "max_model_len": 8192             // reduce for lower VRAM usage
}
```

### Multi-GPU

```json
{ "tensor_parallel_size": 2 }
```

### Using a different local model

Point `LOCAL_MODEL_PATH` at any vLLM-compatible model directory (Mistral, Phi-3, Gemma, etc.) — no other changes needed.

## Troubleshooting

**Triton fails to start / OOM**
- Reduce `gpu_memory_utilization` to `0.7` in `model.json`
- Reduce `max_model_len` to `4096`

**"No such file or directory: /local_model/config.json"**
- `LOCAL_MODEL_PATH` in `.env` is wrong or the directory is missing `config.json`
- Run `./scripts/setup.sh` to validate

**"Model not ready" errors from the API**
- Triton is still loading — the healthcheck retries for up to 5 minutes

## License

Apache 2.0. Model weights are subject to [Meta's Llama 3 Community License](https://llama.meta.com/llama3/license/).
