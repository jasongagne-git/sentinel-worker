# SENTINEL Worker

Inference node for the SENTINEL distributed experiment cluster. Each worker runs an HTTP service that receives requests from the coordinator and executes LLM inference via a local Ollama instance.

## Architecture

```
  Coordinator                        Worker Node
  +-----------+    HMAC-signed    +----------------+
  | run_dist- | ──── HTTP ──────> | run_worker.py  |
  | ributed.py|    (port 9400)    |   HTTP server  |
  +-----------+                   +-------+--------+
                                          |
                                          v
                                  +-------+--------+
                                  |     Ollama      |
                                  |  (port 11434)   |
                                  +----------------+
```

- **Stateless** -- all experiment state lives on the coordinator; the worker only holds the currently loaded model
- **Authenticated** -- every request is HMAC-SHA256 signed with the shared cluster secret
- **Thermally aware** -- monitors hardware temperature and returns 503 when thresholds are exceeded

## Quick Start

### 1. Install

```bash
./install.sh
```

Handles Python/Ollama setup, system specs detection, shared secret acquisition (paste directly, invite token enrollment, or env var), and generates `cluster.json` + `.env`.

On Jetson hardware, also configures CUDA paths and Ollama model storage.

### 2. Launch

```bash
./launch.sh --node-id mayhem1
```

Or directly:

```bash
python3 run_worker.py --config cluster.json --node-id mayhem1
```

### 3. Monitor

Local dashboard that polls the worker's `/v1/dashboard` endpoint:

```bash
python3 run_worker_monitor.py
python3 run_worker_monitor.py --host 192.168.1.101 --port 9400 --refresh 2.0
```

## CLI Options

### `run_worker.py`

```
--config FILE       Cluster config file (default: cluster.json)
--node-id ID        This worker's node ID (required)
--port PORT         Listen port (default: 9400)
--bind ADDR         Bind address (default: 0.0.0.0)
--agents FILE       Agent definitions JSON for open-mode registration
--log-level LEVEL   DEBUG | INFO | WARNING | ERROR
```

### `run_worker_monitor.py`

```
--host HOST         Worker address (default: localhost)
--port PORT         Worker port (default: 9400)
--refresh SECONDS   Poll interval (default: 3.0)
```

## API Endpoints

All endpoints require the `X-Sentinel-Auth` HMAC signature header.

### Inference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/generate` | Run agent inference (full turn) |
| POST | `/v1/probe` | Lightweight inference (calibration, testing) |

**Request body** (`generate` and `probe`):
```json
{
  "agent_config": { "name": "...", "system_prompt": "...", "model": "..." },
  "visible_messages": [...],
  "turn": 42,
  "agent_id": "uuid",
  "request_id": "uuid",
  "experiment_id": "uuid",
  "max_turns": 500
}
```

**Response:**
```json
{
  "content": "...",
  "inference_ms": 1234,
  "prompt_tokens": 100,
  "completion_tokens": 50,
  "thermal_temp_c": 45.2,
  "model_digest": "sha256:...",
  "full_prompt": [...]
}
```

### Model Management

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/load_model` | Preload model into VRAM (drops page caches on Jetson) |
| POST | `/v1/unload_model` | Release model from VRAM |

### Status

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/health` | Full capabilities, model inventory, inference counts |
| GET | `/v1/status` | Quick check (temperature, active agent, loaded model) |
| GET | `/v1/dashboard` | Rich monitoring data (per-agent stats, thermal, progress) |

## Enrollment

Three ways to set up the shared secret during `install.sh`:

1. **Paste directly** -- cluster admin provides the secret
2. **Invite token** -- `POST /v1/enroll` to coordinator with a time-limited token, receive the secret back
3. **Environment variable** -- read from file or pre-set env var

The secret is stored in `.env` (mode 600) and referenced by `cluster.json` via `shared_secret_env`.

## Thermal Management

The worker uses `ThermalGuard` (from sentinel-common) to monitor CPU/GPU temperature continuously. When the critical threshold is exceeded:

- Inference requests return **503 Service Unavailable**
- Temperature is reported in every inference response (`thermal_temp_c`)
- Throttle events are counted (`pause_count`) and visible in the dashboard

## Systemd Service

For running as a system service with auto-restart:

```bash
sudo cp sentinel-worker.service /etc/systemd/system/
sudo systemctl enable --now sentinel-worker
```

## Dependencies

**Sibling repository** (expected at `../sentinel-common/`):
- [sentinel-common](../sentinel-common) -- auth, cluster config, Ollama client, thermal guard, validation

**Runtime requirements:**
- Python 3.10+
- Ollama running locally
- No pip dependencies (stdlib only + sentinel-common)

## License

Apache 2.0 -- see [LICENSE](LICENSE).
