# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SENTINEL Worker — HTTP service that accepts inference requests from the coordinator.

Runs on each worker node.  Uses stdlib ``http.server`` + ``json``.  Manages
the local Ollama instance, thermal monitoring, and model lifecycle.
"""

import http.server
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from sentinel_common.auth import MAX_BODY_SIZE, sign_request, verify_request, validate_input
from sentinel_common.cluster import NodeConfig, WorkerCapabilities
from sentinel.ollama import OllamaClient
from sentinel.thermal import ThermalGuard

log = logging.getLogger(__name__)

# ── Request schemas ─────────────────────────────────────────────────────────

_GENERATE_SCHEMA = {
    "required": ["agent_config", "visible_messages", "turn", "agent_id"],
    "properties": {
        "request_id": {"type": "string", "maxLength": 128},
        "agent_id": {"type": "string", "maxLength": 128},
        "agent_config": {"type": "object"},
        "visible_messages": {"type": "array", "maxItems": 1000},
        "turn": {"type": "integer", "minimum": 0},
        "experiment_id": {"type": "string", "maxLength": 128},
    },
}

_AGENT_CONFIG_SCHEMA = {
    "required": ["name", "system_prompt", "model"],
    "properties": {
        "name": {"type": "string", "maxLength": 256},
        "system_prompt": {"type": "string", "maxLength": 1048576},  # 1 MB
        "model": {"type": "string", "maxLength": 256},
        "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "max_history": {"type": "integer", "minimum": 1, "maximum": 1000},
        "response_limit": {"type": "integer", "minimum": 1, "maximum": 4096},
    },
}

_LOAD_MODEL_SCHEMA = {
    "required": ["model"],
    "properties": {
        "model": {"type": "string", "maxLength": 256},
    },
}


# ── Prompt building (mirrors Agent.build_prompt) ────────────────────────────

def _build_prompt(
    agent_id: str,
    agent_config: dict,
    visible_messages: list[dict],
) -> list[dict]:
    """Build Ollama chat messages from agent config and visible history.

    This replicates the logic of ``Agent.build_prompt`` so that workers
    produce identical results to local single-node execution.
    """
    system_prompt = agent_config["system_prompt"]
    max_history = agent_config.get("max_history", 50)

    messages = [{"role": "system", "content": system_prompt}]

    history = visible_messages[-max_history:]

    for msg in history:
        if msg.get("agent_id") == agent_id:
            messages.append({"role": "assistant", "content": msg["content"]})
        else:
            agent_name = msg.get("agent_name", "Unknown")
            messages.append({
                "role": "user",
                "content": f"[{agent_name}]: {msg['content']}",
            })

    if not history:
        messages.append({
            "role": "user",
            "content": (
                "The conversation is starting. "
                "Please introduce yourself and share your perspective."
            ),
        })

    return messages


# ── Worker service ──────────────────────────────────────────────────────────

class WorkerService:
    """HTTP service that accepts inference requests from the coordinator.

    Manages a local Ollama client, thermal monitor, and model lifecycle.
    """

    def __init__(
        self,
        node_config: NodeConfig,
        shared_secret: str,
        tls_context=None,
    ):
        self.node_config = node_config
        self.shared_secret = shared_secret
        self.tls_context = tls_context
        self.client = OllamaClient(node_config.ollama_url)
        self.thermal = ThermalGuard()
        self.loaded_model: Optional[str] = None
        self.start_time: float = time.monotonic()
        self.total_inferences: int = 0
        self.total_probes: int = 0
        self._active_inference: bool = False
        self._server: Optional[http.server.HTTPServer] = None

    def get_capabilities(self) -> WorkerCapabilities:
        """Collect current hardware capabilities and model inventory."""
        from sentinel.models import get_system_specs

        specs = get_system_specs()

        models = []
        try:
            models = [m["name"] for m in self.client.list_models()]
        except Exception:
            log.warning("Could not list Ollama models")

        thermal = self.thermal.to_dict()

        return WorkerCapabilities(
            node_id=self.node_config.node_id,
            total_ram_bytes=specs.total_ram_bytes,
            available_ram_bytes=specs.available_ram_bytes,
            gpu_name=specs.gpu_name,
            is_jetson=specs.is_jetson,
            is_unified_memory=specs.is_unified_memory,
            max_single_alloc_bytes=specs.max_single_alloc_bytes,
            max_model_vram_bytes=specs.max_model_vram_bytes,
            platform="jetson" if specs.is_jetson else "generic",
            ollama_available=self.client.is_available(),
            models_available=models,
            thermal_temp_c=thermal.get("current_temp_c"),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def start(self):
        """Start the HTTP server (blocking)."""
        handler_class = _make_handler(self)

        bind = self.node_config.bind_interface
        port = self.node_config.port

        self._server = http.server.HTTPServer((bind, port), handler_class)

        if self.tls_context:
            self._server.socket = self.tls_context.wrap_socket(
                self._server.socket, server_side=True,
            )

        log.info(
            "Worker %s listening on %s:%d",
            self.node_config.node_id, bind, port,
        )
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            log.info("Worker shutting down (SIGINT)")
        finally:
            self._server.server_close()

    def stop(self):
        """Graceful shutdown."""
        if self._server:
            self._server.shutdown()


def _make_handler(service: WorkerService):
    """Create a request handler class bound to a WorkerService instance.

    This closure approach avoids global state while working with the stdlib
    ``http.server`` handler pattern.
    """

    class WorkerRequestHandler(http.server.BaseHTTPRequestHandler):
        """Handles HTTP requests from the SENTINEL coordinator."""

        # Suppress default stderr logging — we use the logging module.
        def log_message(self, fmt, *args):
            log.debug("HTTP: %s", fmt % args)

        # ── Auth ────────────────────────────────────────────────────────

        def _authenticate(self, body: bytes) -> bool:
            auth = self.headers.get("X-Sentinel-Auth", "")
            if not auth:
                self._send_error(401, "missing X-Sentinel-Auth header")
                return False
            if not verify_request(auth, body, service.shared_secret):
                self._send_error(401, "authentication failed")
                return False
            return True

        # ── Helpers ─────────────────────────────────────────────────────

        def _read_body(self) -> Optional[bytes]:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_SIZE:
                self._send_error(400, f"body too large ({length} bytes)")
                return None
            return self.rfile.read(length) if length else b""

        def _parse_json(self, body: bytes) -> Optional[dict]:
            try:
                return json.loads(body) if body else {}
            except json.JSONDecodeError as exc:
                self._send_error(400, f"invalid JSON: {exc}")
                return None

        def _send_json(self, status: int, data: dict):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_error(self, status: int, message: str, details=None):
            data = {"error": message}
            if details:
                data["details"] = details
            self._send_json(status, data)

        # ── Routing ─────────────────────────────────────────────────────

        def do_POST(self):
            body = self._read_body()
            if body is None:
                return
            if not self._authenticate(body):
                return
            data = self._parse_json(body)
            if data is None:
                return

            path = self.path.rstrip("/")
            handlers = {
                "/v1/generate": self._handle_generate,
                "/v1/probe": self._handle_probe,
                "/v1/load_model": self._handle_load_model,
                "/v1/unload_model": self._handle_unload_model,
            }
            handler = handlers.get(path)
            if handler:
                try:
                    handler(data)
                except Exception:
                    log.exception("Unhandled error in %s", path)
                    self._send_error(500, "internal server error")
            else:
                self._send_error(404, f"unknown endpoint: {path}")

        def do_GET(self):
            # GET requests have no body; authenticate with empty body.
            if not self._authenticate(b""):
                return

            path = self.path.rstrip("/")
            if path == "/v1/health":
                self._handle_health()
            elif path == "/v1/status":
                self._handle_status()
            else:
                self._send_error(404, f"unknown endpoint: {path}")

        # ── POST /v1/generate ───────────────────────────────────────────

        def _handle_generate(self, data: dict):
            self._do_inference(data, is_probe=False)

        # ── POST /v1/probe ──────────────────────────────────────────────

        def _handle_probe(self, data: dict):
            self._do_inference(data, is_probe=True)

        def _do_inference(self, data: dict, is_probe: bool):
            # Validate top-level request
            errors = validate_input(data, _GENERATE_SCHEMA)
            if errors:
                self._send_error(400, "validation error", errors)
                return

            # Validate agent_config
            agent_config = data["agent_config"]
            errors = validate_input(agent_config, _AGENT_CONFIG_SCHEMA)
            if errors:
                self._send_error(400, "agent_config validation error", errors)
                return

            # Thermal check
            thermal = service.thermal.to_dict()
            temp = thermal.get("current_temp_c")
            if temp is not None and temp >= service.thermal.crit_c:
                self._send_json(503, {
                    "request_id": data.get("request_id", ""),
                    "error": "thermal_throttle",
                    "message": (
                        f"Node temperature {temp:.1f}°C exceeds "
                        f"critical threshold ({service.thermal.crit_c}°C)"
                    ),
                    "retry_after_s": 60,
                })
                return

            # Build prompt (same logic as Agent.build_prompt)
            agent_id = data["agent_id"]
            visible_messages = data["visible_messages"]
            prompt_messages = _build_prompt(agent_id, agent_config, visible_messages)

            # Run inference
            model = agent_config["model"]
            temperature = agent_config.get("temperature", 0.7)
            response_limit = agent_config.get("response_limit", 256)

            service._active_inference = True
            try:
                result = service.client.chat(
                    model=model,
                    messages=prompt_messages,
                    temperature=temperature,
                    num_predict=response_limit,
                )
            except Exception as exc:
                service._active_inference = False
                log.error("Inference failed: %s", exc)
                self._send_error(503, f"ollama error: {exc}")
                return
            finally:
                service._active_inference = False

            if is_probe:
                service.total_probes += 1
            else:
                service.total_inferences += 1

            # Get model digest
            model_digest = service.client.get_model_digest(model)

            # Refresh thermal after inference
            thermal_after = service.thermal.to_dict()

            request_id = data.get("request_id", str(uuid.uuid4()))
            self._send_json(200, {
                "request_id": request_id,
                "content": result["content"],
                "full_prompt": json.dumps(prompt_messages),
                "model_digest": model_digest,
                "inference_ms": result["inference_ms"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "thermal_temp_c": thermal_after.get("current_temp_c"),
                "node_id": service.node_config.node_id,
            })

        # ── POST /v1/load_model ─────────────────────────────────────────

        def _handle_load_model(self, data: dict):
            errors = validate_input(data, _LOAD_MODEL_SCHEMA)
            if errors:
                self._send_error(400, "validation error", errors)
                return

            model = data["model"]
            log.info("Loading model: %s", model)

            start = time.monotonic()
            try:
                # Warm up the model with a trivial inference.
                service.client.chat(
                    model=model,
                    messages=[{"role": "user", "content": "hello"}],
                    temperature=0.0,
                    num_predict=1,
                )
                service.loaded_model = model
            except Exception as exc:
                log.error("Failed to load model %s: %s", model, exc)
                self._send_error(503, f"failed to load model: {exc}")
                return

            load_ms = int((time.monotonic() - start) * 1000)
            digest = service.client.get_model_digest(model)

            self._send_json(200, {
                "status": "loaded",
                "model": model,
                "model_digest": digest,
                "load_ms": load_ms,
            })

        # ── POST /v1/unload_model ───────────────────────────────────────

        def _handle_unload_model(self, data: dict):
            model = data.get("model") or service.loaded_model
            if model:
                log.info("Unloading model: %s", model)
                try:
                    service.client._request(
                        "/api/generate",
                        {"model": model, "keep_alive": 0},
                    )
                except Exception as exc:
                    log.warning("Unload failed for %s: %s", model, exc)
                service.loaded_model = None

            self._send_json(200, {"status": "unloaded"})

        # ── GET /v1/health ──────────────────────────────────────────────

        def _handle_health(self):
            caps = service.get_capabilities()
            response = caps.to_dict()
            response["loaded_model"] = service.loaded_model
            response["uptime_s"] = int(time.monotonic() - service.start_time)
            response["total_inferences"] = service.total_inferences
            response["total_probes"] = service.total_probes
            response["status"] = "healthy"
            self._send_json(200, response)

        # ── GET /v1/status ──────────────────────────────────────────────

        def _handle_status(self):
            thermal = service.thermal.to_dict()
            self._send_json(200, {
                "node_id": service.node_config.node_id,
                "status": "healthy",
                "loaded_model": service.loaded_model,
                "thermal_temp_c": thermal.get("current_temp_c"),
                "active_inference": service._active_inference,
                "total_inferences": service.total_inferences,
                "total_probes": service.total_probes,
            })

    return WorkerRequestHandler
