from __future__ import annotations

import hmac
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vllm import LLM, SamplingParams


APP_NAME = "Assistant Medical LLM"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8010
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = """Tu es un assistant medical virtuel professionnel.
Tu aides les patients a:
- Comprendre leurs symptomes
- Identifier les urgences
- Donner des conseils de premiers soins
- Informer sur les medicaments

Regles:
- Reponds en francais
- Pour tout symptome grave, recommande d'appeler les urgences immediatement
- Rappelle que tu ne remplaces pas un medecin ni le pharmacien
- Reste concis et prudent
"""

_LLM: LLM | None = None


def get_model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM(model=get_model_name())
    return _LLM


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def text_value(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_sampling_params(payload: dict[str, Any]) -> SamplingParams:
    default_temperature = env_float("LLM_TEMPERATURE", 0.3)
    default_top_p = env_float("LLM_TOP_P", 0.9)
    default_max_tokens = env_int("LLM_MAX_TOKENS", 300)

    return SamplingParams(
        temperature=payload_float(payload, "temperature", default_temperature),
        top_p=payload_float(payload, "top_p", default_top_p),
        max_tokens=payload_int(payload, "max_tokens", default_max_tokens),
    )


def build_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if isinstance(messages, list):
        validated = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                validated.append({"role": role, "content": content.strip()})
        if validated:
            return validated

    system_prompt = text_value(payload, "system_prompt") or SYSTEM_PROMPT
    user_prompt = text_value(payload, "user_context", "prompt", "message")
    if not user_prompt:
        return []

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(payload: dict[str, Any]) -> str:
    messages = build_messages(payload)
    if not messages:
        raise ValueError("Champ requis manquant: prompt, user_context ou messages.")

    outputs = get_llm().chat(messages, build_sampling_params(payload))
    return outputs[0].outputs[0].text.strip()


def expected_api_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip()


def is_authorized(header_value: str | None) -> bool:
    token = expected_api_key()
    if not token:
        return True

    expected_header = f"Bearer {token}"
    return hmac.compare_digest(header_value or "", expected_header)


class LLMRequestHandler(BaseHTTPRequestHandler):
    server_version = "AssistantMedicalLLM/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self.send_json(
                {
                    "status": "ok",
                    "service": APP_NAME,
                    "model": get_model_name(),
                    "endpoints": ["/chat", "/api/chat"],
                }
            )
            return

        self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/api/chat"):
            self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if not is_authorized(self.headers.get("Authorization")):
            self.send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
            return

        try:
            payload = self.read_json_payload()
            answer = generate_answer(payload)
        except json.JSONDecodeError:
            self.send_json({"error": "JSON invalide."}, status=HTTPStatus.BAD_REQUEST)
            return
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # pragma: no cover - depends on vLLM runtime
            self.send_json(
                {"error": f"Erreur generation LLM: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json(
            {
                "ok": True,
                "answer": answer,
                "model": get_model_name(),
            }
        )

    def read_json_payload(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Le corps JSON doit etre un objet.")
        return payload

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), LLMRequestHandler)
    print(f"{APP_NAME} disponible sur http://{host}:{port}")
    print("Endpoint chat: POST /chat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du serveur LLM.")
    finally:
        server.server_close()


def main() -> None:
    host = os.getenv("HOST", DEFAULT_HOST)
    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    run_server(host=host, port=port)


if __name__ == "__main__":
    main()
