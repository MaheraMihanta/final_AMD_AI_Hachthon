"""from __future__ import annotations

from dataclasses import dataclass

from app import Article


@dataclass(frozen=True)
class PharmacistContext:
    system_prompt: str
    user_context: str

    def as_text(self) -> str:
        return (
            "SYSTEM PROMPT\n"
            f"{self.system_prompt}\n\n"
            "CONTEXTE UTILISATEUR\n"
            f"{self.user_context}"
        )


def build_pharmacist_context(
    symptoms: str, inventory: list[Article]
) -> PharmacistContext:
    available_articles = [
        article
        for article in inventory
        if article.quantite_stock > 0
    ]
    stock_lines = "\n".join(
        (
            f"- {article.nom} ({article.id_unique}), "
            f"{article.classe_therapeutique}, stock {article.quantite_stock}"
        )
        for article in available_articles
    )

    system_prompt = (
        "Tu es un assistant pour pharmacien. Tu aides a structurer un "
        "echange avec un client, mais tu ne poses pas de diagnostic et tu ne "
        "remplaces pas la validation du pharmacien. En cas de symptomes "
        "graves, persistants, grossesse, enfant tres jeune, allergie connue "
        "ou interaction possible, recommande une evaluation medicale ou la "
        "validation directe du pharmacien."
    )
    user_context = (
        f"Symptomes decrits par le client:\n{symptoms.strip() or '[non renseigne]'}\n\n"
        "Medicaments actuellement disponibles dans le stock local:\n"
        f"{stock_lines or '- Aucun medicament disponible'}\n\n"
        "Tache attendue: proposer des questions de clarification et preparer "
        "un resume pour le pharmacien. Ne pas delivrer automatiquement un "
        "medicament."
    )

    return PharmacistContext(
        system_prompt=system_prompt,
        user_context=user_context,
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from app import Article


LLM_API_URL_ENV = "LLM_API_URL"
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_TIMEOUT_ENV = "LLM_TIMEOUT_SECONDS"
DEFAULT_REMOTE_LLM_API_URL = "https://d337-165-245-132-103.ngrok-free.app"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class PharmacistContext:
    system_prompt: str
    user_context: str

    def as_text(self) -> str:
        return (
            "SYSTEM PROMPT\n"
            f"{self.system_prompt}\n\n"
            "CONTEXTE UTILISATEUR\n"
            f"{self.user_context}"
        )


@dataclass(frozen=True)
class AssistantReply:
    ok: bool
    answer: str = ""
    error: str = ""
    status_code: int | None = None
    raw_payload: dict[str, object] | None = None

    def display_text(self, context: PharmacistContext | None = None) -> str:
        if self.ok:
            return self.answer

        details = self.error or "Impossible de joindre le LLM distant."
        if context is None:
            return details

        return (
            f"{details}\n\n"
            "CONTEXTE PREPARE POUR LE LLM\n"
            f"{context.as_text()}"
        )


def build_pharmacist_context(
    symptoms: str, inventory: list[Article]
) -> PharmacistContext:
    available_articles = [
        article
        for article in inventory
        if article.quantite_stock > 0
    ]
    stock_lines = "\n".join(
        (
            f"- {article.nom} ({article.id_unique}), "
            f"{article.classe_therapeutique}, {article.emplacement_rayon}, "
            f"stock {article.quantite_stock}, prix {article.prix_formate}"
        )
        for article in available_articles
    )

    system_prompt = (
        "Tu es un assistant de gestion de stock pour une pharmacie. Tu aides "
        "le pharmacien a analyser l'inventaire, reperer les ruptures ou "
        "stocks faibles, preparer les priorites de reapprovisionnement et "
        "resumer les mouvements utiles. Tu restes strictement dans la gestion "
        "de stock: pas de diagnostic, pas de conseil medical au patient, pas "
        "de substitution sans validation du pharmacien."
    )
    user_context = (
        "Question ou objectif de gestion de stock:\n"
        f"{symptoms.strip() or '[non renseigne]'}\n\n"
        "Inventaire local actuel:\n"
        f"{stock_lines or '- Aucun medicament disponible'}\n\n"
        "Tache attendue: repondre avec des actions concretes liees au stock, "
        "aux rayons, aux quantites et au reapprovisionnement. Ne pas delivrer "
        "automatiquement un medicament."
    )

    return PharmacistContext(
        system_prompt=system_prompt,
        user_context=user_context,
    )


def is_llm_configured(api_url: str | None = None) -> bool:
    return bool((api_url if api_url is not None else os.getenv(LLM_API_URL_ENV, "")).strip())


def default_remote_llm_api_url() -> str:
    return DEFAULT_REMOTE_LLM_API_URL


def configured_llm_api_url(*, use_default: bool = False) -> str:
    configured = os.getenv(LLM_API_URL_ENV, "").strip()
    if configured:
        return configured
    return DEFAULT_REMOTE_LLM_API_URL if use_default else ""


def normalize_llm_api_url(api_url: str) -> str:
    stripped = api_url.strip()
    if not stripped:
        return ""

    parsed = urlparse(stripped)
    if not parsed.scheme or not parsed.netloc:
        return stripped

    path = parsed.path.rstrip("/")
    known_endpoints = (
        "/chat",
        "/api/chat",
        "/generate",
        "/api/generate",
        "/v1/chat/completions",
    )
    if path.endswith(known_endpoints):
        return stripped

    next_path = f"{path}/chat" if path else "/chat"
    return urlunparse(parsed._replace(path=next_path))


def call_remote_llm(
    context: PharmacistContext,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
) -> AssistantReply:
    configured_url = api_url if api_url is not None else os.getenv(LLM_API_URL_ENV, "")
    url = normalize_llm_api_url(configured_url)
    if not url:
        return AssistantReply(
            ok=False,
            error=(
                "LLM distant non configure. Definissez LLM_API_URL avec l'URL "
                "du serveur LLM, par exemple https://votre-cloud-llm/chat."
            ),
        )

    payload = {
        "system_prompt": context.system_prompt,
        "user_context": context.user_context,
        "prompt": context.user_context,
        "message": context.user_context,
        "messages": [
            {"role": "system", "content": context.system_prompt},
            {"role": "user", "content": context.user_context},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "ngrok-skip-browser-warning": "true",
    }

    token = api_key if api_key is not None else os.getenv(LLM_API_KEY_ENV, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, data=data, headers=headers, method="POST")
    timeout = timeout_seconds if timeout_seconds is not None else _llm_timeout_from_env()

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            status_code = getattr(response, "status", None)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return AssistantReply(
            ok=False,
            error=f"Erreur HTTP du serveur LLM ({exc.code}): {_compact_error(error_body)}",
            status_code=exc.code,
        )
    except TimeoutError:
        return AssistantReply(
            ok=False,
            error=f"Delai depasse lors de l'appel au LLM distant ({timeout:g}s).",
        )
    except URLError as exc:
        return AssistantReply(
            ok=False,
            error=f"Impossible de joindre le serveur LLM: {exc.reason}",
        )
    except OSError as exc:
        return AssistantReply(
            ok=False,
            error=f"Erreur reseau lors de l'appel au LLM: {exc}",
        )

    payload_json = _parse_json_object(response_body)
    if payload_json is None:
        answer = response_body.strip()
        return AssistantReply(
            ok=bool(answer),
            answer=answer,
            error="" if answer else "Reponse vide du serveur LLM.",
            status_code=status_code,
        )

    answer = _extract_answer(payload_json)
    if answer:
        return AssistantReply(
            ok=True,
            answer=answer,
            status_code=status_code,
            raw_payload=payload_json,
        )

    return AssistantReply(
        ok=False,
        error="Le serveur LLM a repondu sans champ de reponse exploitable.",
        status_code=status_code,
        raw_payload=payload_json,
    )


def _llm_timeout_from_env() -> float:
    raw_timeout = os.getenv(LLM_TIMEOUT_ENV, "").strip()
    if not raw_timeout:
        return DEFAULT_LLM_TIMEOUT_SECONDS

    try:
        return max(1.0, float(raw_timeout))
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS


def _parse_json_object(body: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_answer(payload: dict[str, object]) -> str:
    for key in ("answer", "response", "text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            text = first_choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

            choice_message = first_choice.get("message")
            if isinstance(choice_message, dict):
                content = choice_message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return ""


def _compact_error(body: str) -> str:
    payload = _parse_json_object(body)
    if payload is not None:
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    compacted = " ".join(body.split())
    return compacted[:300] if compacted else "reponse vide"
