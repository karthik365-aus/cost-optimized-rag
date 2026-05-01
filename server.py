"""
OpenAI-compatible wrapper for the local RAG pipeline.

Exposes:
- GET /                -> health
- GET /v1/models       -> model list for OpenAI-compatible clients
- POST /v1/chat/completions
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

LOG = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "notre-dame-assistant"
    messages: List[ChatMessage] = Field(default_factory=list)
    files: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


OFFTOPIC_COVERAGE_THRESHOLD = float(os.getenv("OFFTOPIC_COVERAGE_THRESHOLD", "0.30"))
OFFTOPIC_CONFIDENCE_THRESHOLD = float(os.getenv("OFFTOPIC_CONFIDENCE_THRESHOLD", "0.40"))
OPEN_WEBUI_BASE_URL = os.getenv("OPEN_WEBUI_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
OPEN_WEBUI_BEARER_TOKEN = (os.getenv("OPEN_WEBUI_BEARER_TOKEN") or "").strip()
OPEN_WEBUI_TIMEOUT_SECONDS = float(os.getenv("OPEN_WEBUI_TIMEOUT_SECONDS", "10"))


def _extract_last_user_query(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if (msg.role or "").lower() != "user":
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts).strip()
    return ""


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Best-effort PDF text extraction; returns empty string on failure."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def _decode_pdf_bytes(raw: str) -> bytes | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    payload = raw.strip()
    if payload.startswith("data:application/pdf;base64,"):
        payload = payload.split(",", 1)[1].strip()
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None


def _extract_text_from_file_dict(f: Dict[str, Any]) -> str:
    """Extract text from a file payload dict (plain text or PDF bytes)."""
    filename = str(f.get("name") or f.get("filename") or "").lower()
    mime = str(f.get("mime_type") or f.get("content_type") or f.get("type") or "").lower()
    is_pdf = filename.endswith(".pdf") or "application/pdf" in mime

    # Direct text fields
    for key in ("text", "content", "body"):
        raw = f.get(key)
        if isinstance(raw, str) and raw.strip():
            if is_pdf:
                pdf_bytes = _decode_pdf_bytes(raw)
                if pdf_bytes:
                    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
                    if extracted:
                        return extracted
            return raw.strip()

    # Base64-like binary fields commonly used by clients
    for key in ("data", "base64", "file"):
        raw = f.get(key)
        if isinstance(raw, str) and raw.strip():
            if is_pdf:
                pdf_bytes = _decode_pdf_bytes(raw)
                if pdf_bytes:
                    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
                    if extracted:
                        return extracted

    return ""


def _extract_candidate_file_ids(req: ChatCompletionRequest) -> List[str]:
    ids: List[str] = []

    def _capture_id(obj: Dict[str, Any]):
        for key in ("id", "file_id"):
            raw = obj.get(key)
            if isinstance(raw, str) and raw.strip():
                ids.append(raw.strip())

    for f in req.files or []:
        if isinstance(f, dict):
            _capture_id(f)

    for msg in req.messages:
        content = msg.content
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                _capture_id(part)

    # de-duplicate preserving order
    seen = set()
    deduped: List[str] = []
    for fid in ids:
        if fid in seen:
            continue
        seen.add(fid)
        deduped.append(fid)
    return deduped


def _openwebui_headers() -> Dict[str, str]:
    headers = {}
    if OPEN_WEBUI_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {OPEN_WEBUI_BEARER_TOKEN}"
    return headers


def _fetch_openwebui_file_text(file_id: str) -> str:
    """
    Best-effort retrieval from Open WebUI API.
    Requires OPEN_WEBUI_BEARER_TOKEN in authenticated setups.
    """
    if not file_id:
        return ""
    headers = _openwebui_headers()

    # 1) Prefer extracted content stored in file.data.content
    try:
        url = f"{OPEN_WEBUI_BASE_URL}/api/v1/files/{file_id}/data/content"
        r = requests.get(url, headers=headers, timeout=OPEN_WEBUI_TIMEOUT_SECONDS)
        if r.ok:
            payload = r.json() if "application/json" in (r.headers.get("content-type", "")) else {}
            content = payload.get("content", "") if isinstance(payload, dict) else ""
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass

    # 2) Fallback to raw file download endpoint
    try:
        url = f"{OPEN_WEBUI_BASE_URL}/api/v1/files/{file_id}/content"
        r = requests.get(url, headers=headers, timeout=OPEN_WEBUI_TIMEOUT_SECONDS)
        if not r.ok:
            return ""
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/pdf" in ctype:
            return _extract_text_from_pdf_bytes(r.content)
        # Assume text-ish fallback
        text = r.text or ""
        return text.strip()
    except Exception:
        return ""


def _extract_session_documents(req: ChatCompletionRequest) -> List[str]:
    docs: List[str] = []

    # 1) Extract file-like payloads if client sends them at top-level.
    for f in req.files or []:
        if not isinstance(f, dict):
            continue
        text = _extract_text_from_file_dict(f)
        if len(text) >= 80:
            docs.append(text)

    # 2) Extract structured content parts from user messages when available.
    for msg in req.messages:
        if (msg.role or "").lower() != "user":
            continue
        content = msg.content
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            p_type = str(part.get("type", "")).lower()
            if p_type in {"input_text", "text", "document", "input_file", "file"}:
                text = ""
                if isinstance(part, dict):
                    text = _extract_text_from_file_dict(part)
                if len(text) >= 80:
                    docs.append(text)

    # De-duplicate while preserving order.
    seen = set()
    unique_docs: List[str] = []
    for d in docs:
        if d in seen:
            continue
        seen.add(d)
        unique_docs.append(d)

    # 3) If only file IDs were sent, fetch content from Open WebUI API.
    file_ids = _extract_candidate_file_ids(req)
    if file_ids:
        LOG.info("session_upload_file_ids_detected=%s", len(file_ids))
    for fid in file_ids:
        fetched = _fetch_openwebui_file_text(fid)
        if len(fetched) >= 80 and fetched not in seen:
            seen.add(fetched)
            unique_docs.append(fetched)

    return unique_docs


def _openai_style_response(model: str, answer: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> Dict[str, Any]:
    total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "rag-pipeline",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": total_tokens,
        },
    }


def _is_off_topic(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True

    chitchat = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "ok",
        "okay",
        "cool",
        "great",
        "nice",
    }
    short_allow = {"rag", "chroma", "embedding", "embeddings", "retrieval"}
    if q in chitchat:
        return True
    if len(q.split()) <= 2 and q not in short_allow:
        return True

    personal_patterns = [
        r"^my name is\b",
        r"^i am\b",
        r"^i'm\b",
        r"^i like\b",
        r"^i want\b",
        r"^can you help me with\b",
        r"^what do you think about\b",
        r"^tell me a joke\b",
        r"^who are you\b",
        r"^what are you\b",
    ]
    return any(re.match(p, q) for p in personal_patterns)


app = FastAPI(title="RAG OpenAI-Compatible Server", version="1.0.0")
pipeline = RAGPipeline()


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            ModelCard(
                id="notre-dame-assistant",
                created=int(time.time()),
                owned_by="cost-optimized-rag",
            ).model_dump()
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    # Some clients (e.g., Open WebUI) may send stream=true by default.
    # For now, degrade gracefully to non-streaming response.
    _ = bool(req.stream)

    query = _extract_last_user_query(req.messages)
    if not query:
        raise HTTPException(
            status_code=400,
            detail="No valid user message found in request.messages",
        )
    if _is_off_topic(query):
        return _openai_style_response(
            req.model,
            (
                "I can only answer questions about the course material. "
                "Try asking something like 'What is RAG?' or "
                "'How does retrieval work in a RAG pipeline?'"
            ),
        )

    try:
        session_documents = _extract_session_documents(req)
        LOG.info(
            "session_upload_docs_detected=%s query_chars=%s",
            len(session_documents),
            len(query),
        )
        result = pipeline.run(query, session_documents=session_documents)
        if (
            float(result.get("coverage_score", 0.0) or 0.0) < OFFTOPIC_COVERAGE_THRESHOLD
            and float(result.get("confidence_score_final", 0.0) or 0.0) < OFFTOPIC_CONFIDENCE_THRESHOLD
        ):
            return _openai_style_response(
                req.model,
                (
                    "I couldn't find relevant information in the course material to answer that. "
                    "Please ask a question related to the lecture content."
                ),
            )
        answer = str(result.get("final_answer", "")).strip()
        if not answer:
            answer = "I could not generate a reliable answer."
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    prompt_tokens = int(result.get("input_tokens", 0) or 0)
    completion_tokens = int(result.get("output_tokens", 0) or 0)
    return _openai_style_response(req.model, answer, prompt_tokens, completion_tokens)
