"""
LLM routing: ``simple`` / ``medium`` → local OpenAI-compatible server (e.g. LM Studio);
``complex`` → OpenAI API.

Uses ``langchain_openai.ChatOpenAI`` with ``base_url`` for local. Env: ``LOCAL_OPENAI_*``,
``LOCAL_SIMPLE_MODEL``, ``LOCAL_MEDIUM_MODEL``, ``OPENAI_COMPLEX_MODEL``. Optional
``GET .../v1/models`` health check so model IDs match what is loaded.
"""
import time
import os
from typing import Any, Dict, Optional

import requests
import tiktoken
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class _WhitespaceTokenizer:
    def encode(self, text: str):
        return text.split()


def _build_tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Offline-safe fallback for local-only runs.
        return _WhitespaceTokenizer()


_TOKENIZER = _build_tokenizer()


class ModelRouter:
    """Route simple/medium to local endpoint and complex to OpenAI."""

    def __init__(
        self,
        model_map: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
    ):
        load_dotenv()

        self.model_map = model_map or {
            "simple": os.getenv("LOCAL_SIMPLE_MODEL", "tinyllama"),
            "medium": os.getenv("LOCAL_MEDIUM_MODEL", "ministral"),
            "complex": os.getenv("OPENAI_COMPLEX_MODEL", "gpt-4o-mini"),
        }
        self.local_openai_base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1")
        self.local_openai_api_key = os.getenv("LOCAL_OPENAI_API_KEY", "lm-studio")
        self.local_healthcheck_enabled = os.getenv("LOCAL_HEALTHCHECK_ENABLED", "true").lower() == "true"
        self.local_healthcheck_timeout = float(os.getenv("LOCAL_HEALTHCHECK_TIMEOUT_SECONDS", "3"))
        self.temperature = temperature
        self._local_models_cache: Optional[set[str]] = None

    def route(
        self,
        query: str,
        complexity: str,
        compression_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an answer using the model tier mapped to the query complexity.

        Args:
            query: Original user question.
            complexity: Query complexity label: simple, medium, or complex.
            compression_result: Output dict from ContextCompressor.compress().

        Returns:
            Dict containing the generated answer and routing metadata.
        """
        normalized_complexity = (complexity or "medium").strip().lower()
        model_used = self.model_map.get(normalized_complexity, self.model_map["medium"])
        model_source = "openai" if normalized_complexity == "complex" else "local_openai_compatible"
        fallback_reason = None
        compressed_context = (compression_result or {}).get("compressed_context", "").strip()

        if not compressed_context:
            return {
                "answer": "I could not find enough context to answer the question.",
                "model_used": model_used,
                "model_source": model_source,
                "fallback_reason": "empty_context",
                "complexity": normalized_complexity,
                "input_tokens": 0,
                "output_tokens": 0,
                "time_taken_seconds": 0.0,
            }

        input_text = query + "\n\n" + compressed_context
        input_tokens = len(_TOKENIZER.encode(input_text))

        start = time.time()
        answer = ""

        try:
            answer = self._generate_answer(model_used, model_source, query, compressed_context)
        except Exception as exc:
            error_message = self._format_exception(exc)
            fallback_reason = f"generation_failure:{type(exc).__name__}:{error_message[:220]}"
            print(f"[ModelRouter] Generation failed for model={model_used} source={model_source}: {type(exc).__name__}: {error_message}")
            answer = "I could not generate a reliable answer."

        elapsed = round(time.time() - start, 3)
        output_tokens = len(_TOKENIZER.encode(answer))

        return {
            "answer": answer,
            "model_used": model_used,
            "model_source": model_source,
            "fallback_reason": fallback_reason,
            "complexity": normalized_complexity,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken_seconds": elapsed,
        }

    def _generate_answer(self, model_name: str, model_source: str, query: str, compressed_context: str) -> str:
        llm_kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
        }
        if model_source == "local_openai_compatible":
            if self.local_healthcheck_enabled:
                self._ensure_local_model_available(model_name)
            llm_kwargs["base_url"] = self.local_openai_base_url
            llm_kwargs["api_key"] = self.local_openai_api_key

        llm = ChatOpenAI(**llm_kwargs)
        response = llm.invoke(self._build_messages(query, compressed_context))
        answer = self._extract_text(response)
        if not answer:
            raise RuntimeError("Model returned empty output")
        return answer

    def _ensure_local_model_available(self, model_name: str) -> None:
        if self._local_models_cache is None:
            self._local_models_cache = self._fetch_local_models()
        if model_name in self._local_models_cache:
            return
        available = ", ".join(sorted(self._local_models_cache)) if self._local_models_cache else "(none)"
        raise RuntimeError(
            f"Local model '{model_name}' is not loaded on {self.local_openai_base_url}. "
            f"Available models: {available}"
        )

    def _fetch_local_models(self) -> set[str]:
        base = self.local_openai_base_url.rstrip("/")
        if base.endswith("/v1"):
            models_url = f"{base}/models"
        else:
            models_url = f"{base}/v1/models"

        response = requests.get(models_url, timeout=self.local_healthcheck_timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        return {item.get("id", "").strip() for item in data if isinstance(item, dict) and item.get("id")}

    def _format_exception(self, exc: Exception) -> str:
        parts = [str(exc).strip().replace("\n", " ")]
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            parts.append(f"status_code={status_code}")
        response = getattr(exc, "response", None)
        if response is not None:
            body = getattr(response, "text", "")
            if body:
                parts.append(f"response_body={body[:220]}")
        return " | ".join(part for part in parts if part)

    def _build_messages(self, query: str, compressed_context: str):
        return [
            SystemMessage(
                content=(
                    "You answer questions using only the provided context. "
                    "If the context is insufficient, say that clearly instead of guessing."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {query}\n\n"
                    f"Context:\n{compressed_context}\n\n"
                    "Answer the question as concisely as possible."
                )
            ),
        ]

    def _extract_text(self, response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    text_parts.append(item.text)
            return "\n".join(part.strip() for part in text_parts if part).strip()
        return str(content).strip()
