from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ModelRouter:
    """Route a query to an LLM tier based on complexity."""

    def __init__(
        self,
        model_map: Optional[Dict[str, str]] = None,
        temperature: float = 0.0,
    ):
        load_dotenv()

        self.model_map = model_map or {
            "simple": "gpt-3.5-turbo",
            "medium": "gpt-4o-mini",
            "complex": "gpt-4o",
        }
        self.temperature = temperature

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
        compressed_context = (compression_result or {}).get("compressed_context", "").strip()

        if not compressed_context:
            return {
                "answer": "I could not find enough context to answer the question.",
                "model_used": model_used,
                "complexity": normalized_complexity,
            }

        llm = ChatOpenAI(model=model_used, temperature=self.temperature)
        messages = self._build_messages(query, compressed_context)
        response = llm.invoke(messages)

        return {
            "answer": self._extract_text(response),
            "model_used": model_used,
            "complexity": normalized_complexity,
        }

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
