"""
Query complexity for routing and retrieval sizing.

Primary path: rule + keyword heuristics → ``complexity_label``, ``complexity_score``,
``confidence``, ``reason_codes``. If confidence is below ``QUERY_ANALYZER_LOW_CONF_THRESHOLD``
and ``LLM_CLASSIFIER_ENABLED``, runs a small FLAN text2text classifier (timeout-guarded)
and may override the label. See env vars in ``QueryAnalyzer.__init__``.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


class QueryAnalyzer:
    """Classify query complexity with rules+heuristics and optional FLAN fallback."""

    COMPLEX_KEYWORDS = [
        "analyze", "analyse", "evaluate", "design", "compare", "comparing",
        "explain", "predict", "recommend", "strategy", "framework",
        "trade-off", "tradeoff", "assess", "develop", "critique",
        "justify", "synthesize", "investigate", "optimize", "benchmark",
        "contrast", "strategic", "comprehensive",
    ]
    MEDIUM_KEYWORDS = [
        "how", "why", "difference", "relationship", "describe",
        "what factors", "what role", "how does", "how do", "what are the",
        "what led", "what caused", "in what way", "to what extent",
        "differences", "impact", "role", "function",
    ]
    SIMPLE_STARTERS = [
        "what is", "what was", "who", "when", "where", "which",
        "how many", "how often", "how much", "is there", "does",
    ]
    MULTI_QUESTION_CONNECTORS = [
        "and also", "additionally", "as well as", "furthermore", "moreover",
    ]
    COMPLEXITY_ORDER = {"simple": 0, "medium": 1, "complex": 2}

    def __init__(
        self,
        low_conf_threshold: float = None,
        high_conf_threshold: float = None,
        llm_classifier_enabled: bool = None,
        llm_model_name: str = None,
        llm_timeout_ms: int = None,
        max_query_chars: int = None,
    ):
        self.low_conf_threshold = (
            low_conf_threshold
            if low_conf_threshold is not None
            else float(os.getenv("QUERY_ANALYZER_LOW_CONF_THRESHOLD", "0.60"))
        )
        self.high_conf_threshold = (
            high_conf_threshold
            if high_conf_threshold is not None
            else float(os.getenv("QUERY_ANALYZER_HIGH_CONF_THRESHOLD", "0.80"))
        )
        self.llm_classifier_enabled = (
            llm_classifier_enabled
            if llm_classifier_enabled is not None
            else os.getenv("LLM_CLASSIFIER_ENABLED", "true").lower() == "true"
        )
        self.llm_model_name = llm_model_name or os.getenv("LLM_CLASSIFIER_MODEL", "google/flan-t5-large")
        self.llm_timeout_ms = (
            llm_timeout_ms
            if llm_timeout_ms is not None
            else int(os.getenv("LLM_CLASSIFIER_TIMEOUT_MS", "2500"))
        )
        self.max_query_chars = (
            max_query_chars
            if max_query_chars is not None
            else int(os.getenv("QUERY_ANALYZER_MAX_INPUT_CHARS", "700"))
        )
        self._llm_pipeline = None

    def analyze(self, query: str) -> Dict[str, Any]:
        normalized = (query or "").lower().strip()
        word_count = len(normalized.split())
        baseline = self._baseline_decision(normalized, word_count)

        final_label = baseline["complexity_label"]
        final_confidence = baseline["confidence"]
        final_complexity_score = baseline["complexity_score"]
        source = "heuristic"
        llm_status = "not_called"
        llm_result = None

        if self.llm_classifier_enabled and baseline["confidence"] < self.low_conf_threshold:
            llm_result = self._classify_with_flan(query)
            llm_status = llm_result["status"]

            if llm_status == "ok" and llm_result["confidence"] >= self.high_conf_threshold:
                final_label = llm_result["complexity_label"]
                final_confidence = llm_result["confidence"]
                final_complexity_score = llm_result["complexity_score"]
                source = "llm"
            else:
                final_label = baseline["complexity_label"]
                final_confidence = baseline["confidence"]
                final_complexity_score = baseline["complexity_score"]
                source = "heuristic_fallback"

        return {
            "query": query,
            "word_count": word_count,
            "complexity_score": round(float(final_complexity_score), 2),
            "complexity": final_label,  # backwards compatibility
            "complexity_label": final_label,
            "confidence": round(float(final_confidence), 2),
            "source": source,
            "reason_codes": baseline["reason_codes"],
            "baseline": baseline,
            "llm": llm_result,
            "llm_status": llm_status,
            "thresholds": {
                "low_conf_threshold": self.low_conf_threshold,
                "high_conf_threshold": self.high_conf_threshold,
            },
        }

    def _baseline_decision(self, normalized: str, word_count: int) -> Dict[str, Any]:
        label, score, reason_codes = self._heuristic_score(normalized, word_count)
        return {
            "complexity_label": label,
            "confidence": round(score, 2),
            "complexity_score": round(self._label_confidence_to_score(label, score), 2),
            "reason_codes": reason_codes,
        }

    def _heuristic_score(self, normalized: str, word_count: int) -> Tuple[str, float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        complex_hits = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in normalized)
        medium_hits = sum(1 for kw in self.MEDIUM_KEYWORDS if kw in normalized)
        simple_starter = any(normalized.startswith(kw) for kw in self.SIMPLE_STARTERS)
        multi_question = normalized.count("?") > 1
        connector = any(c in normalized for c in self.MULTI_QUESTION_CONNECTORS)

        if complex_hits:
            score += min(0.25 + (0.08 * complex_hits), 0.45)
            reasons.append("complex_keyword")
        if medium_hits:
            score += min(0.12 + (0.05 * medium_hits), 0.30)
            reasons.append("medium_keyword")
        if multi_question or connector or (" and " in normalized and word_count > 10):
            score += 0.20
            reasons.append("multi_part_query")
        if re.search(r"\b(compare|versus|vs|trade[- ]?off|impact|effect)\b", normalized):
            score += 0.18
            reasons.append("comparative_or_relational")

        if word_count > 20:
            score += 0.15
            reasons.append("long_query")
        elif word_count >= 10:
            score += 0.08
            reasons.append("medium_length_query")
        elif word_count <= 5:
            score -= 0.10
            reasons.append("very_short_query")

        score = max(0.0, min(1.0, score))

        if simple_starter and score < 0.50:
            label = "simple"
            confidence = max(0.75, 0.9 - (score / 2))
            # Penalize confidence near boundaries where class flips are likely.
            min_distance = min(abs(score - 0.38), abs(score - 0.70))
            if min_distance < 0.05:
                confidence -= 0.20
                reasons.append("boundary_low_margin")
            elif min_distance < 0.10:
                confidence -= 0.10
                reasons.append("boundary_mid_margin")
            reasons.append("simple_starter")
            return label, max(0.0, min(1.0, confidence)), reasons

        if score >= 0.70:
            confidence = max(0.72, score)
            min_distance = abs(score - 0.70)
            if min_distance < 0.05:
                confidence -= 0.20
                reasons.append("boundary_low_margin")
            elif min_distance < 0.10:
                confidence -= 0.10
                reasons.append("boundary_mid_margin")
            return "complex", max(0.0, min(1.0, confidence)), reasons
        if score >= 0.38:
            confidence = max(0.62, score)
            min_distance = min(abs(score - 0.38), abs(score - 0.70))
            if min_distance < 0.05:
                confidence -= 0.20
                reasons.append("boundary_low_margin")
            elif min_distance < 0.10:
                confidence -= 0.10
                reasons.append("boundary_mid_margin")
            return "medium", max(0.0, min(1.0, confidence)), reasons
        confidence = max(0.68, 1 - score)
        min_distance = abs(score - 0.38)
        if min_distance < 0.05:
            confidence -= 0.20
            reasons.append("boundary_low_margin")
        elif min_distance < 0.10:
            confidence -= 0.10
            reasons.append("boundary_mid_margin")
        return "simple", max(0.0, min(1.0, confidence)), reasons

    def _classify_with_flan(self, query: str) -> Dict[str, Any]:
        truncated_query = (query or "").strip()[: self.max_query_chars]

        def _run() -> Dict[str, Any]:
            pipeline = self._get_pipeline()
            prompt = self._build_prompt(truncated_query)
            output = pipeline(prompt, max_new_tokens=80, do_sample=False)
            generated = output[0]["generated_text"] if output else ""
            return self._parse_llm_response(generated)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_run)
            parsed = future.result(timeout=self.llm_timeout_ms / 1000)
            parsed["status"] = "ok"
            return parsed
        except FuturesTimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return {"status": "timeout", "complexity_label": None, "confidence": 0.0, "reason": "llm_timeout"}
        except ValueError as exc:
            return {"status": "parse_error", "complexity_label": None, "confidence": 0.0, "reason": str(exc)}
        except Exception as exc:  # runtime/import/parse issues
            return {"status": "runtime_error", "complexity_label": None, "confidence": 0.0, "reason": str(exc)}
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _get_pipeline(self):
        if self._llm_pipeline is not None:
            return self._llm_pipeline
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("transformers package is required for LLM classifier") from exc
        self._llm_pipeline = pipeline("text2text-generation", model=self.llm_model_name)
        return self._llm_pipeline

    def _build_prompt(self, query: str) -> str:
        return (
            "Classify query complexity.\n"
            "Return strict JSON with keys: complexity_label, confidence, reason.\n"
            "complexity_label must be one of: simple, medium, complex.\n"
            "confidence must be a number between 0 and 1.\n"
            f"Query: {query}"
        )

    def _parse_llm_response(self, output_text: str) -> Dict[str, Any]:
        # FLAN may echo prompt text, so parse the first JSON object in output.
        match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM output")
        payload = json.loads(match.group(0))
        label = str(payload.get("complexity_label", "")).strip().lower()
        if label not in self.COMPLEXITY_ORDER:
            raise ValueError("Invalid complexity_label in LLM output")
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        return {
            "complexity_label": label,
            "confidence": round(confidence, 2),
            "complexity_score": round(self._label_confidence_to_score(label, confidence), 2),
            "reason": str(payload.get("reason", "")).strip(),
            "raw_output": output_text,
        }

    def _label_confidence_to_score(self, label: str, confidence: float) -> float:
        """
        Convert bucketed complexity + confidence into a continuous complexity score.
        This preserves label semantics while enabling score-first downstream logic.
        """
        bounded_conf = max(0.0, min(1.0, float(confidence)))
        if label == "simple":
            return 0.0 + (0.4 * (1.0 - bounded_conf))
        if label == "medium":
            return 0.4 + (0.3 * bounded_conf)
        if label == "complex":
            return 0.7 + (0.3 * bounded_conf)
        return 0.5
