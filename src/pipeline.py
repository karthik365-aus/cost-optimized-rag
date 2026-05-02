"""
End-to-end RAG orchestration for the BSAN 765 pipeline.

Flow (single ``RAGPipeline.run``):
    1. QueryAnalyzer — complexity label + score + confidence (+ optional FLAN).
    2. AdaptiveRetriever — Chroma similarity search; dynamic k + coverage retry.
    3. ContextCompressor — sentence selection; hybrid TF-IDF / embeddings; adaptive budget.
    4. ModelRouter — simple/medium via local OpenAI-compatible API; complex via OpenAI.
    5. confidence_checker — scores answer vs compressed context; optional stronger-model retry.

The return dict aggregates stage timings, retrieval/compression/router/confidence metadata,
and the final answer (for ``evaluate.py`` and logging).
"""
from pathlib import Path
import difflib
import hashlib
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.query_analyzer import QueryAnalyzer
from src.adaptive_retriever import AdaptiveRetriever
from src.context_compression import ContextCompressor
from src.model_router import ModelRouter
from src.confidence_checker import check_confidence
from src.preflight import log_pipeline_startup
from src.semantic_cache import SemanticCache
from src.adaptive_retriever import CHROMA_DIR
from src.session_multimodal_retriever import (
    SessionMultimodalError,
    SessionMultimodalRetriever,
    detect_visual_pdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RAGPipeline:
    """Wire analyzer → retriever → compressor → router → confidence checker."""

    def __init__(self, documents_path=None):
        documents_path = documents_path or str(PROJECT_ROOT / "data" / "documents")
        self.documents_path = Path(documents_path)
        self.analyzer = QueryAnalyzer()
        self.retriever = AdaptiveRetriever(documents_path=documents_path)
        self.compressor = ContextCompressor()
        self.router = ModelRouter()
        self.semantic_cache = SemanticCache()
        self.corpus_hash = SemanticCache.get_corpus_hash(CHROMA_DIR)
        self._vocab_index_cache = None
        self._vocab_all_cache = None
        self._session_query_cache: dict[tuple[str, str], dict] = {}
        log_pipeline_startup(PROJECT_ROOT, documents_path)

    SPELLCHECK_EXCLUDE = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
        "or", "for", "in", "on", "at", "with", "by", "from", "that", "this",
        "it", "as", "what", "which", "who", "when", "where", "how", "why",
        "university", "notre", "dame",
    }
    QUERY_HINT_VOCAB = {
        "founded", "founder", "founding", "established", "history", "campus",
        "academics", "admissions", "students", "faculty", "library", "college",
        "program", "tuition", "ranking", "rankings", "notre", "dame", "university",
    }
    USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
    HYDE_MIN_COVERAGE_GAIN = float(os.getenv("HYDE_MIN_COVERAGE_GAIN", "0.03"))
    HYDE_MAX_CHARS = int(os.getenv("HYDE_MAX_CHARS", "700"))
    MAX_HEAVY_FALLBACKS = int(os.getenv("PIPELINE_MAX_HEAVY_FALLBACKS", "1"))
    DEEP_OPENAI_MIN_COVERAGE = float(os.getenv("DEEP_OPENAI_MIN_COVERAGE", "0.40"))
    USE_SESSION_MULTIMODAL_EMBEDDING = (
        os.getenv("USE_SESSION_MULTIMODAL_EMBEDDING", "false").lower() == "true"
    )
    SESSION_UPLOAD_MIN_COMPLEXITY_SCORE = float(
        os.getenv("SESSION_UPLOAD_MIN_COMPLEXITY_SCORE", "0.60")
    )
    SESSION_PRIORITY_HITS = int(os.getenv("SESSION_PRIORITY_HITS", "2"))
    PROTECTED_FACTUAL_PREFIXES = (
        "how many ",
        "who ",
        "when ",
        "where ",
        "which ",
        "what is ",
        "what was ",
    )
    TECHNICAL_ENTITY_ALLOWLIST = {
        "bm25",
        "tf-idf",
        "bert",
        "embeddings",
        "cosine",
        "langchain",
        "llamaindex",
        "chromadb",
        "rag",
        "llm",
        "api",
    }

    def _build_vocab_index(self) -> dict[str, set[str]]:
        if self._vocab_index_cache is not None:
            return self._vocab_index_cache
        index: dict[str, set[str]] = {}
        if self.documents_path.exists():
            for txt_file in self.documents_path.glob("**/*.txt"):
                try:
                    content = txt_file.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                for token in re.findall(r"[a-z]{3,}", content):
                    index.setdefault(token[0], set()).add(token)
        self._vocab_index_cache = index
        self._vocab_all_cache = sorted({tok for bucket in index.values() for tok in bucket})
        return index

    def _spell_correct_query(self, normalized_query: str) -> str:
        vocab_index = self._build_vocab_index()
        full_vocab = set(self._vocab_all_cache or []) | self.QUERY_HINT_VOCAB

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            if len(token) < 4 or token in self.SPELLCHECK_EXCLUDE:
                return token
            if token[1:] in full_vocab:
                return token[1:]
            if token[:-1] in full_vocab:
                return token[:-1]
            if len(token) > 5 and token[0] == token[1] and token[1:] in full_vocab:
                return token[1:]
            candidates = list(vocab_index.get(token[0], set()))
            if not candidates or token in candidates:
                candidates = list(full_vocab)
                if not candidates or token in candidates:
                    return token
            close = difflib.get_close_matches(token, candidates, n=1, cutoff=0.78)
            return close[0] if close else token

        for prefix in self.PROTECTED_FACTUAL_PREFIXES:
            if normalized_query.startswith(prefix):
                suffix = normalized_query[len(prefix):]
                corrected_suffix = re.sub(r"[a-z]+", _replace, suffix)
                return f"{prefix}{corrected_suffix}".strip()

        return re.sub(r"[a-z]+", _replace, normalized_query)

    def _normalize_query_text(self, query: str) -> tuple[str, str]:
        text = (query or "").strip()
        if not text:
            return "", ""
        normalized = re.sub(r"\s+", " ", text)
        lowered = normalized.lower()
        corrected = self._spell_correct_query(lowered)
        return lowered, corrected

    def _build_query_variants(self, original_query: str, normalized_query: str, corrected_query: str) -> list[str]:
        variants: list[str] = []
        for candidate in [original_query.strip(), normalized_query.strip(), corrected_query.strip()]:
            if candidate and candidate not in variants:
                variants.append(candidate)
        return variants

    def _generate_hyde_text(self, query: str, complexity: str) -> str:
        if not self.USE_HYDE:
            return ""
        complexity_rank = {"simple": 0, "medium": 1, "complex": 2}
        preferred = "medium" if complexity_rank.get(complexity, 1) < 1 else complexity
        model_name = self.router.model_map.get(preferred, self.router.model_map.get("medium"))
        try:
            llm_kwargs = {
                "model": model_name,
                "temperature": 0.0,
                "base_url": self.router.local_openai_base_url,
                "api_key": self.router.local_openai_api_key,
            }
            if self.router.local_healthcheck_enabled:
                self.router._ensure_local_model_available(model_name)
            llm = ChatOpenAI(**llm_kwargs)
            msg = [
                SystemMessage(
                    content=(
                        "Generate a concise, factual paragraph that could answer the question. "
                        "Use neutral wording and avoid speculation. No bullet points."
                    )
                ),
                HumanMessage(content=f"Question: {query}"),
            ]
            response = llm.invoke(msg)
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    for item in content
                )
            return str(content).strip()[: self.HYDE_MAX_CHARS]
        except Exception as exc:
            print(f"[HyDE] skipped due to generation failure: {type(exc).__name__}: {exc}")
            return ""

    @staticmethod
    def _extract_answer_entities(text: str) -> set[str]:
        raw = text or ""
        # Capture title-cased entity-like spans (1-4 words) and normalize.
        spans = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", raw)
        entities = {re.sub(r"\s+", " ", s.strip().lower()) for s in spans}
        # Remove common sentence starters and generic fillers.
        drop = {
            "the", "this", "that", "it", "i", "we", "you", "notre", "dame",
            "university", "context", "provided", "information",
        }
        return {e for e in entities if e not in drop and len(e) >= 4}

    def _grounding_gate(self, answer: str, context: str) -> tuple[bool, dict]:
        ans = (answer or "").strip()
        ctx = (context or "").lower()
        if not ans or not ctx:
            return False, {"reason": "empty_answer_or_context", "unsupported_entities": []}

        entities = self._extract_answer_entities(ans)
        filtered_entities = []
        for e in entities:
            parts = set(re.findall(r"[a-z0-9-]+", e.lower()))
            if parts & self.TECHNICAL_ENTITY_ALLOWLIST:
                continue
            filtered_entities.append(e)
        unsupported = [e for e in filtered_entities if e not in ctx]

        # If answer introduces unseen entities, treat as ungrounded.
        if unsupported:
            return False, {"reason": "unsupported_entities", "unsupported_entities": unsupported}
        return True, {"reason": "grounded", "unsupported_entities": []}

    @staticmethod
    def _is_factual_query(query: str) -> bool:
        q = (query or "").lower().strip()
        factual_starts = ("who ", "when ", "where ", "how many", "what is", "what was", "which ")
        analytical_markers = ("compare", "analyze", "explain why", "strategy", "impact of", "trade-off")
        return q.startswith(factual_starts) and not any(m in q for m in analytical_markers)

    @staticmethod
    def _is_session_extractive_query(query: str) -> bool:
        """
        In upload-only sessions, keep extractive path very narrow (numeric/date facts).
        Conceptual definitions and entity questions should use LLM generation.
        """
        q = (query or "").lower().strip()
        return bool(
            re.match(
                r"^(how many\b|what year\b|when\b)",
                q,
            )
        )

    @staticmethod
    def _subject_aliases(subject: str) -> set[str]:
        raw = re.sub(r"\s+", " ", (subject or "").strip().lower())
        raw = re.sub(r"[^\w\s&.-]", "", raw).strip()
        if not raw:
            return set()
        aliases = {raw}
        if raw.startswith("the "):
            aliases.add(raw[4:].strip())
        if raw.startswith("university of "):
            core = raw[len("university of "):].strip()
            if core:
                aliases.add(core)
        else:
            aliases.add(f"university of {raw}")
        if raw.startswith("college of "):
            core = raw[len("college of "):].strip()
            if core:
                aliases.add(core)
        if " and " in raw:
            aliases.add(raw.replace(" and ", " & "))
        if " & " in raw:
            aliases.add(raw.replace(" & ", " and "))
        words = re.findall(r"[a-z0-9]+", raw)
        if len(words) >= 2:
            acronym = "".join(w[0] for w in words if w)
            if len(acronym) >= 2:
                aliases.add(acronym)
                aliases.add(" ".join(acronym))
                aliases.add(".".join(acronym) + ".")
        return {a for a in aliases if a}

    @classmethod
    def _alias_match_strength(cls, text: str, aliases: set[str]) -> float:
        text_l = (text or "").lower()
        if not text_l or not aliases:
            return 0.0
        best = 0.0
        text_tokens = set(re.findall(r"[a-z0-9]+", text_l))
        for alias in aliases:
            alias_l = alias.lower().strip()
            if not alias_l:
                continue
            if alias_l in text_l:
                best = max(best, 1.0)
                continue
            alias_tokens = set(re.findall(r"[a-z0-9]+", alias_l))
            if not alias_tokens:
                continue
            j = len(alias_tokens & text_tokens) / max(1, len(alias_tokens))
            best = max(best, j)
        return best

    @staticmethod
    def _alias_near_founding_verb(text: str, aliases: set[str], max_distance: int = 90) -> bool:
        text_l = (text or "").lower()
        if not text_l:
            return False
        verb_positions = [m.start() for m in re.finditer(r"\b(founded|founder|established|started)\b", text_l)]
        if not verb_positions:
            return False
        phrase_aliases = [a for a in aliases if len(a.strip()) >= 4]
        for alias in phrase_aliases:
            start = text_l.find(alias)
            if start == -1:
                continue
            if any(abs(start - vp) <= max_distance for vp in verb_positions):
                return True
        return False

    def _extractive_answer(self, query: str, factual_hits: list[dict]) -> str | None:
        if not factual_hits:
            return None
        q = (query or "").lower()
        word_to_num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
        }
        query_tokens = set(re.findall(r"[a-z0-9]+", q))
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "in",
            "on", "at", "and", "or", "do", "does", "did", "how", "what", "who",
            "when", "where", "which", "many",
        }
        query_tokens = {t for t in query_tokens if t not in stop}

        def overlap_score(text: str) -> float:
            tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
            return len(tokens & query_tokens) / max(1, len(query_tokens))

        if re.match(r"^who\s+(founded|established|started)\s+", q):
            subject = re.sub(r"^who\s+(founded|established|started)\s+", "", q).strip(" ?.")
            aliases = self._subject_aliases(subject)
            candidates = []
            for h in factual_hits:
                text = h.get("text", "")
                text_l = text.lower()
                if not any(v in text_l for v in ("founded", "founder", "established", "started")):
                    continue
                if not self._alias_near_founding_verb(text_l, aliases):
                    continue
                alias_strength = self._alias_match_strength(text_l, aliases)
                if alias_strength < 0.75:
                    continue
                subentity_penalty = any(
                    kw in text_l
                    for kw in ("review", "journal", "newspaper", "magazine", "program", "department")
                )
                by_phrase_bonus = 0.2 if re.search(r"\b(founded|established|started)\s+by\b", text_l) else 0.0
                score = (
                    0.45 * overlap_score(text)
                    + 0.35 * float(h.get("score", 0.0))
                    + 0.20 * alias_strength
                    + by_phrase_bonus
                    - (0.35 if subentity_penalty else 0.0)
                )
                candidates.append((score, text))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_text = candidates[0]
                if best_score >= 0.45:
                    return best_text
            return None

        if "how many" in q:
            subject = re.sub(r"^how many\s+", "", q).strip()
            subject = re.split(r"\b(is|are|was|were|do|does|did|at|in|for|of|to)\b", subject, maxsplit=1)[0].strip(" ?.,")
            for h in sorted(factual_hits, key=lambda x: overlap_score(x["text"]), reverse=True):
                matches = re.findall(
                    r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b",
                    h["text"],
                    re.I,
                )
                if matches:
                    values = []
                    for tok in matches:
                        t = tok.lower()
                        if t.isdigit():
                            values.append(int(t))
                        elif t in word_to_num:
                            values.append(int(word_to_num[t]))
                    if not values:
                        continue
                    num = str(max(values))
                    if subject:
                        return f"{num} {subject}."
                    return f"{num}."
        if "when" in q or "what year" in q:
            for h in sorted(factual_hits, key=lambda x: overlap_score(x["text"]), reverse=True):
                if re.search(r"\b(18|19|20)\d{2}\b", h["text"]):
                    return h["text"]
        if re.match(r"^(what is|what was|define)\b", q):
            subject = re.sub(r"^(what is|what was|define)\s+", "", q).strip(" ?.")
            subject_tokens = set(re.findall(r"[a-z0-9]+", subject))
            ranked_defs = []
            for h in factual_hits:
                text = (h.get("text") or "").strip()
                normalized_text = re.sub(r"\s+", " ", text).strip()
                candidate_spans = []
                if subject:
                    subj_pat = re.escape(subject)
                    clause_patterns = [
                        rf"\b{subj_pat}\b\s+is\s+(.{{8,260}}?)(?:[.!?]|$)",
                        rf"\b{subj_pat}\b\s+refers to\s+(.{{8,260}}?)(?:[.!?]|$)",
                        rf"\b{subj_pat}\b\s+means\s+(.{{8,260}}?)(?:[.!?]|$)",
                        rf"\b{subj_pat}\b\s+defined as\s+(.{{8,260}}?)(?:[.!?]|$)",
                    ]
                    for pat in clause_patterns:
                        for m in re.finditer(pat, normalized_text, flags=re.I):
                            head = re.search(r"\b(is|refers to|means|defined as)\b", m.group(0), flags=re.I)
                            if head:
                                candidate_spans.append(m.group(0).strip())
                # Split noisy page-level blocks into candidate definition-like lines/sentences.
                spans = [
                    re.sub(r"\s+", " ", s).strip()
                    for s in re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
                    if (s or "").strip()
                ]
                for span in candidate_spans + spans:
                    span_l = span.lower()
                    if len(span.split()) < 6:
                        continue
                    if not any(m in span_l for m in (" is ", " are ", " refers to ", " defined as ", " means ")):
                        continue
                    if subject_tokens and not (set(re.findall(r"[a-z0-9]+", span_l)) & subject_tokens):
                        continue
                    score = (
                        0.6 * overlap_score(span)
                        + 0.4 * float(h.get("score", 0.0))
                        + (0.15 if span_l.startswith(subject) else 0.0)
                    )
                    ranked_defs.append((score, span))
            if ranked_defs:
                ranked_defs.sort(key=lambda x: x[0], reverse=True)
                if ranked_defs[0][0] >= 0.35:
                    best = ranked_defs[0][1]
                    if not re.search(r"[.!?]$", best):
                        best = f"{best}."
                    return best
            return None
        ranked = sorted(
            factual_hits,
            key=lambda x: (overlap_score(x["text"]), x.get("score", 0.0)),
            reverse=True,
        )
        return ranked[0]["text"] if ranked else None

    @staticmethod
    def _session_scope_hash(
        session_uploads: list[dict],
        session_documents: list[str],
        retrieval_mode: str,
    ) -> str:
        digest = hashlib.md5()
        digest.update((retrieval_mode or "").encode("utf-8"))
        for upload in session_uploads or []:
            name = str(upload.get("name", ""))
            b = upload.get("bytes")
            digest.update(name.encode("utf-8", errors="ignore"))
            if isinstance(b, (bytes, bytearray)):
                digest.update(hashlib.md5(bytes(b)).digest())
        if not session_uploads:
            for doc in session_documents or []:
                digest.update(hashlib.md5((doc or "")[:3000].encode("utf-8", errors="ignore")).digest())
        return digest.hexdigest()

    @staticmethod
    def _retrieve_session_factual_sentences(session_context_docs: list[str], query: str, top_n: int = 20) -> list[dict]:
        q = (query or "").lower().strip()
        is_definition_query = bool(re.match(r"^(what is|what was|define)\b", q))
        q_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "in",
            "on", "at", "and", "or", "do", "does", "did", "how", "what", "who",
            "when", "where", "which", "many",
        }
        q_tokens = {t for t in q_tokens if t not in stop}
        candidates: list[dict] = []
        for doc_idx, doc in enumerate(session_context_docs or []):
            for sent in re.split(r"(?<=[.!?])\s+", doc or ""):
                text = (sent or "").strip()
                if len(text) < 20:
                    continue
                s_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
                overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                if overlap <= 0:
                    continue
                score = float(overlap)
                if is_definition_query:
                    text_l = text.lower()
                    definition_markers = (" is ", " are ", " refers to ", " defined as ", " means ")
                    if any(m in text_l for m in definition_markers):
                        score += 0.25
                    # Penalize heading-like fragments for definition prompts.
                    if len(text.split()) < 7:
                        score -= 0.2
                    if not re.search(r"[.!?]$", text):
                        score -= 0.1
                candidates.append(
                    {
                        "text": text,
                        "source": f"session_upload_{doc_idx + 1}",
                        "score": round(score, 4),
                    }
                )
        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates[: max(1, int(top_n))]

    @staticmethod
    def _build_uncompressed_context(docs, max_chars: int = 20000) -> str:
        parts = []
        total = 0
        for d in docs or []:
            text = getattr(d, "page_content", d if isinstance(d, str) else str(d))
            if not text:
                continue
            chunk = str(text).strip()
            if not chunk:
                continue
            if total + len(chunk) > max_chars:
                remain = max_chars - total
                if remain > 0:
                    parts.append(chunk[:remain])
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _inject_session_priority_context(compression: dict, priority_texts: list[str], limit: int = 2) -> dict:
        if not compression or not priority_texts:
            return compression
        compressed = (compression.get("compressed_context") or "").strip()
        if not compressed:
            return compression

        additions = []
        compressed_l = compressed.lower()
        for txt in priority_texts:
            snippet = (txt or "").strip()
            if not snippet:
                continue
            if snippet.lower() in compressed_l:
                continue
            additions.append(snippet)
            if len(additions) >= max(1, int(limit)):
                break
        if not additions:
            return compression

        merged = " ".join(additions + [compressed]).strip()
        compression["compressed_context"] = merged
        compression["selected_sentences"] = additions + list(compression.get("selected_sentences") or [])
        compression["compressed_token_count"] = len(merged.split())
        original_token_count = int(compression.get("original_token_count") or 0)
        if original_token_count > 0:
            ratio = (original_token_count - compression["compressed_token_count"]) / original_token_count
            compression["compression_ratio"] = round(ratio, 4)
        meta = compression.get("compression_metadata") or {}
        meta["session_priority_hits_injected"] = len(additions)
        compression["compression_metadata"] = meta
        return compression

    def run(
        self,
        query: str,
        session_documents: list[str] | None = None,
        session_uploads: list[dict] | None = None,
        session_retrieval_mode: str | None = None,
    ) -> dict:
        """Execute one full RAG pass and return answer + diagnostics payload."""
        run_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        normalized_query, corrected_query = self._normalize_query_text(query)
        if corrected_query and corrected_query != normalized_query:
            print(f"Spell-corrected query: {corrected_query}")
        elif normalized_query and normalized_query != (query or "").strip().lower():
            print(f"Normalized query: {normalized_query}")
        session_documents = [d for d in (session_documents or []) if isinstance(d, str) and d.strip()]
        session_uploads = [u for u in (session_uploads or []) if isinstance(u, dict)]
        has_session_docs = len(session_documents) > 0 or len(session_uploads) > 0
        retrieval_mode = (session_retrieval_mode or os.getenv("SESSION_RETRIEVAL_MODE", "hybrid")).strip().lower()
        if retrieval_mode not in {"upload_only", "hybrid"}:
            retrieval_mode = "hybrid"
        session_scope_hash = self._session_scope_hash(session_uploads, session_documents, retrieval_mode)
        multimodal_embedding_used = False
        multimodal_embedding_reason = "no_session_upload"
        multimodal_hits_count = 0
        multimodal_session_texts: list[str] = []
        session_index_rebuilt = False
        session_index_chunk_count = 0
        session_embedding_tokens_saved = 0

        if session_uploads:
            target_upload = None
            for upload in session_uploads:
                pdf_bytes = upload.get("bytes")
                page_map = upload.get("pdf_text_by_page") or {}
                if isinstance(pdf_bytes, (bytes, bytearray)) and pdf_bytes:
                    target_upload = upload
                    break
                if isinstance(page_map, dict) and page_map:
                    target_upload = upload
                    break

            if target_upload is None:
                multimodal_embedding_reason = "no_upload_payload"
            else:
                try:
                    upload_bytes = bytes(target_upload.get("bytes") or b"")
                    page_map = target_upload.get("pdf_text_by_page") or {}
                    has_visual_content = (
                        bool(upload_bytes) and detect_visual_pdf(upload_bytes)
                    )
                    gemini_api_key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
                    use_gemini = bool(has_visual_content and gemini_api_key)
                    retriever = SessionMultimodalRetriever(
                        api_key=gemini_api_key,
                        pdf_bytes=upload_bytes,
                        pdf_text_by_page=page_map,
                        use_gemini=use_gemini,
                    )
                    indexed_count = retriever.build_index()
                    session_index_chunk_count = indexed_count
                    session_index_rebuilt = not retriever.last_build_reused_cache
                    if retriever.last_build_reused_cache:
                        # Rough savings estimate for skipped embedding calls on cache hit.
                        session_embedding_tokens_saved = int(
                            indexed_count * float(retriever.avg_chunk_tokens or 0.0)
                        )
                        print(
                            "[SessionMultimodal] "
                            f"reusing index for {retriever.collection_name}, "
                            f"skipping {indexed_count} chunk embeddings"
                        )
                    multimodal_hits = retriever.retrieve(corrected_query or normalized_query or query)
                    multimodal_session_texts = [
                        (h.get("text") or "").strip()
                        for h in (multimodal_hits or [])
                        if isinstance(h, dict) and (h.get("text") or "").strip()
                    ]
                    multimodal_hits_count = len(multimodal_session_texts)
                    multimodal_embedding_used = multimodal_hits_count > 0
                    if use_gemini:
                        multimodal_embedding_reason = "pdf_contains_images"
                    elif has_visual_content and not gemini_api_key:
                        multimodal_embedding_reason = "visual_pdf_bge_fallback_no_key"
                    else:
                        multimodal_embedding_reason = "text_only_bge"
                    print(
                        "[SessionMultimodal] "
                        f"indexed={indexed_count} | hits={multimodal_hits_count} | "
                        f"use_gemini={use_gemini}"
                    )
                except SessionMultimodalError as exc:
                    multimodal_embedding_reason = "session_retriever_failed"
                    print(f"[SessionMultimodal] disabled due to retriever failure: {exc}")

        # Fast path — cache lookup before analyzer/retrieval/compression/router.
        if has_session_docs:
            for candidate_query in self._build_query_variants(query, normalized_query, corrected_query):
                ck = (session_scope_hash, candidate_query.lower().strip())
                cached_result = self._session_query_cache.get(ck)
                if cached_result is not None:
                    out = dict(cached_result)
                    out["cache_hit"] = True
                    out["cache_similarity"] = 1.0
                    diag = dict(out.get("retrieval_diagnostics") or {})
                    diag["cache_hit"] = True
                    out["retrieval_diagnostics"] = diag
                    print("[SessionCache] Exact-match cache hit for current upload session.")
                    return out
        else:
            for candidate_query in self._build_query_variants(query, normalized_query, corrected_query):
                cached_result = self.semantic_cache.lookup(candidate_query, self.corpus_hash)
                if cached_result is not None:
                    print(
                        f"[SemanticCache] Cache hit (similarity={cached_result.get('cache_similarity')})"
                    )
                    return cached_result

        # Step 1 — Classify query complexity
        t0 = time.perf_counter()
        analysis = self.analyzer.analyze(corrected_query or normalized_query or query)
        query_analyzer_ms = round((time.perf_counter() - t0) * 1000, 2)
        complexity = analysis["complexity_label"]
        print(
            f"Complexity: {complexity} "
            f"(confidence={analysis['confidence']}, source={analysis['source']}, llm_status={analysis['llm_status']})"
        )

        # Step 2 — Retrieve relevant chunks (optional analyzer-triggered factual fast path, then optional HyDE)
        t0 = time.perf_counter()
        retrieval_query = corrected_query or normalized_query or query
        reason_codes = set(analysis.get("reason_codes") or [])
        complexity_score_val = float(analysis.get("complexity_score") or 0.0)
        factual_fastpath_attempted = False
        factual_fastpath_used = False
        factual_fastpath_hits = []
        factual_fastpath_answer = None
        session_corpus_supplement_used = False
        session_corpus_supplement_count = 0
        should_factual_fastpath = (
            not has_session_docs
            and ("simple_starter" in reason_codes)
            and (complexity_score_val < 0.30)
            and self._is_factual_query(normalized_query or query)
        )
        if should_factual_fastpath:
            factual_fastpath_attempted = True
            factual_fastpath_hits = self.retriever.retrieve_factual_sentences(
                normalized_query or query,
                top_n=20,
            )
            factual_fastpath_answer = self._extractive_answer(normalized_query or query, factual_fastpath_hits)
            if factual_fastpath_answer:
                factual_fastpath_used = True
                retrieval = {
                    "docs": [],
                    "k": 0,
                    "complexity_used": complexity,
                    "complexity_score_used": round(complexity_score_val, 2),
                    "analyzer_confidence": analysis.get("confidence"),
                    "k_base": 0,
                    "k_final": 0,
                    "coverage_score": 1.0,
                    "coverage_threshold_used": None,
                    "retry_confidence_gate_used": None,
                    "retrieval_retry": False,
                    "chunks": [],
                    "rerank_enabled": False,
                    "rerank_used": False,
                    "rerank_model": "",
                    "rerank_latency_ms": 0.0,
                }
                print("[FactualFastPath] Using sentence-index extraction path; skipping Chroma retrieval.")
            else:
                print("[FactualFastPath] No strong extractive hit; falling back to standard retrieval.")
        if not factual_fastpath_used:
            if has_session_docs and retrieval_mode == "upload_only":
                retrieval = {
                    "docs": [],
                    "k": 0,
                    "complexity_used": complexity,
                    "complexity_score_used": round(complexity_score_val, 2),
                    "analyzer_confidence": analysis.get("confidence"),
                    "k_base": 0,
                    "k_final": 0,
                    "coverage_score": 1.0,
                    "coverage_threshold_used": None,
                    "retry_confidence_gate_used": None,
                    "retrieval_retry": False,
                    "chunks": [],
                    "rerank_enabled": False,
                    "rerank_used": False,
                    "rerank_model": "",
                    "rerank_latency_ms": 0.0,
                }
                print("[SessionUploadMode] upload_only enabled; skipping Notre Dame corpus retrieval.")
            else:
                retrieval = self.retriever.retrieve(
                    query=retrieval_query,
                    complexity=complexity,
                    complexity_score=analysis.get("complexity_score"),
                    analyzer_confidence=analysis.get("confidence"),
                )
        hyde_attempted = False
        hyde_used = False
        hyde_query = ""
        hyde_coverage = None
        if (
            self.USE_HYDE
            and not has_session_docs
            and not factual_fastpath_used
            and self._is_factual_query(normalized_query or query)
        ):
            hyde_attempted = True
            hyde_query = self._generate_hyde_text(normalized_query or query, complexity)
            if hyde_query:
                hyde_retrieval = self.retriever.retrieve(
                    query=hyde_query,
                    complexity=complexity,
                    complexity_score=analysis.get("complexity_score"),
                    analyzer_confidence=analysis.get("confidence"),
                )
                hyde_coverage = hyde_retrieval.get("coverage_score")
                base_cov = retrieval.get("coverage_score", 0.0)
                if (hyde_coverage or 0.0) >= (base_cov + self.HYDE_MIN_COVERAGE_GAIN):
                    retrieval = hyde_retrieval
                    hyde_used = True
                    print(
                        f"[HyDE] adopted alternate retrieval query "
                        f"(coverage {round(base_cov, 4)} -> {round(hyde_coverage or 0.0, 4)})."
                    )
                else:
                    print(
                        f"[HyDE] not adopted "
                        f"(base coverage={round(base_cov, 4)}, hyde coverage={round(hyde_coverage or 0.0, 4)})."
                    )
            else:
                print("[HyDE] no alternate text generated; using original retrieval.")

        retrieval_query_used = hyde_query if hyde_used else retrieval_query
        retrieval = {
            **retrieval,
            "hyde_attempted": hyde_attempted,
            "hyde_used": hyde_used,
            "hyde_query_preview": (hyde_query[:200] if hyde_query else ""),
            "hyde_coverage_score": hyde_coverage,
            "retrieval_query_used": retrieval_query_used,
            "factual_fastpath_attempted": factual_fastpath_attempted,
            "factual_fastpath_used": factual_fastpath_used,
        }
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 2)
        corpus_retrieval_docs = list(retrieval.get("docs") or [])
        docs = list(corpus_retrieval_docs)
        session_context_docs = list(multimodal_session_texts) + list(session_documents)
        if session_context_docs:
            docs = docs + session_context_docs
            print(
                f"Added {len(session_context_docs)} session-level context chunk(s) "
                f"(uploads={len(session_documents)}, multimodal_hits={len(multimodal_session_texts)})."
            )
            session_chunks = []
            base_idx = len(retrieval["chunks"])
            for i, _ in enumerate(multimodal_session_texts):
                session_chunks.append(
                    {
                        "chunk_index": base_idx + i,
                        "source": f"session_multimodal_hit_{i+1}",
                        "chroma_distance": "",
                        "similarity_score": "",
                    }
                )
            start = len(multimodal_session_texts)
            for i, _ in enumerate(session_documents):
                session_chunks.append(
                    {
                        "chunk_index": base_idx + start + i,
                        "source": f"session_upload_{i+1}",
                        "chroma_distance": "",
                        "similarity_score": "",
                    }
                )
            retrieval["chunks"] = retrieval["chunks"] + session_chunks
        print(
            f"Retrieved {retrieval['k']} chunks "
            f"(k_base={retrieval['k_base']}, coverage={retrieval['coverage_score']}, retry={retrieval['retrieval_retry']}):"
        )
        for c in retrieval["chunks"]:
            _d = c.get("chroma_distance", c.get("similarity_score", ""))
            print(f"  [{c['chunk_index']}] {c['source']} (chroma distance: {_d}, lower is better)")
        if retrieval.get("hyde_attempted"):
            print(
                f"[HyDE] attempted={retrieval.get('hyde_attempted')} used={retrieval.get('hyde_used')} "
                f"alt_coverage={retrieval.get('hyde_coverage_score')}"
            )

        # Step 3/4 — Factual extraction-first path OR regular compression+router path
        use_factual_branch = False
        factual_hits = []
        extracted_answer = None
        factual_query_check = self._is_factual_query(normalized_query or query)
        session_extractive_allowed = (
            has_session_docs
            and retrieval_mode == "upload_only"
            and self._is_session_extractive_query(normalized_query or query)
        )
        factual_extraction_attempted = factual_fastpath_attempted or (
            (not has_session_docs and factual_query_check)
            or session_extractive_allowed
        )
        if factual_fastpath_used:
            factual_hits = factual_fastpath_hits
            extracted_answer = factual_fastpath_answer
            use_factual_branch = True
        elif (not has_session_docs) and factual_query_check:
            factual_hits = self.retriever.retrieve_factual_sentences(normalized_query or query, top_n=20)
            extracted_answer = self._extractive_answer(normalized_query or query, factual_hits)
            if extracted_answer:
                use_factual_branch = True
        elif session_extractive_allowed:
            factual_hits = self._retrieve_session_factual_sentences(
                session_context_docs,
                normalized_query or query,
                top_n=20,
            )
            extracted_answer = self._extractive_answer(normalized_query or query, factual_hits)
            if extracted_answer:
                use_factual_branch = True

        if use_factual_branch:
            t0 = time.perf_counter()
            factual_context = "\n".join(h["text"] for h in factual_hits)
            orig_tok = len(factual_context.split())
            out_tok = len(extracted_answer.split())
            compression = {
                "compressed_context": factual_context,
                "selected_sentences": [h["text"] for h in factual_hits],
                "all_sentences": [h["text"] for h in factual_hits],
                "sentence_scores": [{"sentence": h["text"], "final_score": h["score"]} for h in factual_hits],
                "selected_indices": list(range(len(factual_hits))),
                "original_text": factual_context,
                "original_token_count": orig_tok,
                "compressed_token_count": orig_tok,
                "compression_ratio": 0.0,
                "compression_metadata": {
                    "mode": "factual_sentence_index_extractive",
                    "top_n": len(factual_hits),
                },
            }
            compression_ms = round((time.perf_counter() - t0) * 1000, 2)
            router_result = {
                "answer": extracted_answer,
                "model_used": "extractive-factual",
                "model_source": "extractive",
                "fallback_reason": None,
                "complexity": complexity,
                "input_tokens": orig_tok,
                "output_tokens": out_tok,
                "time_taken_seconds": 0.0,
            }
            model_router_ms = 0.0
            print("Using factual extraction-first path (sentence index, no compression).")
            print(f"Factual hits: {len(factual_hits)} | Extracted answer: {extracted_answer}")
        else:
            # Step 3 — Compress context
            t0 = time.perf_counter()
            compression_complexity_score = analysis.get("complexity_score")
            if has_session_docs:
                try:
                    compression_complexity_score = max(
                        float(compression_complexity_score or 0.0),
                        self.SESSION_UPLOAD_MIN_COMPLEXITY_SCORE,
                    )
                except (TypeError, ValueError):
                    compression_complexity_score = self.SESSION_UPLOAD_MIN_COMPLEXITY_SCORE
            compression = self.compressor.compress(
                corrected_query or normalized_query or query,
                docs,
                complexity,
                complexity_score=compression_complexity_score,
                coverage_score=retrieval.get("coverage_score"),
            )
            if has_session_docs and multimodal_session_texts:
                compression = self._inject_session_priority_context(
                    compression,
                    multimodal_session_texts,
                    limit=self.SESSION_PRIORITY_HITS,
                )
            compression_ms = round((time.perf_counter() - t0) * 1000, 2)
            _cm = compression.get("compression_metadata") or {}
            print(f"Tokens: {compression['original_token_count']} → {compression['compressed_token_count']} (ratio: {compression['compression_ratio']})")
            print(
                f"Sentences kept: {compression['selected_indices']} of {len(compression['all_sentences'])} total "
                f"(budget={_cm.get('adaptive_top_n', '?')}, embed={_cm.get('used_embeddings', '?')})"
            )

            # Step 4 — Route to appropriate model and generate answer
            t0 = time.perf_counter()
            router_result = self.router.route(corrected_query or normalized_query or query, complexity, compression)
            model_router_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"Model used: {router_result['model_used']} | Input tokens: {router_result['input_tokens']} | Output tokens: {router_result['output_tokens']} | Time: {router_result['time_taken_seconds']}s")
            print(f"Initial answer: {router_result['answer']}")

        # Step 5 — Confidence check and retry if needed
        t_conf = time.perf_counter()
        confidence_result = check_confidence(
            query=corrected_query or normalized_query or query,
            compressed_context=compression["compressed_context"],
            router_output=router_result,
            analyzer_output=analysis,
            context_sentences=compression.get("selected_sentences"),
            context_sentence_embeddings=compression.get("_selected_sentence_embeddings"),
            allow_retry=not (has_session_docs and retrieval_mode == "upload_only"),
        )
        if (
            has_session_docs
            and retrieval_mode == "upload_only"
            and router_result.get("model_source") != "extractive"
            and confidence_result.get("model_used_final") != router_result.get("model_used")
        ):
            # In upload-only mode, keep local answer quality stable if stronger-model retry
            # reroutes away from local model (often due unavailable networked fallback).
            confidence_result["final_answer"] = router_result.get("answer", confidence_result.get("final_answer", ""))
            confidence_result["model_used_final"] = router_result.get("model_used", confidence_result.get("model_used_final", ""))
            confidence_result["retried"] = False
            confidence_result["retry_reason"] = (confidence_result.get("retry_reason") or "") + " | kept_local_after_retry_failure"
            confidence_result["retry_reason"] = confidence_result["retry_reason"].strip(" |")
            confidence_result["confidence_score_final"] = float(
                confidence_result.get("confidence_score_original", confidence_result.get("confidence_score_final", 0.0))
            )
        if (
            has_session_docs
            and retrieval_mode == "hybrid"
            and multimodal_session_texts
            and confidence_result.get("confidence_score_final", 1.0) < 0.50
            and corpus_retrieval_docs
            and not use_factual_branch
        ):
            supplement_docs = list(corpus_retrieval_docs[:3])
            if supplement_docs:
                session_corpus_supplement_used = True
                session_corpus_supplement_count = len(supplement_docs)
                print(
                    "[SessionUploadMode] Low-confidence session answer; "
                    f"supplementing with {session_corpus_supplement_count} corpus chunk(s)."
                )
                supplemented_docs = docs + supplement_docs
                compression = self.compressor.compress(
                    corrected_query or normalized_query or query,
                    supplemented_docs,
                    complexity,
                    complexity_score=compression_complexity_score,
                    coverage_score=retrieval.get("coverage_score"),
                )
                if multimodal_session_texts:
                    compression = self._inject_session_priority_context(
                        compression,
                        multimodal_session_texts,
                        limit=self.SESSION_PRIORITY_HITS,
                    )
                router_result = self.router.route(
                    corrected_query or normalized_query or query,
                    complexity,
                    compression,
                )
                confidence_result = check_confidence(
                    query=corrected_query or normalized_query or query,
                    compressed_context=compression["compressed_context"],
                    router_output=router_result,
                    analyzer_output=analysis,
                    context_sentences=compression.get("selected_sentences"),
                    context_sentence_embeddings=compression.get("_selected_sentence_embeddings"),
                    allow_retry=not (has_session_docs and retrieval_mode == "upload_only"),
                )
        confidence_scoring_ms = round((time.perf_counter() - t_conf) * 1000, 2)
        robust_retry_ms = 0.0
        deep_openai_ms = 0.0
        grounding_fallback_ms = 0.0
        heavy_fallbacks_used = 0
        low_quality_answer = any(
            marker in (confidence_result.get("final_answer", "").lower())
            for marker in ["does not provide information", "insufficient", "could you clarify", "not enough context"]
        )
        should_retry_robust = (
            not has_session_docs
            and complexity != "simple"
            and (
                (
                    retrieval.get("coverage_score", 1.0) < 0.55
                    and confidence_result.get("confidence_score_final", 1.0) < 0.50
                )
                or (confidence_result.get("confidence_score_final", 1.0) < 0.60 and low_quality_answer)
            )
            and not retrieval.get("retrieval_retry", False)
            and heavy_fallbacks_used < self.MAX_HEAVY_FALLBACKS
        )
        if should_retry_robust:
            t_robust = time.perf_counter()
            heavy_fallbacks_used += 1
            print("Low-quality signal detected. Running typo-tolerant retrieval retry...")
            robust_query = self._build_query_variants(query, normalized_query, corrected_query)[-1]
            robust_retrieval = self.retriever.retrieve(
                query=robust_query,
                complexity=max(complexity, "medium", key=lambda x: {"simple": 0, "medium": 1, "complex": 2}[x]),
                complexity_score=max(float(analysis.get("complexity_score") or 0.0), 0.55),
                analyzer_confidence=0.0,
            )
            robust_docs = robust_retrieval["docs"]
            robust_compression = self.compressor.compress(
                robust_query,
                robust_docs,
                complexity,
                complexity_score=max(float(analysis.get("complexity_score") or 0.0), 0.55),
                coverage_score=robust_retrieval.get("coverage_score"),
            )
            robust_router = self.router.route(robust_query, complexity, robust_compression)
            robust_confidence = check_confidence(
                query=robust_query,
                compressed_context=robust_compression["compressed_context"],
                router_output=robust_router,
                analyzer_output=analysis,
                context_sentences=robust_compression.get("selected_sentences"),
                context_sentence_embeddings=robust_compression.get("_selected_sentence_embeddings"),
                allow_retry=True,
            )
            if robust_confidence.get("confidence_score_final", 0.0) > confidence_result.get("confidence_score_final", 0.0):
                print("Using robust retry result (improved confidence).")
                retrieval = robust_retrieval
                compression = robust_compression
                router_result = robust_router
                confidence_result = robust_confidence
            robust_retry_ms = round((time.perf_counter() - t_robust) * 1000, 2)

        deep_openai_attempted = False
        deep_openai_used = False
        factual_numeric_answer = bool(use_factual_branch and re.match(r"^\d+\b", (extracted_answer or "").strip()))
        deep_openai_low_quality_signal = (
            confidence_result.get("confidence_score_final", 1.0) < 0.55
            or low_quality_answer
        )
        deep_openai_api_available = bool(os.getenv("OPENAI_API_KEY", "").strip())
        deep_openai_base_eligible = (
            not has_session_docs
            and not factual_numeric_answer
            and deep_openai_low_quality_signal
            and deep_openai_api_available
            and heavy_fallbacks_used < self.MAX_HEAVY_FALLBACKS
        )
        deep_openai_coverage = float(retrieval.get("coverage_score", 0.0) or 0.0)
        should_deep_openai = (
            deep_openai_base_eligible
            and deep_openai_coverage >= self.DEEP_OPENAI_MIN_COVERAGE
        )
        if deep_openai_base_eligible and not should_deep_openai:
            print(
                "[Pipeline] Skipping deep OpenAI fallback — "
                f"coverage_score={round(deep_openai_coverage, 4)} below "
                f"DEEP_OPENAI_MIN_COVERAGE={self.DEEP_OPENAI_MIN_COVERAGE}. "
                "Corpus likely lacks relevant content for this query."
            )
        if should_deep_openai:
            t_deep_openai = time.perf_counter()
            heavy_fallbacks_used += 1
            deep_openai_attempted = True
            print("Low-quality signal persists. Running deep retrieval with OpenAI and no compression...")
            try:
                deep_retrieval = self.retriever.retrieve(
                    query=corrected_query or normalized_query or query,
                    complexity="complex",
                    complexity_score=1.0,
                    analyzer_confidence=0.0,
                )
                deep_context = self._build_uncompressed_context(deep_retrieval.get("docs") or [])
                deep_compression = {
                    "compressed_context": deep_context,
                    "all_sentences": [],
                    "sentence_scores": [],
                    "selected_indices": [],
                    "original_token_count": 0,
                    "compressed_token_count": 0,
                    "compression_ratio": 0.0,
                    "compression_metadata": {"mode": "deep_openai_no_compression"},
                }
                deep_router = self.router.route(
                    corrected_query or normalized_query or query,
                    "complex",
                    deep_compression,
                )
                deep_confidence = check_confidence(
                    query=corrected_query or normalized_query or query,
                    compressed_context=deep_context,
                    router_output=deep_router,
                    analyzer_output=analysis,
                    allow_retry=True,
                )
                deep_grounded_ok, _ = self._grounding_gate(
                    deep_confidence.get("final_answer", ""),
                    deep_context,
                )
                if deep_grounded_ok and deep_confidence.get("confidence_score_final", 0.0) >= confidence_result.get("confidence_score_final", 0.0):
                    print("Using deep OpenAI fallback result.")
                    retrieval = deep_retrieval
                    compression = deep_compression
                    router_result = deep_router
                    confidence_result = deep_confidence
                    deep_openai_used = True
            except Exception as exc:
                print(f"[Pipeline] Deep OpenAI fallback failed: {type(exc).__name__}: {exc}")
            deep_openai_ms = round((time.perf_counter() - t_deep_openai) * 1000, 2)

        t_grounding = time.perf_counter()
        grounded_ok, grounding_meta = self._grounding_gate(
            confidence_result.get("final_answer", ""),
            compression.get("compressed_context", ""),
        )
        if not grounded_ok and router_result.get("model_source") == "extractive":
            print("[Pipeline] Extractive answer failed grounding; retrying with model generation.")
            fallback_complexity_score = analysis.get("complexity_score")
            if has_session_docs:
                try:
                    fallback_complexity_score = max(
                        float(fallback_complexity_score or 0.0),
                        self.SESSION_UPLOAD_MIN_COMPLEXITY_SCORE,
                    )
                except (TypeError, ValueError):
                    fallback_complexity_score = self.SESSION_UPLOAD_MIN_COMPLEXITY_SCORE
            fallback_compression = self.compressor.compress(
                corrected_query or normalized_query or query,
                docs,
                complexity,
                complexity_score=fallback_complexity_score,
                coverage_score=retrieval.get("coverage_score"),
            )
            if has_session_docs and multimodal_session_texts:
                fallback_compression = self._inject_session_priority_context(
                    fallback_compression,
                    multimodal_session_texts,
                    limit=self.SESSION_PRIORITY_HITS,
                )
            fallback_router = self.router.route(
                corrected_query or normalized_query or query,
                complexity,
                fallback_compression,
            )
            fallback_confidence = check_confidence(
                query=corrected_query or normalized_query or query,
                compressed_context=fallback_compression["compressed_context"],
                router_output=fallback_router,
                analyzer_output=analysis,
                context_sentences=fallback_compression.get("selected_sentences"),
                context_sentence_embeddings=fallback_compression.get("_selected_sentence_embeddings"),
                allow_retry=not (has_session_docs and retrieval_mode == "upload_only"),
            )
            fallback_grounded_ok, fallback_grounding_meta = self._grounding_gate(
                fallback_confidence.get("final_answer", ""),
                fallback_compression.get("compressed_context", ""),
            )
            if fallback_grounded_ok:
                compression = fallback_compression
                router_result = fallback_router
                confidence_result = fallback_confidence
                grounded_ok = fallback_grounded_ok
                grounding_meta = fallback_grounding_meta
        openai_fallback_attempted = False
        openai_fallback_used = False
        if not grounded_ok:
            can_escalate_openai = (
                router_result.get("model_source") == "local_openai_compatible"
                and not (has_session_docs and retrieval_mode == "upload_only")
                and bool(os.getenv("OPENAI_API_KEY", "").strip())
                and heavy_fallbacks_used < self.MAX_HEAVY_FALLBACKS
            )
            if can_escalate_openai:
                heavy_fallbacks_used += 1
                openai_fallback_attempted = True
                try:
                    openai_router_result = self.router.route(
                        corrected_query or normalized_query or query,
                        "complex",
                        compression,
                    )
                    openai_confidence = check_confidence(
                        query=corrected_query or normalized_query or query,
                        compressed_context=compression["compressed_context"],
                        router_output=openai_router_result,
                        analyzer_output=analysis,
                        context_sentences=compression.get("selected_sentences"),
                        context_sentence_embeddings=compression.get("_selected_sentence_embeddings"),
                        allow_retry=True,
                    )
                    openai_grounded_ok, openai_grounding_meta = self._grounding_gate(
                        openai_confidence.get("final_answer", ""),
                        compression.get("compressed_context", ""),
                    )
                    if openai_grounded_ok:
                        openai_fallback_used = True
                        router_result = openai_router_result
                        confidence_result = openai_confidence
                        grounded_ok = openai_grounded_ok
                        grounding_meta = openai_grounding_meta
                except Exception as exc:
                    print(f"[Pipeline] OpenAI fallback attempt failed: {type(exc).__name__}: {exc}")

            if not openai_fallback_used:
                if has_session_docs and retrieval_mode == "upload_only" and router_result.get("model_source") != "extractive":
                    print("[Pipeline] Grounding gate soft-fail in upload_only mode; keeping local answer.")
                else:
                    confidence_result["final_answer"] = (
                        "I don't have enough grounded evidence in the retrieved context to answer this reliably. "
                        "Please rephrase the query or provide more specific context."
                    )
                    confidence_result["confidence_score_final"] = min(
                        float(confidence_result.get("confidence_score_final", 0.0)), 0.35
                    )
                    confidence_result["retry_reason"] = (
                        (confidence_result.get("retry_reason") or "") + " | grounding_gate"
                    ).strip(" |")
        grounding_fallback_ms = round((time.perf_counter() - t_grounding) * 1000, 2)
        confidence_checker_ms = round(
            confidence_scoring_ms + robust_retry_ms + deep_openai_ms + grounding_fallback_ms,
            2,
        )
        breakdown = confidence_result["score_breakdown"]
        print(f"Score breakdown — heuristic: {breakdown['heuristic']} | tfidf: {breakdown['tfidf']} | embedding: {breakdown['embedding']} | semantic: {breakdown['semantic']}")
        print(f"Confidence: {confidence_result['confidence_score_final']} | Retried: {confidence_result['retried']}" + (f" | Reason: {confidence_result['retry_reason']}" if confidence_result["retried"] else ""))
        print(f"Final model: {confidence_result['model_used_final']}")
        print(f"Final answer: {confidence_result['final_answer']}")
        total_pipeline_ms = round((time.perf_counter() - run_start) * 1000, 2)
        print(
            f"Stage timing (ms) — analyzer: {query_analyzer_ms} | retrieval: {retrieval_ms} | "
            f"compression: {compression_ms} | router: {model_router_ms} | "
            f"confidence_checker: {confidence_checker_ms} "
            f"(score={confidence_scoring_ms}, robust={robust_retry_ms}, deep_openai={deep_openai_ms}, grounding={grounding_fallback_ms}) "
            f"| total: {total_pipeline_ms}"
        )

        _chunks = retrieval.get("chunks") or []
        _chunk_distances = []
        for c in _chunks:
            raw = c.get("chroma_distance", c.get("similarity_score", ""))
            try:
                _chunk_distances.append(float(raw))
            except (TypeError, ValueError):
                continue
        _retrieval_avg_chunk_metric = (
            round(
                sum(_chunk_distances) / len(_chunk_distances),
                4,
            )
            if _chunk_distances
            else ""
        )
        _tok_orig = int(compression.get("original_token_count") or 0)
        _tok_comp = int(compression.get("compressed_token_count") or 0)
        _router_in = int(router_result.get("input_tokens") or 0)
        _router_out = int(router_result.get("output_tokens") or 0)
        _model_rerouted = confidence_result.get("model_used_final") != confidence_result.get(
            "model_used_original"
        )

        # --- Return payload (consumed by evaluate.py, logging, experiments) ---
        result = {
            "query": query,
            "complexity": complexity,
            "complexity_source": analysis["source"],
            "complexity_score": analysis.get("complexity_score"),
            "complexity_confidence": analysis["confidence"],
            "complexity_reason_codes": analysis["reason_codes"],
            "query_analyzer_llm_status": analysis["llm_status"],
            "query_word_count": analysis.get("word_count"),
            # minimum useful stage observability
            "query_analyzer_ms": query_analyzer_ms,
            "retrieval_ms": retrieval_ms,
            "compression_ms": compression_ms,
            "model_router_ms": model_router_ms,
            "confidence_checker_ms": confidence_checker_ms,
            "confidence_scoring_ms": confidence_scoring_ms,
            "robust_retry_ms": robust_retry_ms,
            "deep_openai_ms": deep_openai_ms,
            "grounding_fallback_ms": grounding_fallback_ms,
            "total_pipeline_ms": total_pipeline_ms,
            # retrieval metadata
            "k": retrieval["k"],
            "retrieval_complexity_used": retrieval["complexity_used"],
            "retrieval_complexity_score_used": retrieval["complexity_score_used"],
            "retrieval_analyzer_confidence": retrieval["analyzer_confidence"],
            "k_base": retrieval["k_base"],
            "k_final": retrieval["k_final"],
            "coverage_score": retrieval["coverage_score"],
            "retrieval_retry": retrieval["retrieval_retry"],
            "coverage_threshold_used": retrieval.get("coverage_threshold_used"),
            "retrieval_retry_confidence_gate": retrieval.get("retry_confidence_gate_used"),
            "rerank_enabled": retrieval.get("rerank_enabled", False),
            "rerank_used": retrieval.get("rerank_used", False),
            "rerank_model": retrieval.get("rerank_model", ""),
            "rerank_latency_ms": retrieval.get("rerank_latency_ms", 0.0),
            "hyde_attempted": retrieval.get("hyde_attempted", False),
            "hyde_used": retrieval.get("hyde_used", False),
            "hyde_query_preview": retrieval.get("hyde_query_preview", ""),
            "hyde_coverage_score": retrieval.get("hyde_coverage_score"),
            "retrieval_query_used": retrieval.get("retrieval_query_used", retrieval_query),
            "factual_fastpath_attempted": retrieval.get("factual_fastpath_attempted", False),
            "factual_fastpath_used": retrieval.get("factual_fastpath_used", False),
            "retrieval_avg_chunk_distance": _retrieval_avg_chunk_metric,
            "chunks": retrieval["chunks"],
            # compression metadata
            "compressed_context": compression["compressed_context"],
            "all_sentences": compression["all_sentences"],
            "sentence_scores": compression["sentence_scores"],
            "selected_indices": compression["selected_indices"],
            "original_token_count": compression["original_token_count"],
            "compressed_token_count": compression["compressed_token_count"],
            "compression_ratio": compression["compression_ratio"],
            "compression_metadata": compression.get("compression_metadata") or {},
            "compression_sentence_count": len(compression.get("all_sentences") or []),
            "compression_selected_sentence_count": len(compression.get("selected_indices") or []),
            "compression_tokens_saved": _tok_orig - _tok_comp,
            "pipeline_tokens_context_to_llm_est": _tok_comp,
            "pipeline_tokens_llm_total_est": _router_in + _router_out,
            # router metadata
            "model_used_original": router_result["model_used"],
            "router_model_source": router_result.get("model_source"),
            "router_fallback_reason": router_result.get("fallback_reason") or "",
            "input_tokens": router_result["input_tokens"],
            "output_tokens": router_result["output_tokens"],
            "router_total_tokens_est": _router_in + _router_out,
            "time_taken_seconds": router_result["time_taken_seconds"],
            # confidence metadata
            "score_breakdown": confidence_result["score_breakdown"],
            "confidence_score_original": confidence_result["confidence_score_original"],
            "confidence_score_final": confidence_result["confidence_score_final"],
            "confidence_semantic": confidence_result.get("confidence_semantic"),
            "confidence_threshold": confidence_result.get("confidence_threshold"),
            "confidence_checker_embedding_enabled": confidence_result.get("confidence_checker_embedding_enabled"),
            "retried": confidence_result["retried"],
            "retry_reason": confidence_result.get("retry_reason") or "",
            "model_used_final": confidence_result["model_used_final"],
            "model_rerouted": _model_rerouted,
            "final_answer": confidence_result["final_answer"],
            "grounding_gate_passed": grounded_ok,
            "grounding_gate_reason": grounding_meta.get("reason"),
            "grounding_gate_unsupported_entities": grounding_meta.get("unsupported_entities"),
            "openai_fallback_attempted": openai_fallback_attempted,
            "openai_fallback_used": openai_fallback_used,
            "deep_openai_attempted": deep_openai_attempted,
            "deep_openai_used": deep_openai_used,
            "max_heavy_fallbacks": self.MAX_HEAVY_FALLBACKS,
            "heavy_fallbacks_used": heavy_fallbacks_used,
            "multimodal_embedding_used": multimodal_embedding_used,
            "multimodal_embedding_reason": multimodal_embedding_reason,
            "multimodal_hits_count": multimodal_hits_count,
            "session_retrieval_mode": retrieval_mode,
            "session_corpus_supplement_used": session_corpus_supplement_used,
            "session_corpus_supplement_count": session_corpus_supplement_count,
            "session_index_rebuilt": session_index_rebuilt,
            "session_index_chunk_count": session_index_chunk_count,
            "session_embedding_tokens_saved": session_embedding_tokens_saved,
            "cache_hit": False,
        }
        top_factual_hits = [
            {
                "text": (h.get("text") or "")[:180],
                "source": h.get("source", "unknown"),
                "score": h.get("score", 0.0),
            }
            for h in (factual_hits or [])[:5]
        ]
        result["retrieval_diagnostics"] = {
            "factual_extraction_attempted": factual_extraction_attempted,
            "factual_extraction_used": use_factual_branch,
            "factual_hit_count": len(factual_hits or []),
            "factual_top_hits": top_factual_hits,
            "compression_mode": (compression.get("compression_metadata") or {}).get("mode", "standard"),
            "k_base": retrieval.get("k_base"),
            "k_final": retrieval.get("k_final"),
            "coverage_score": retrieval.get("coverage_score"),
            "retrieval_retry": retrieval.get("retrieval_retry"),
            "rerank_enabled": retrieval.get("rerank_enabled", False),
            "rerank_used": retrieval.get("rerank_used", False),
            "rerank_model": retrieval.get("rerank_model", ""),
            "rerank_latency_ms": retrieval.get("rerank_latency_ms", 0.0),
            "hyde_attempted": retrieval.get("hyde_attempted", False),
            "hyde_used": retrieval.get("hyde_used", False),
            "hyde_coverage_score": retrieval.get("hyde_coverage_score"),
            "factual_fastpath_attempted": retrieval.get("factual_fastpath_attempted", False),
            "factual_fastpath_used": retrieval.get("factual_fastpath_used", False),
            "retrieval_sources_top5": [
                {
                    "source": c.get("source", "unknown"),
                    "distance": c.get("chroma_distance", c.get("similarity_score", "")),
                }
                for c in (retrieval.get("chunks") or [])[:5]
            ],
            "grounding_gate_passed": grounded_ok,
            "grounding_gate_reason": grounding_meta.get("reason"),
            "openai_fallback_attempted": openai_fallback_attempted,
            "openai_fallback_used": openai_fallback_used,
            "deep_openai_attempted": deep_openai_attempted,
            "deep_openai_used": deep_openai_used,
            "max_heavy_fallbacks": self.MAX_HEAVY_FALLBACKS,
            "heavy_fallbacks_used": heavy_fallbacks_used,
            "multimodal_embedding_used": multimodal_embedding_used,
            "multimodal_embedding_reason": multimodal_embedding_reason,
            "multimodal_hits_count": multimodal_hits_count,
            "session_retrieval_mode": retrieval_mode,
            "session_corpus_supplement_used": session_corpus_supplement_used,
            "session_corpus_supplement_count": session_corpus_supplement_count,
            "session_index_rebuilt": session_index_rebuilt,
            "session_index_chunk_count": session_index_chunk_count,
            "session_embedding_tokens_saved": session_embedding_tokens_saved,
            "cache_hit": False,
        }
        if has_session_docs:
            for candidate_query in self._build_query_variants(query, normalized_query, corrected_query):
                ck = (session_scope_hash, candidate_query.lower().strip())
                self._session_query_cache[ck] = dict(result)
        else:
            self.semantic_cache.store(query, result, self.corpus_hash)
        return result
