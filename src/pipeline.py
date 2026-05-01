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
    PROTECTED_FACTUAL_PREFIXES = (
        "how many ",
        "who ",
        "when ",
        "where ",
        "which ",
        "what is ",
        "what was ",
    )

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
        unsupported = [e for e in entities if e not in ctx]

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
        ranked = sorted(
            factual_hits,
            key=lambda x: (overlap_score(x["text"]), x.get("score", 0.0)),
            reverse=True,
        )
        return ranked[0]["text"] if ranked else None

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

    def run(self, query: str, session_documents: list[str] | None = None) -> dict:
        run_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        normalized_query, corrected_query = self._normalize_query_text(query)
        if corrected_query and corrected_query != normalized_query:
            print(f"Spell-corrected query: {corrected_query}")
        elif normalized_query and normalized_query != (query or "").strip().lower():
            print(f"Normalized query: {normalized_query}")
        session_documents = [d for d in (session_documents or []) if isinstance(d, str) and d.strip()]
        has_session_docs = len(session_documents) > 0

        # Fast path — semantic cache lookup before analyzer/retrieval/compression/router.
        if not has_session_docs:
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
        docs = retrieval["docs"]
        if has_session_docs:
            docs = docs + session_documents
            print(f"Added {len(session_documents)} session-level uploaded document(s) to context.")
            session_chunks = []
            base_idx = len(retrieval["chunks"])
            for i, _ in enumerate(session_documents):
                session_chunks.append(
                    {
                        "chunk_index": base_idx + i,
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
        if factual_fastpath_used:
            factual_hits = factual_fastpath_hits
            extracted_answer = factual_fastpath_answer
            use_factual_branch = True
        elif self._is_factual_query(normalized_query or query):
            factual_hits = self.retriever.retrieve_factual_sentences(
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
            compression = self.compressor.compress(
                corrected_query or normalized_query or query,
                docs,
                complexity,
                complexity_score=analysis.get("complexity_score"),
                coverage_score=retrieval.get("coverage_score"),
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
        openai_fallback_attempted = False
        openai_fallback_used = False
        if not grounded_ok:
            can_escalate_openai = (
                router_result.get("model_source") == "local_openai_compatible"
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
            "factual_extraction_attempted": self._is_factual_query(normalized_query or query),
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
            "cache_hit": False,
        }
        if not has_session_docs:
            self.semantic_cache.store(query, result, self.corpus_hash)
        return result
