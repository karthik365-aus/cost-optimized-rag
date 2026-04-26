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
import re
import time

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

        # Step 2 — Retrieve relevant chunks
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=corrected_query or normalized_query or query,
            complexity=complexity,
            complexity_score=analysis.get("complexity_score"),
            analyzer_confidence=analysis.get("confidence"),
        )
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
        t0 = time.perf_counter()
        confidence_result = check_confidence(
            query=corrected_query or normalized_query or query,
            compressed_context=compression["compressed_context"],
            router_output=router_result,
            analyzer_output=analysis,
        )
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
        )
        if should_retry_robust:
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
        confidence_checker_ms = round((time.perf_counter() - t0) * 1000, 2)
        breakdown = confidence_result["score_breakdown"]
        print(f"Score breakdown — heuristic: {breakdown['heuristic']} | tfidf: {breakdown['tfidf']} | embedding: {breakdown['embedding']} | semantic: {breakdown['semantic']}")
        print(f"Confidence: {confidence_result['confidence_score_final']} | Retried: {confidence_result['retried']}" + (f" | Reason: {confidence_result['retry_reason']}" if confidence_result["retried"] else ""))
        print(f"Final model: {confidence_result['model_used_final']}")
        print(f"Final answer: {confidence_result['final_answer']}")
        total_pipeline_ms = round((time.perf_counter() - run_start) * 1000, 2)
        print(
            f"Stage timing (ms) — analyzer: {query_analyzer_ms} | retrieval: {retrieval_ms} | "
            f"compression: {compression_ms} | router: {model_router_ms} | "
            f"confidence_checker: {confidence_checker_ms} | total: {total_pipeline_ms}"
        )

        _chunks = retrieval.get("chunks") or []
        _retrieval_avg_chunk_metric = (
            round(
                sum(
                    float(
                        c.get("chroma_distance", c.get("similarity_score", 0.0))
                    )
                    for c in _chunks
                )
                / len(_chunks),
                4,
            )
            if _chunks
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
            "cache_hit": False,
        }
        if not has_session_docs:
            self.semantic_cache.store(query, result, self.corpus_hash)
        return result
