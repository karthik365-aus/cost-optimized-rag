"""
Hybrid retrieval over ``chroma_db`` using dense vectors + lexical matching.

Key behavior:
- Rebuilds Chroma when chunking config changes.
- Uses smaller, sentence-friendly chunking defaults.
- Retrieves with MMR (dense) + keyword-overlap (lexical), then merges/reranks.
"""
from pathlib import Path
import json
import os
import re
import shutil
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_experimental.text_splitter import SemanticChunker
except Exception:  # optional dependency; fallback splitter remains available
    SemanticChunker = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = str(_PROJECT_ROOT / "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class AdaptiveRetriever:
    K_BASE_MIN = 5
    K_BASE_MAX = 10
    K_HARD_CAP = 20
    LOW_CONF_BONUS = 2
    RETRY_BONUS = 2
    CANDIDATE_EXTRA = 4

    # Hybrid merge weights (general purpose)
    HYBRID_DENSE_WEIGHT = 0.65
    HYBRID_LEXICAL_WEIGHT = 0.35

    # MMR settings
    MMR_LAMBDA = 0.75
    MMR_FETCH_K_MIN = 12
    MMR_FETCH_K_MAX = 40

    # Chunking defaults (shorter, more semantic windows)
    CHUNK_SIZE = int(os.getenv("RETRIEVER_CHUNK_SIZE", "450"))
    CHUNK_OVERLAP = int(os.getenv("RETRIEVER_CHUNK_OVERLAP", "80"))
    USE_SEMANTIC_CHUNKER = os.getenv("USE_SEMANTIC_CHUNKER", "true").lower() == "true"
    SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")
    SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "85"))
    CHUNKING_VERSION = "v3_semantic_chunker_toggle"

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
        "or", "for", "in", "on", "at", "with", "by", "from", "that", "this",
        "it", "as", "what", "which", "who", "when", "where", "how", "why",
    }
    COVERAGE_THRESHOLDS = {
        "simple": 0.45,
        "medium": 0.55,
        "complex": 0.60,
    }
    RETRY_CONFIDENCE_GATE = 0.75

    def __init__(self, documents_path="data/documents"):
        self.documents_path = str(documents_path)
        self.lexical_entries = []
        self._factual_sentences = []
        self._factual_vectorizer = None
        self._factual_matrix = None
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        chroma_path = Path(CHROMA_DIR)
        if self._should_rebuild_index(chroma_path):
            self._build_vector_db(chroma_path)
        else:
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self._embeddings,
            )
            print("Vector database loaded")

        self._load_lexical_index()
        self._build_factual_sentence_index()

    def retrieve(
        self,
        query: str,
        complexity: str = "medium",
        complexity_score: float = None,
        analyzer_confidence: float = None,
        force_k: int = None,
    ) -> dict:
        normalized_complexity = (complexity or "medium").strip().lower()
        fallback_score_map = {"simple": 0.2, "medium": 0.55, "complex": 0.85}
        score = complexity_score if complexity_score is not None else fallback_score_map.get(normalized_complexity, 0.55)
        score = max(0.0, min(1.0, float(score)))

        k_base = max(self.K_BASE_MIN, min(self.K_BASE_MAX, round(3 + score * 7)))
        if force_k is not None:
            k_forced = max(self.K_BASE_MIN, min(self.K_HARD_CAP, int(force_k)))
            k_base = k_forced
        low_confidence = analyzer_confidence is not None and float(analyzer_confidence) < 0.70
        if force_k is not None:
            k_final = k_base
        else:
            k_final = min(k_base + self.LOW_CONF_BONUS, self.K_HARD_CAP) if low_confidence else k_base
        print(
            f"Retrieving {k_final} chunks for {normalized_complexity} query "
            f"(score={round(score, 2)}, confidence={analyzer_confidence})..."
        )

        docs, chunks_metadata = self._hybrid_retrieve(query=query, k_select=k_final)
        coverage_score = self._keyword_coverage_score(query=query, docs=docs)
        retrieval_retry = False
        coverage_threshold = self.COVERAGE_THRESHOLDS.get(normalized_complexity, 0.50)
        low_confidence_for_retry = analyzer_confidence is None or float(analyzer_confidence) < self.RETRY_CONFIDENCE_GATE

        if force_k is None and coverage_score < coverage_threshold and low_confidence_for_retry and k_final < self.K_HARD_CAP:
            retry_k = min(k_final + self.RETRY_BONUS, self.K_HARD_CAP)
            docs, chunks_metadata = self._hybrid_retrieve(query=query, k_select=retry_k)
            k_final = retry_k
            coverage_score = self._keyword_coverage_score(query=query, docs=docs)
            retrieval_retry = True

        for chunk in chunks_metadata:
            chunk["coverage_score"] = round(float(coverage_score), 4)

        return {
            "docs": docs,
            "k": k_final,
            "complexity_used": normalized_complexity,
            "complexity_score_used": round(score, 2),
            "analyzer_confidence": None if analyzer_confidence is None else round(float(analyzer_confidence), 2),
            "k_base": k_base,
            "k_final": k_final,
            "coverage_score": round(float(coverage_score), 4),
            "coverage_threshold_used": coverage_threshold,
            "retry_confidence_gate_used": self.RETRY_CONFIDENCE_GATE,
            "retrieval_retry": retrieval_retry,
            "chunks": chunks_metadata,
        }

    def _hybrid_retrieve(self, query: str, k_select: int) -> Tuple[List, List]:
        fetch_k = min(self.K_HARD_CAP, max(k_select, k_select + self.CANDIDATE_EXTRA))
        dense_docs, dense_meta = self._run_dense_retrieval(query=query, k=fetch_k)
        lexical_docs, lexical_meta = self._run_lexical_retrieval(query=query, k=fetch_k)
        return self._merge_hybrid(dense_docs, dense_meta, lexical_docs, lexical_meta, k_select)

    def _run_dense_retrieval(self, query: str, k: int) -> Tuple[List, List]:
        fetch_k = min(self.MMR_FETCH_K_MAX, max(self.MMR_FETCH_K_MIN, k * 3))
        mmr_docs = self.vectordb.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=self.MMR_LAMBDA,
        )
        score_candidates = self.vectordb.similarity_search_with_score(query, k=fetch_k)
        score_map = {}
        for doc, score in score_candidates:
            score_map[self._doc_key(doc)] = round(float(score), 4)

        docs = []
        meta = []
        for idx, doc in enumerate(mmr_docs):
            d = score_map.get(self._doc_key(doc), "")
            docs.append(doc)
            meta.append({
                "chunk_index": idx,
                "source": doc.metadata.get("source", "unknown"),
                "chroma_distance": d,
                "similarity_score": d,
                "dense_score": 1.0 / (1.0 + float(d)) if d != "" else 0.0,
                "lexical_score": 0.0,
                "retrieval_mode": "dense_mmr",
            })
        return docs, meta

    def _run_lexical_retrieval(self, query: str, k: int) -> Tuple[List, List]:
        keywords = self._extract_keywords(query)
        if not keywords or not self.lexical_entries:
            return [], []

        scored = []
        for item in self.lexical_entries:
            overlap = sum(1 for tok in keywords if tok in item["tokens"])
            if overlap == 0:
                continue
            lexical_score = overlap / len(keywords)
            scored.append((lexical_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        docs = []
        meta = []
        for idx, (lex_score, item) in enumerate(top):
            text = item["text"]
            docs.append(text)
            meta.append({
                "chunk_index": idx,
                "source": item["source"],
                "chroma_distance": "",
                "similarity_score": "",
                "dense_score": 0.0,
                "lexical_score": round(float(lex_score), 4),
                "retrieval_mode": "lexical",
            })
        return docs, meta

    def _merge_hybrid(
        self,
        dense_docs: List,
        dense_meta: List,
        lexical_docs: List,
        lexical_meta: List,
        k_select: int,
    ) -> Tuple[List, List]:
        merged = {}

        for doc, meta in zip(dense_docs, dense_meta):
            key = self._doc_key_with_source(doc, meta.get("source", "unknown"))
            merged[key] = {"doc": doc, "meta": dict(meta)}

        for doc, meta in zip(lexical_docs, lexical_meta):
            key = self._doc_key_with_source(doc, meta.get("source", "unknown"))
            if key in merged:
                merged[key]["meta"]["lexical_score"] = max(
                    merged[key]["meta"].get("lexical_score", 0.0),
                    meta.get("lexical_score", 0.0),
                )
                merged[key]["meta"]["retrieval_mode"] = "dense+lexical"
            else:
                merged[key] = {"doc": doc, "meta": dict(meta)}

        scored = []
        for payload in merged.values():
            m = payload["meta"]
            combined = (
                self.HYBRID_DENSE_WEIGHT * float(m.get("dense_score", 0.0))
                + self.HYBRID_LEXICAL_WEIGHT * float(m.get("lexical_score", 0.0))
            )
            scored.append((combined, payload["doc"], m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k_select]
        out_docs = []
        out_meta = []
        for idx, (_score, doc, meta) in enumerate(top):
            m = dict(meta)
            m["chunk_index"] = idx
            out_docs.append(doc)
            out_meta.append(m)
        return out_docs, out_meta

    def _keyword_coverage_score(self, query: str, docs: List) -> float:
        keywords = self._extract_keywords(query)
        if not keywords:
            return 1.0
        retrieved_text = " ".join(self._doc_text(doc) for doc in docs).lower()
        matched = sum(1 for kw in keywords if kw in retrieved_text)
        return matched / len(keywords)

    def _extract_keywords(self, query: str) -> set:
        query_tokens = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
        return {tok for tok in query_tokens if len(tok) > 2 and tok not in self.STOPWORDS}

    def _load_lexical_index(self) -> None:
        self.lexical_entries = []
        try:
            raw = self.vectordb.get(include=["documents", "metadatas"])
        except Exception:
            return
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        for i, text in enumerate(docs):
            if not isinstance(text, str):
                continue
            cleaned = text.strip()
            if not cleaned:
                continue
            meta = metas[i] if i < len(metas) else {}
            source = (meta or {}).get("source", "unknown")
            tokens = set(re.findall(r"[a-zA-Z0-9]+", cleaned.lower()))
            self.lexical_entries.append({
                "text": cleaned,
                "source": source,
                "tokens": tokens,
            })

    def _build_factual_sentence_index(self) -> None:
        self._factual_sentences = []
        docs_path = Path(self.documents_path)
        if not docs_path.exists():
            return
        for txt_file in sorted(docs_path.glob("**/*.txt")):
            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            content = re.sub(r"(?m)^Title:.*$", " ", content)
            content = re.sub(r"(?m)^Section::.*$", " ", content)
            for sentence in self._split_sentences(content):
                cleaned = re.sub(r"\s+", " ", sentence).strip()
                if len(cleaned) < 30:
                    continue
                self._factual_sentences.append(
                    {"text": cleaned, "source": str(txt_file)}
                )
        if not self._factual_sentences:
            return
        texts = [x["text"] for x in self._factual_sentences]
        self._factual_vectorizer = TfidfVectorizer(stop_words="english")
        self._factual_matrix = self._factual_vectorizer.fit_transform(texts)

    def retrieve_factual_sentences(self, query: str, top_n: int = 20) -> List[dict]:
        if (
            not self._factual_sentences
            or self._factual_vectorizer is None
            or self._factual_matrix is None
        ):
            return []
        qv = self._factual_vectorizer.transform([query])
        sims = cosine_similarity(qv, self._factual_matrix).flatten()
        if sims.size == 0:
            return []
        query_tokens = set(self._tokenize(query))
        rescored = []
        for idx, sem_score in enumerate(sims):
            item = self._factual_sentences[int(idx)]
            sent_tokens = set(self._tokenize(item["text"]))
            overlap = len(query_tokens & sent_tokens) / max(1, len(query_tokens))
            final_score = (0.7 * float(sem_score)) + (0.3 * overlap)
            if final_score <= 0:
                continue
            rescored.append((final_score, idx))
        rescored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for score, idx in rescored[: max(1, int(top_n))]:
            item = self._factual_sentences[int(idx)]
            hits.append(
                {
                    "text": item["text"],
                    "source": item["source"],
                    "score": round(float(score), 4),
                }
            )
        return hits

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s and s.strip()]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _should_rebuild_index(self, chroma_path: Path) -> bool:
        if not chroma_path.is_dir():
            return True
        cfg_path = chroma_path / "_chunking_config.json"
        expected = self._chunking_config()
        if not cfg_path.exists():
            return True
        try:
            current = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        return current != expected

    def _build_vector_db(self, chroma_path: Path) -> None:
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)
        print("Loading documents...")
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        use_semantic = self.USE_SEMANTIC_CHUNKER
        if use_semantic and SemanticChunker is not None:
            print("Using SemanticChunker (topic-aware splits)...")
            splitter = SemanticChunker(
                self._embeddings,
                breakpoint_threshold_type=self.SEMANTIC_BREAKPOINT_TYPE,
                breakpoint_threshold_amount=self.SEMANTIC_BREAKPOINT_AMOUNT,
            )
        else:
            if use_semantic and SemanticChunker is None:
                print("SemanticChunker unavailable; falling back to RecursiveCharacterTextSplitter.")
            else:
                print("Using RecursiveCharacterTextSplitter (fallback)...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            )
        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Creating vector database...")
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=CHROMA_DIR,
        )
        print("Vector database created")

        cfg_path = Path(CHROMA_DIR) / "_chunking_config.json"
        cfg_path.write_text(json.dumps(self._chunking_config(), indent=2), encoding="utf-8")

    def _chunking_config(self) -> dict:
        return {
            "version": self.CHUNKING_VERSION,
            "use_semantic_chunker": self.USE_SEMANTIC_CHUNKER,
            "semantic_breakpoint_type": self.SEMANTIC_BREAKPOINT_TYPE,
            "semantic_breakpoint_amount": self.SEMANTIC_BREAKPOINT_AMOUNT,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
        }

    @staticmethod
    def _doc_text(doc) -> str:
        if hasattr(doc, "page_content"):
            return getattr(doc, "page_content", "") or ""
        if isinstance(doc, str):
            return doc
        return str(doc)

    def _doc_key(self, doc) -> tuple:
        text = self._doc_text(doc)
        source = getattr(doc, "metadata", {}).get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        return source, text[:220]

    def _doc_key_with_source(self, doc, source: str) -> tuple:
        return source, self._doc_text(doc)[:220]
