"""
Vector retrieval over ``chroma_db`` using ``HuggingFaceEmbeddings`` (default BGE small).

``retrieve`` chooses k from query complexity score, may bump k when analyzer confidence is low,
re-runs once if keyword *coverage* of the query vs retrieved text is below a label-specific
threshold (when analyzer confidence passes a gate). Returns docs + rich metadata for logging.
"""
from pathlib import Path
import re
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = str(_PROJECT_ROOT / "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class AdaptiveRetriever:
    K_BASE_MIN = 3
    K_BASE_MAX = 10
    K_HARD_CAP = 10
    LOW_CONF_BONUS = 2
    RETRY_BONUS = 1
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

    def __init__(self, documents_path='data/documents'):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

        if Path(CHROMA_DIR).is_dir():
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )
            print("Vector database loaded")
        else:
            print("Loading documents...")
            loader = DirectoryLoader(
                documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.chunks = text_splitter.split_documents(documents)
            print(f"Created {len(self.chunks)} chunks")

            print("Creating vector database...")
            self.vectordb = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
            )
            print("Vector database created")

    def retrieve(
        self,
        query: str,
        complexity: str = "medium",
        complexity_score: float = None,
        analyzer_confidence: float = None,
    ) -> dict:
        normalized_complexity = (complexity or "medium").strip().lower()
        fallback_score_map = {'simple': 0.2, 'medium': 0.55, 'complex': 0.85}
        score = complexity_score if complexity_score is not None else fallback_score_map.get(normalized_complexity, 0.55)
        score = max(0.0, min(1.0, float(score)))

        # Score-first retrieval size: k = 3 + complexity_score * 7 with single hard cap.
        k_base = max(self.K_BASE_MIN, min(self.K_BASE_MAX, round(3 + score * 7)))
        low_confidence = analyzer_confidence is not None and float(analyzer_confidence) < 0.70
        k_final = min(k_base + self.LOW_CONF_BONUS, self.K_HARD_CAP) if low_confidence else k_base
        print(
            f"Retrieving {k_final} chunks for {normalized_complexity} query "
            f"(score={round(score, 2)}, confidence={analyzer_confidence})..."
        )

        docs, chunks_metadata = self._run_retrieval(query=query, k=k_final)
        coverage_score = self._keyword_coverage_score(query=query, docs=docs)
        retrieval_retry = False
        coverage_threshold = self.COVERAGE_THRESHOLDS.get(normalized_complexity, 0.50)
        low_confidence_for_retry = analyzer_confidence is None or float(analyzer_confidence) < self.RETRY_CONFIDENCE_GATE

        if coverage_score < coverage_threshold and low_confidence_for_retry and k_final < self.K_HARD_CAP:
            retry_k = min(k_final + self.RETRY_BONUS, self.K_HARD_CAP)
            docs, chunks_metadata = self._run_retrieval(query=query, k=retry_k)
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

    def _run_retrieval(self, query: str, k: int) -> Tuple[List, List]:
        results = self.vectordb.similarity_search_with_score(query, k=k)
        docs = []
        chunks_metadata = []
        for i, (doc, score) in enumerate(results):
            docs.append(doc)
            d = round(float(score), 4)
            chunks_metadata.append({
                "chunk_index": i,
                "source": doc.metadata.get("source", "unknown"),
                # Chroma / LangChain: distance in embedding space; LOWER = closer to query
                "chroma_distance": d,
                "similarity_score": d,
            })
        return docs, chunks_metadata

    def _keyword_coverage_score(self, query: str, docs: List) -> float:
        query_tokens = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
        keywords = {tok for tok in query_tokens if len(tok) > 2 and tok not in self.STOPWORDS}
        if not keywords:
            return 1.0

        retrieved_text = " ".join(getattr(doc, "page_content", "") for doc in docs).lower()
        matched = sum(1 for kw in keywords if kw in retrieved_text)
        return matched / len(keywords)
