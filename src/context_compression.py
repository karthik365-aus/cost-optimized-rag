import re
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContextCompressor:
    def __init__(self, max_sentences_map=None, min_sentence_length=20):
        self.max_sentences_map = max_sentences_map or {
            "simple": 2,
            "medium": 4,
            "complex": 6,
        }
        self.min_sentence_length = min_sentence_length

    def compress(self, query: str, retrieved_docs: List[Any], complexity: str = "medium") -> Dict[str, Any]:
        texts = self._extract_texts(retrieved_docs)
        original_text = "\n\n".join(texts).strip()

        if not original_text:
            return {
                "compressed_context": "",
                "selected_sentences": [],
                "original_text": "",
                "original_token_count": 0,
                "compressed_token_count": 0,
                "compression_ratio": 0.0,
            }

        sentences = self._split_into_sentences(original_text)
        sentences = self._clean_and_filter_sentences(sentences)

        if not sentences:
            original_tokens = self._count_tokens(original_text)
            return {
                "compressed_context": original_text,
                "selected_sentences": [],
                "original_text": original_text,
                "original_token_count": original_tokens,
                "compressed_token_count": original_tokens,
                "compression_ratio": 0.0,
            }

        ranked_sentences = self._rank_sentences(query, sentences)
        top_n = self.max_sentences_map.get(complexity, 4)
        selected_sentences = [s for s, _ in ranked_sentences[:top_n]]

        compressed_context = " ".join(selected_sentences).strip()

        original_tokens = self._count_tokens(original_text)
        compressed_tokens = self._count_tokens(compressed_context)
        compression_ratio = 0.0 if original_tokens == 0 else (original_tokens - compressed_tokens) / original_tokens

        return {
            "compressed_context": compressed_context,
            "selected_sentences": selected_sentences,
            "original_text": original_text,
            "original_token_count": original_tokens,
            "compressed_token_count": compressed_tokens,
            "compression_ratio": round(compression_ratio, 4),
        }

    def _extract_texts(self, retrieved_docs: List[Any]) -> List[str]:
        texts = []
        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                texts.append(doc.page_content)
            elif isinstance(doc, str):
                texts.append(doc)
        return texts

    def _split_into_sentences(self, text: str) -> List[str]:
        text = text.replace("\n", " ").strip()
        if not text:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _clean_and_filter_sentences(self, sentences: List[str]) -> List[str]:
        seen = set()
        cleaned = []

        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            key = normalized.lower()

            if len(normalized) < self.min_sentence_length:
                continue
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(normalized)

        return cleaned

    def _rank_sentences(self, query: str, sentences: List[str]):
        corpus = [query] + sentences
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)

        query_vector = tfidf_matrix[0:1]
        sentence_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(query_vector, sentence_vectors).flatten()

        ranked = list(zip(sentences, similarities))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _count_tokens(self, text: str) -> int:
        return len(text.split())
