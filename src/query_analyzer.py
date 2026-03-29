import re


class QueryAnalyzer:
    """Classify a query as simple, medium, or complex using rule-based logic."""

    COMPLEX_KEYWORDS = [
        "analyze", "analyse", "evaluate", "design", "compare", "comparing",
        "explain", "predict", "recommend", "strategy", "framework",
        "trade-off", "tradeoff", "assess", "develop", "critique",
        "justify", "synthesize", "investigate",
    ]

    MEDIUM_KEYWORDS = [
        "how", "why", "difference", "relationship", "describe",
        "what factors", "what role", "how does", "how do", "what are the",
        "what led", "what caused", "in what way", "to what extent",
    ]

    SIMPLE_STARTERS = [
        "what is", "what was", "who", "when", "where", "which",
        "how many", "how often", "how much", "is there", "does",
    ]

    MULTI_QUESTION_CONNECTORS = [
        "and also", "additionally", "as well as", "furthermore", "moreover",
    ]

    def analyze(self, query: str) -> dict:
        """
        Classify the query complexity.

        Args:
            query: The user's question.

        Returns:
            Dict with complexity label and reasoning.
        """
        normalized = query.lower().strip()
        word_count = len(normalized.split())

        complexity = self._classify(normalized, word_count)
        complexity = self._bump_if_multi_question(normalized, complexity)

        return {
            "query": query,
            "complexity": complexity,
            "word_count": word_count,
        }

    def _classify(self, normalized: str, word_count: int) -> str:
        # Rule 1 — Complex keywords take highest priority
        if any(kw in normalized for kw in self.COMPLEX_KEYWORDS):
            return "complex"

        # Rule 2 — Medium keywords
        if any(kw in normalized for kw in self.MEDIUM_KEYWORDS):
            # Rule 3 — But if very short (5 words or less), keep it simple
            if word_count <= 5:
                return "simple"
            return "medium"

        # Rule 5 — Simple starters
        if any(normalized.startswith(kw) for kw in self.SIMPLE_STARTERS):
            return "simple"

        # Rule 3 — Length as tiebreaker
        if word_count > 20:
            return "complex"
        elif word_count >= 10:
            return "medium"
        else:
            return "simple"

    def _bump_if_multi_question(self, normalized: str, complexity: str) -> str:
        # Rule 4 — Multiple questions bump up one level
        has_multiple_question_marks = normalized.count("?") > 1
        has_connector = any(c in normalized for c in self.MULTI_QUESTION_CONNECTORS)

        if has_multiple_question_marks or has_connector:
            if complexity == "simple":
                return "medium"
            elif complexity == "medium":
                return "complex"

        return complexity
