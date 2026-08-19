"""
Query Transformation Engine — HyDE & Sub-Query Decomposition
============================================================
Provides query rewriting and expansion strategies to improve retrieval recall:
  1. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer passage
     to align query embeddings with document-level semantic representations.
  2. MultiQueryDecomposer: Breaks complex or multi-faceted queries into distinct
     sub-queries for broader coverage.
  3. QueryTransformer: Unified orchestrator supporting configurable expansion strategies.
"""

from __future__ import annotations

import enum
from typing import Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)


class QueryTransformStrategy(str, enum.Enum):
    NONE = "none"
    HYDE = "hyde"
    MULTI_QUERY = "multi_query"
    ALL = "all"


class HyDEGenerator:
    """Generates a hypothetical document based on the user's query."""

    DEFAULT_PROMPT = (
        "You are an expert technical domain assistant. Given the following query, "
        "write a concise, factual, and informative paragraph that answers the question. "
        "Do not include conversational filler. Focus purely on technical content.\n\n"
        "Query: {query}\n\n"
        "Hypothetical Answer:"
    )

    def __init__(self, llm_client: Optional[Any] = None, prompt_template: Optional[str] = None):
        self.llm = llm_client
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def generate(self, query: str) -> str:
        """Generate a hypothetical document passage for dense embedding alignment."""
        if not self.llm:
            logger.warning("No LLM client configured for HyDE. Returning original query.")
            return query

        formatted_prompt = self.prompt_template.format(query=query)
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(formatted_prompt)
                content = response.content if hasattr(response, "content") else str(response)
                return content.strip()
            elif callable(self.llm):
                return str(self.llm(formatted_prompt)).strip()
            return query
        except Exception as e:
            logger.error("HyDE generation failed", error=str(e), query=query)
            return query


class MultiQueryDecomposer:
    """Decomposes a complex query into 2-4 distinct sub-queries."""

    DEFAULT_PROMPT = (
        "You are an expert research assistant. Break down the following user query "
        "into 2 to 4 diverse sub-queries from different perspectives or sub-topics "
        "to ensure comprehensive document retrieval.\n"
        "Output each sub-query on a new line. Do not number the lines.\n\n"
        "User Query: {query}\n\n"
        "Sub-queries:"
    )

    def __init__(self, llm_client: Optional[Any] = None, num_queries: int = 3):
        self.llm = llm_client
        self.num_queries = num_queries

    def decompose(self, query: str) -> List[str]:
        """Decompose query into a list of sub-queries (always includes original query)."""
        queries = [query]
        if not self.llm:
            logger.warning("No LLM client configured for MultiQuery. Returning original query.")
            return queries

        formatted_prompt = self.DEFAULT_PROMPT.format(query=query)
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(formatted_prompt)
                raw_text = response.content if hasattr(response, "content") else str(response)
            elif callable(self.llm):
                raw_text = str(self.llm(formatted_prompt))
            else:
                return queries

            lines = [line.strip().lstrip("0123456789.- ") for line in raw_text.splitlines() if line.strip()]
            for line in lines[: self.num_queries]:
                if line and line.lower() != query.lower() and line not in queries:
                    queries.append(line)
            return queries
        except Exception as e:
            logger.error("MultiQuery decomposition failed", error=str(e), query=query)
            return queries


class QueryTransformer:
    """Unified query transformation orchestrator."""

    def __init__(
        self,
        strategy: QueryTransformStrategy = QueryTransformStrategy.NONE,
        llm_client: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.hyde_generator = HyDEGenerator(llm_client)
        self.multi_query_decomposer = MultiQueryDecomposer(llm_client)

    def transform(self, query: str) -> List[str]:
        """Apply the selected transformation strategy and return the list of queries to retrieve."""
        if self.strategy == QueryTransformStrategy.HYDE:
            hyde_doc = self.hyde_generator.generate(query)
            logger.info("Generated HyDE passage", original_query=query, hyde_passage=hyde_doc[:80])
            return [query, hyde_doc] if hyde_doc != query else [query]

        elif self.strategy == QueryTransformStrategy.MULTI_QUERY:
            sub_queries = self.multi_query_decomposer.decompose(query)
            logger.info("Decomposed into sub-queries", count=len(sub_queries), original_query=query)
            return sub_queries

        elif self.strategy == QueryTransformStrategy.ALL:
            sub_queries = self.multi_query_decomposer.decompose(query)
            hyde_doc = self.hyde_generator.generate(query)
            result = list(dict.fromkeys(sub_queries + [hyde_doc]))
            return result

        return [query]
