"""
Web search via Tavily.
Provides a simple trigger heuristic and a formatted result builder.
"""

from __future__ import annotations

import logging
import os

from langchain_community.tools.tavily_search import TavilySearchResults

from app.utils.config import SEARCH_MAX_RESULTS, SEARCH_TRIGGER_KEYWORDS

logger = logging.getLogger(__name__)


def should_search(question: str) -> bool:
    """Return True if the question likely needs live web data."""
    q = question.lower()
    return any(kw in q for kw in SEARCH_TRIGGER_KEYWORDS)


def run(question: str, tavily_api_key: str) -> str:
    """
    Execute a Tavily search and return formatted results as a plain string.

    Parameters
    ----------
    question : str
        The user's question (used directly as the search query).
    tavily_api_key : str
        Tavily API key.

    Returns
    -------
    str
        Formatted search results, or a 'no results' message.
    """
    os.environ["TAVILY_API_KEY"] = tavily_api_key   # LangChain reads from env

    try:
        tool    = TavilySearchResults(max_results=SEARCH_MAX_RESULTS)
        results = tool.invoke({"query": question})
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return ""

    if not results:
        return "No web results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Summary: {r.get('content', '').strip()}"
        )
    return "\n\n".join(lines)
