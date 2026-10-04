import math
from typing import Any

from ddgs import DDGS

from knowledge_base import KnowledgeBase


def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web for current or external information.

    Args:
        query: A concise web search query.
        max_results: Number of results to return, between 1 and 8.

    Returns:
        Search results containing titles, URLs, and snippets.
    """
    max_results = max(1, min(int(max_results), 8))

    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as exc:
        return f"Web search failed: {exc}"

    if not results:
        return "No web results were found."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("href") or item.get("url", "")
        snippet = item.get("body", "")
        lines.append(f"[WEB {i}] {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n\n".join(lines)


def search_knowledge_base(kb: KnowledgeBase, query: str, top_k: int = 5) -> str:
    """Search uploaded documents for passages relevant to a query.

    Args:
        kb: The application's document knowledge base.
        query: A natural-language query.
        top_k: Maximum number of passages to return.

    Returns:
        Relevant document passages with source filenames.
    """
    if kb.chunk_count == 0:
        return "The knowledge base is empty. No uploaded documents are available."

    results = kb.search(query, top_k=max(1, min(int(top_k), 8)))
    if not results:
        return "No relevant passages were found in the uploaded documents."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"[KB {i}] Source: {item['source']} | Score: {item['score']:.3f}\n"
            f"{item['text']}"
        )

    return "\n\n".join(lines)


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: A basic arithmetic expression such as '1200 * 0.18'.

    Returns:
        The calculated numeric result.
    """
    allowed = set("0123456789+-*/().% ")
    if not expression or any(char not in allowed for char in expression):
        return "Calculation rejected: only basic arithmetic characters are allowed."

    try:
        value = eval(expression, {"__builtins__": {}}, {})
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return str(value)
        return "Calculation produced a non-finite result."
    except Exception as exc:
        return f"Calculation failed: {exc}"
