import ast
import math
import operator

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
    try:
        max_results = max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        max_results = 5

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

    try:
        top_k = max(1, min(int(top_k), 8))
    except (TypeError, ValueError):
        top_k = 5

    try:
        results = kb.search(query, top_k=top_k)
    except Exception as exc:
        return f"Knowledge base search failed: {exc}"

    if not results:
        return "No relevant passages were found in the uploaded documents."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"[KB {i}] Source: {item['source']} | Score: {item['score']:.3f}\n"
            f"{item['text']}"
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------- calculator
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MAX_OPERAND = 1e12
_MAX_EXPONENT = 100


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        if abs(node.value) > _MAX_OPERAND:
            raise ValueError("number too large")
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_EXPONENT or abs(left) > _MAX_OPERAND:
                raise ValueError("exponent too large")

        result = _BINARY_OPS[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > 1e300:
            raise ValueError("result too large")
        return result

    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: A basic arithmetic expression such as '1200 * 0.18' or '(180 - 120) / 120 * 100'.

    Returns:
        The calculated numeric result.
    """
    if not expression or len(expression) > 200:
        return "Calculation rejected: expression is empty or too long."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_node(tree)
    except ZeroDivisionError:
        return "Calculation failed: division by zero."
    except (ValueError, SyntaxError) as exc:
        return f"Calculation rejected: {exc}."
    except Exception as exc:
        return f"Calculation failed: {exc}"

    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return str(value)
    return "Calculation produced a non-finite result."