import time

from google import genai
from google.genai import errors, types

from knowledge_base import KnowledgeBase
from tools import calculate, search_knowledge_base, web_search

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_HISTORY_MESSAGES = 6
RETRY_DELAYS = [2, 5, 10]  # seconds to wait after a temporary 503/429 error

SYSTEM_INSTRUCTION = """
You are an AI Research & Analysis Agent.

Your job is to produce useful, evidence-aware research answers.

When the user asks for research:
1. Understand the objective.
2. Use web_search for current or external information.
3. Use search_uploaded_documents when the uploaded knowledge base may contain relevant information.
4. Use calculate when numerical calculations are needed.
5. Synthesize the retrieved evidence instead of simply copying it.
6. Clearly separate facts from interpretation.
7. Include a Sources section (with URLs or document filenames) whenever web or knowledge-base sources are used.
8. Never invent a source, URL, quote, statistic, or citation.
9. If evidence is insufficient, say so.
10. Keep the final report structured with headings and concise bullet points where useful.

You are an agent because you can choose and call tools to complete multi-step research tasks.
"""


class ResearchAgent:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        api_key: str,
        model: str | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.model = model or DEFAULT_MODEL
        self.client = genai.Client(api_key=api_key)

    def run(self, user_request: str, history: list[dict] | None = None) -> str:
        kb = self.knowledge_base

        # A plain function (not a bound method) so the SDK can safely copy the
        # tool list. It captures `kb` from this scope instead of using `self`.
        def search_uploaded_documents(query: str) -> str:
            """Search the user's uploaded documents for information relevant to a query.

            Args:
                query: A natural-language search query describing the information needed.

            Returns:
                Relevant passages from the uploaded knowledge base.
            """
            return search_knowledge_base(kb, query)

        tools = [web_search, search_uploaded_documents, calculate]

        response = self._generate_with_retry(
            model=self.model,
            contents=self._build_contents(user_request, history),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=tools,
                temperature=0.2,
                max_output_tokens=8192,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=10
                ),
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            return "The model did not return a text response. Please try rephrasing your request."

        if self._was_truncated(response):
            text += "\n\n*Note: the report was cut off because it hit the output length limit.*"

        return text

    def _generate_with_retry(self, **kwargs):
        """Retry on temporary Gemini overload (503) and rate limits (429)."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return self.client.models.generate_content(**kwargs)
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code not in (429, 503) or attempt == len(RETRY_DELAYS):
                    raise
                time.sleep(RETRY_DELAYS[attempt])

    @staticmethod
    def _build_contents(user_request: str, history: list[dict] | None) -> list[types.Content]:
        contents: list[types.Content] = []

        for message in (history or [])[-MAX_HISTORY_MESSAGES:]:
            role = "user" if message["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=message["content"])])
            )

        contents.append(types.Content(role="user", parts=[types.Part(text=user_request)]))
        return contents

    @staticmethod
    def _was_truncated(response) -> bool:
        try:
            reason = response.candidates[0].finish_reason
            return "MAX_TOKENS" in str(reason)
        except (AttributeError, IndexError, TypeError):
            return False