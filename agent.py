import os
from typing import Any

from google import genai
from google.genai import types

from knowledge_base import KnowledgeBase
from tools import calculate, search_knowledge_base, web_search


SYSTEM_INSTRUCTION = """
You are an AI Research & Analysis Agent.

Your job is to produce useful, evidence-aware research answers.

When the user asks for research:
1. Understand the objective.
2. Use web_search for current or external information.
3. Use search_knowledge_base when the uploaded knowledge base may contain relevant information.
4. Use calculate when numerical calculations are needed.
5. Synthesize the retrieved evidence instead of simply copying it.
6. Clearly separate facts from interpretation.
7. Include a Sources section whenever web or knowledge-base sources are used.
8. Never invent a source, URL, quote, statistic, or citation.
9. If evidence is insufficient, say so.
10. Keep the final report structured with headings and concise bullet points where useful.

You are an agent because you can choose and call tools to complete multi-step research tasks.
"""


class ResearchAgent:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def run(self, user_request: str) -> str:
        tools = [
            web_search,
            self._knowledge_search,
            calculate,
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_request,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=tools,
                temperature=0.2,
                max_output_tokens=4000,
            ),
        )

        text = getattr(response, "text", None)
        if text:
            return text

        return "The model did not return a text response."

    def _knowledge_search(self, query: str) -> str:
        """Search the user's uploaded documents for information relevant to a query.

        Args:
            query: A natural-language search query describing the information needed.

        Returns:
            Relevant passages from the uploaded knowledge base.
        """
        return search_knowledge_base(self.knowledge_base, query)
