# AI Research & Analysis Agent

**Author:** Ponthapalli Arun Kumar  
**GitHub:** https://github.com/ponthapalliarunkumar  
**Live demo:** https://ai-research-analysis-agent-ea47ucepzoa85j6fnqmzcy.streamlit.app/

**GenAI + Agentic AI + RAG + Web Research**

A portfolio research agent built with Python, Gemini, Streamlit, and tool calling.

## What it does

Give the agent a research request such as:

> Research how generative AI is changing cybersecurity in 2026 and prepare a concise report.

The agent can:

1. Understand the research objective.
2. Search the public web.
3. Search uploaded TXT/PDF/DOCX documents (RAG).
4. Use a safe calculator for numerical analysis.
5. Synthesize the retrieved evidence into a structured report.
6. Include a Sources section (requested in the system prompt; always verify the links).
7. Remember the recent conversation for follow-up questions.
8. Export each report as Markdown.

## Architecture

```text
User
  |
  v
Gemini Research Agent
  |
  +---- web_search ----------> Public web (DuckDuckGo via ddgs)
  |
  +---- document search -----> Uploaded documents (Gemini embeddings + cosine similarity)
  |
  +---- calculate -----------> AST-based safe arithmetic
  |
  v
Evidence synthesis
  |
  v
Research report with sources
```

## Tech stack

Python, Google Gemini API (tool calling + embeddings), RAG, Streamlit, NumPy, ddgs, pypdf, python-docx.

## Project structure

```text
ai-research-analysis-agent/
+-- app.py
+-- agent.py
+-- tools.py
+-- knowledge_base.py
+-- requirements.txt
+-- .env.example
+-- .gitignore
+-- README.md
+-- .streamlit/
    +-- config.toml
```

## Run locally

```bash
git clone https://github.com/ponthapalliarunkumar/ai-research-analysis-agent.git
cd ai-research-analysis-agent
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API key:

```text
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-3.8-flash
```

Never commit `.env`. Then start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app from the repository and set the main file to `app.py`.
3. In Settings > Secrets, add:

```toml
GEMINI_API_KEY = "your_real_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Do not put the API key in GitHub.

## Security notes

- API keys live only in Streamlit Secrets or `.env`, both excluded from Git.
- The calculator parses expressions with Python's `ast` module (numbers and basic operators only, with size limits). It never uses `eval`.
- Tool calls are capped per request to limit cost.

## Example prompts

- Research the impact of generative AI on cybersecurity in 2026.
- Compare RAG and fine-tuning for enterprise knowledge assistants.
- Use my uploaded documents and the web to summarize this topic.
- Calculate the percentage increase from 120 to 180.

## Future improvements

Multi-agent planning, source quality scoring, persistent vector database, user authentication, research history, evaluation and observability.