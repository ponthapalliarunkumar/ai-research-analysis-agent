# AI Research & Analysis Agent

**Author:** Ponthapalli Arun Kumar  
**GitHub:** https://github.com/ponthapalliarunkumar  
**Live demo:** https://ai-research-analysis-agent-ea47ucepzoa85j6fnqmzcy.streamlit.app/

**GenAI + Agentic AI + RAG + Web Research**

A portfolio-ready research agent built with Python, Gemini, Streamlit, and tool calling.

## What it does

Give the agent a research request such as:

> Research how generative AI is changing cybersecurity in 2026 and prepare a concise report.

The agent can:

1. Understand the research objective.
2. Search the public web.
3. Search uploaded TXT/PDF/DOCX documents.
4. Use a calculator when numerical analysis is needed.
5. Synthesize the retrieved evidence.
6. Generate a structured report with sources.
7. Export the report as Markdown.

## Architecture

```text
User
  |
  v
Gemini Research Agent
  |
  +---- web_search ----------> Public web
  |
  +---- search_knowledge_base -> Uploaded documents
  |
  +---- calculate -----------> Arithmetic
  |
  v
Evidence synthesis
  |
  v
Cited research report
```

## Tech stack

- Python
- Google Gemini API
- Gemini function/tool calling
- Gemini embeddings
- Retrieval-Augmented Generation (RAG)
- Streamlit
- NumPy
- DuckDuckGo search via `ddgs`
- PDF/DOCX/TXT processing

## Project structure

```text
ai-research-analysis-agent/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ app.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ agent.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tools.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ knowledge_base.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .env.example
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ .gitignore
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ README.md
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ data/
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ .streamlit/
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ config.toml
```

## Run locally

### 1. Clone

```bash
git clone https://github.com/ponthapalliarunkumar/ai-research-analysis-agent.git
cd ai-research-analysis-agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Copy `.env.example` to `.env` and add your API key:

```text
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-3.8-flash
```

Never commit `.env`.

### 5. Start the app

```bash
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from this repository.
4. Set the main file to `app.py`.
5. Add this secret:

```toml
GEMINI_API_KEY = "your_real_key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Do not put the API key in GitHub.

## Example prompts

- Research the impact of generative AI on cybersecurity in 2026.
- Compare RAG and fine-tuning for enterprise knowledge assistants.
- Research the current challenges of AI agents in software engineering.
- Use my uploaded documents and the web to prepare a summary of this topic.
- Calculate the percentage increase from 120 to 180.

## Interview explanation

The project demonstrates:

- Generative AI
- Agentic AI
- Tool/function calling
- RAG
- Embeddings
- Semantic search
- Prompt engineering
- API integration
- Document processing
- Streamlit deployment

## Future improvements

- Multi-agent planning
- Source quality scoring
- Better citation extraction
- Persistent vector database
- User authentication
- Research history
- Scheduled research
- Evaluation and observability
