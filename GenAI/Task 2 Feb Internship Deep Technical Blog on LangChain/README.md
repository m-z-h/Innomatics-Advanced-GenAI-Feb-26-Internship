# LangChain Deep Tech Blog

A complete technical blog project that combines conceptual depth, runnable examples, and architecture guidance for LangChain.

## Repository structure

- `blog.md` — Full medium-ready article covering LangChain architecture, components, examples, and use cases.
- `langchain_blog.ipynb` — Jupyter notebook with section-wise explanations and executable code.
- `requirements.txt` — Python dependencies.
- `langchain_blog_demo.py` — Modular example functions that can be imported into notebooks or scripts.
- `images/diagram_description.txt` — Diagram design guidance and ASCII architecture.
- `images/architecture_diagram.txt` — ASCII diagram of LangChain flow.

## Setup

1. Create a virtual environment.

```powershell
python -m venv venv
```

2. Activate the environment.

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Set your OpenAI API key.

```powershell
$env:OPENAI_API_KEY = "your_api_key_here"
```

> Note: This project uses `langchain-classic` for chain, agent, and vector store compatibility with the current LangChain package layout.

5. Launch the notebook.

```powershell
jupyter notebook langchain_blog.ipynb
```

jupyter notebook langchain_blog.ipynb
```

## How to use this project

- Read `blog.md` for a detailed technical article.
- Open `langchain_blog.ipynb` to run the same examples interactively.
- Use `langchain_blog_demo.py` for reusable helper functions in your own experiments.
- Refer to `images/architecture_diagram.txt` for a clear ASCII diagram of the LangChain workflow.

## Key notebook sections

- Setup and environment validation
- Basic model calls
- Prompt template construction
- Chain composition and sequential workflows
- Memory-enabled conversation examples
- Agent and tool orchestration
- Vector store and retrieval examples

## Notes

- This project uses LangChain's `init_chat_model` chat model initialization and assumes an OpenAI-compatible API key.
- The notebook and helper module are designed for clean, modular, runnable demonstrations.
- If you want to explore beyond this project, extend the notebook with document loaders, multi-agent workflows, and retrieval pipelines.
