# srs-agent

Final-year project: an autonomous agent that turns a software requirements specification (SRS) into a tested Python project, with a base agent compared against a memory-enhanced (RAG) agent.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                    # create .venv with Python 3.12 and install dependencies
cp .env.example .env       # then fill in your API keys
```

## Development

```bash
uv run pytest              # run the tests
uv run ruff check .        # lint
uv run ruff format .       # format
```
