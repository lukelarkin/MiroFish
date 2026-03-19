# MiroFish — Developer Guide

## What This Is

Swarm Intelligence AI Prediction Engine. Simulates social media discourse across thousands of AI agents to predict real-world outcomes. Bilingual (Chinese/English).

Pipeline: **Upload docs → Build knowledge graph (Zep) → Generate agent profiles → Run OASIS simulation → Generate prediction report**

## Quick Start

```bash
cp .env.example .env   # Fill in your API keys
npm run setup:all      # Install Python + Node dependencies
npm run dev            # Starts Flask (5001) + Vite (5173) concurrently
```

Docker alternative:
```bash
docker compose up
```

## Required API Keys

| Key | Required For | Get It |
|-----|-------------|--------|
| `LLM_API_KEY` | All AI features (ontology, profiles, reports) | [Anthropic Console](https://console.anthropic.com/) (sk-ant-*) or any OpenAI-compatible provider |
| `ZEP_API_KEY` | Knowledge graph, entity memory | [Zep Cloud](https://app.getzep.com/) (free tier available) |
| `FRED_API_KEY` | Broker trends economic data | [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html) (free) |

Without keys: health check and project CRUD work. Everything else needs at minimum `LLM_API_KEY`.

## Project Structure

```
backend/
  app/
    api/            # Flask route handlers (graph, simulation, report, broker_trends)
    services/       # Business logic (20 modules — the core engine)
    models/         # Project + Task persistence (JSON file-based)
    utils/          # LLM client, logger, file parser, retry logic
    config.py       # Centralized config from .env
  run.py            # Entry point: python run.py
  tests/            # pytest suite (68 test cases for data pipeline)

frontend/
  src/
    views/          # Vue 3 pages (Home, MainView, SimulationView, ReportView, etc.)
    components/     # Reusable Vue components
    api/            # Axios API clients (graph.js, simulation.js, report.js)
    router/         # Vue Router config
```

## Key Files

- `backend/app/utils/llm_client.py` — Unified LLM wrapper (auto-detects Anthropic vs OpenAI from key prefix)
- `backend/app/services/broker_trends.py` — End-to-end prediction orchestrator
- `backend/app/services/report_agent.py` — ReACT-pattern report generation with tool calling
- `backend/app/services/cross_validator.py` — Actuarial-style confidence scoring (Wilson CIs)
- `backend/app/services/data_pipeline.py` — Multi-source data validation pipeline
- `backend/app/services/actuary.py` — Risk assessment and prediction reliability grading
- `frontend/src/views/MainView.vue` — Main 5-step workflow UI

## API Conventions

- All responses: `{"success": bool, "data": {...}}` or `{"success": false, "error": "message"}`
- Route prefixes: `/api/graph/`, `/api/simulation/`, `/api/report/`, `/api/broker-trends/`
- Health check: `GET /health`
- Long operations return a `task_id` — poll via `GET /api/graph/task/{task_id}`

## Testing

```bash
cd backend && python -m pytest tests/ -v
```

## Architecture Notes

- **LLM provider**: Auto-detected from API key. `sk-ant-*` → Anthropic Claude, otherwise → OpenAI SDK format
- **State**: File-system JSON (no database). Projects in `backend/uploads/projects/`
- **Simulation**: Uses CAMEL-AI OASIS framework, runs as subprocess
- **Memory**: Zep Cloud for knowledge graphs and entity memory
- **Frontend**: Vue 3 + Vite + D3.js for graph visualization
