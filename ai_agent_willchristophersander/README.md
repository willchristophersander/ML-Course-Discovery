# Financial Market Data Collection Agent

Automated agent that ingests high-signal financial and macroeconomic datasets to monitor volatility across a technology equities portfolio. The system reuses the modular configuration, logging, and quality evaluation practices from the AppleAppSander project while re-targeting the domain to API-driven data collection.

## Features
- **Config-driven orchestration**: JSON configuration specifies APIs, tickers/series, and environment variable bindings for credentials.
- **Reusable API clients**: Typed clients with retry logic, rate limiting, and respectful headers for Alpha Vantage, FRED, and Financial Modeling Prep.
- **Adaptive safeguards**: Circuit breaker avoids hammering failing endpoints; per-task quality issues feed back into metadata summaries.
- **Quality automation**: Completeness, freshness, and anomaly metrics produce an HTML quality report plus machine-readable metadata.
- **Deliverables-ready artefacts**: Raw/processed data, collection log, metadata JSON, quality report, and Markdown collection summary conform to assignment requirements.

## Repository Layout
```
ai_agent_willchristophersander/
├── agent/
│   ├── api_clients.py            # API connectors with retry + rate limiting
│   ├── config.py                 # Dataclasses + loader for config.json
│   ├── data_collection_agent.py  # FinancialDataCollectionAgent orchestrator
│   ├── logging_utils.py          # Shared logging setup
│   ├── metadata.py               # Metadata aggregation helpers
│   ├── quality.py                # Quality evaluator + HTML report generator
│   ├── strategies.py             # Task planner + circuit breaker
│   ├── requirements.txt          # Minimal dependency list
│   └── tests/
│       └── test_agent.py         # Pytest suite using stubbed clients
├── data/
│   ├── raw/                      # Raw API responses (JSON)
│   ├── processed/                # Normalised CSVs with pct-change
│   └── metadata/                 # Metadata + schema artefacts
├── demo/
│   ├── api_exercises.py          # Part 2 exercises (cat facts + holidays)
│   └── demo_screenshots/         # Screenshot staging folder
├── logs/
│   └── collection.log            # Runtime log output
├── reports/
│   ├── collection_summary.md     # Markdown summary (generated)
│   └── quality_report.html       # HTML quality report (generated)
├── data_management_plan.md       # Filled-in Mini DMP (Part 1)
└── README.md                     # This guide
```

## Quick Start
1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r ai_agent_willchristophersander/agent/requirements.txt
   ```

2. **Provide credentials**
   ```bash
   export ALPHAVANTAGE_API_KEY="your-key"
   export FRED_API_KEY="your-key"
   export FMP_API_KEY="optional-backup-key"
   ```
   The configuration references environment variable names only; no secrets are committed.

3. **Run the agent**
   ```bash
   python - <<'PY'
   from pathlib import Path
   from ai_agent_willchristophersander.agent import FinancialDataCollectionAgent

   config_path = Path("ai_agent_willchristophersander/agent/config.json")
   FinancialDataCollectionAgent(config_path).run()
   PY
   ```
   Outputs are written under `ai_agent_willchristophersander/data`, `reports`, and `logs`.

4. **Review artefacts**
   - `logs/collection.log`: runtime log with retry/skip messages.
   - `data/raw/*.json`: timestamped raw API responses.
   - `data/processed/*.csv`: cleaned dataset with percent change column.
   - `data/metadata/collection_metadata.json`: machine-readable metadata.
   - `reports/quality_report.html`: HTML report covering quality metrics.
   - `reports/collection_summary.md`: Markdown summary for documentation.

## Demo Exercises (Part 2)
The `demo/api_exercises.py` script contains the cat facts and public holiday exercises:
```bash
python ai_agent_willchristophersander/demo/api_exercises.py
```
- Fetches five unique cat facts, adds error handling, and saves them to `demo/cat_facts.json`.
- Queries three countries via the Nager.Date API and prints a holiday count comparison.

## Testing
Pytest suite uses stubbed clients to avoid real network calls while asserting that metadata and reports are generated correctly.
```bash
pytest ai_agent_willchristophersander/agent/tests/test_agent.py
```

## Extending the Agent
- Add tickers or macro series via `agent/config.json`; the task planner automatically expands work items.
- Implement additional clients in `api_clients.py` (e.g., Quandl) and register them in `build_client`.
- Tune quality thresholds in `config.json` → `quality` block.

## Respectful Data Practices
- Rate limiting defaults (12 seconds for Alpha Vantage) are configurable per source.
- Circuit breaker prevents repeated failures from overwhelming APIs.
- Logs include enough context for auditability without leaking keys or PII.

## Next Steps
- Capture API key creation screenshots for Part 3.
- Run the agent with real credentials to populate `logs/`, `reports/`, and `data/` directories for reporting.
- Convert `data_management_plan.md` and generated summaries into PDF/HTML deliverables as required by the assignment rubric.
