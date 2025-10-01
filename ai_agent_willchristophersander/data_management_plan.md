# Mini Data Management Plan: Financial Market Risk Monitoring Agent

## Project Overview
- **Objective**: Build an automated agent that collects daily equity price movements and macroeconomic indicators to monitor downside risk for a university-managed technology equities portfolio.
- **Research Questions**:
  1. How do short-term price swings in core holdings compare to macroeconomic signals (e.g., inflation trends, consumer sentiment)?
  2. Can we detect anomalous volatility or data quality issues quickly enough to inform rebalancing decisions?
  3. What collection cadence and validation checks are needed to maintain clean, analysis-ready data sets?

## Data Sources
| Source | API Endpoint | Access | Key Variables | Justification |
|--------|--------------|--------|---------------|---------------|
| Alpha Vantage | `https://www.alphavantage.co/query` | Requires API key (free tier) | Daily OHLC prices, adjusted closing price, volume for primary tickers (AAPL, MSFT, NVDA) | High quality equity pricing with generous free allotment and reliable metadata |
| FRED (Federal Reserve Economic Data) | `https://api.stlouisfed.org/fred/series/observations` | Requires API key (free tier) | CPI (`CPIAUCSL`), 10-year Treasury yield (`DGS10`), Consumer Sentiment (`UMCSENT`) | Provides macro indicators to contextualize equity movements |
| Financial Modeling Prep (Backup) | `https://financialmodelingprep.com/api/v3/profile/{symbol}` | Requires API key (free tier) | Company fundamentals (beta, marketCap) | Used for enrichment when Alpha Vantage throttles or fails |

## Data Types
- **Time series metrics**: Daily OHLC price data, macroeconomic observations (float, timestamp).
- **Categorical metadata**: Ticker symbol, data source, economic series IDs.
- **Quality fields**: Status, error messages, retrieval timestamp, response latency.

## Geographic Scope
- Global publicly traded companies with a focus on US markets; macro indicators represent US economy.

## Time Range & Cadence
- Historical backfill: previous 90 days on first run.
- Daily incremental updates executed at 22:00 UTC, configurable per source.

## Collection & Storage Strategy
- **Configuration**: JSON file describing sources, required symbols/series, and environment variable names for API keys. All credentials pulled from `.env` or OS environment.
- **Storage**: Raw responses saved as JSON in `data/raw/`; normalized parquet/CSV stored in `data/processed/`; metadata (schema, fetch stats) stored in `data/metadata/`.
- **Versioning**: Timestamped filenames with ISO 8601 suffix; Git tracks configuration and code while raw data directories are excluded from version control by default.

## Quality Management
- **Validation Checks**: Schema validation (required columns), freshness check (latest point must be within threshold), completeness ratio (non-null vs expected rows), anomaly detection (Z-score > 3 for returns flagged).
- **Manual Review Hooks**: Summaries pushed to `reports/collection_summary.md` and HTML quality report to `reports/quality_report.html` for analyst review.
- **Error Handling**: Exponential backoff retries, fallback to backup source, logging of failures with severity classification.

## Responsible & Respectful Collection
- Automatic rate limiter honours published free-tier quotas (12-second delay for Alpha Vantage, configurable for others).
- User agent and contact email included in headers; requests contain disclaimers in comment logs.
- All keys stored outside source control with `.env` guidance.

## Ethics & Compliance
- APIs provide publicly available financial data for personal/educational use; terms of service reviewed (Alpha Vantage and FRED allow non-commercial use with attribution).
- No personal identifiable information collected. Data stored in university-approved secure location.

## Risk Mitigation
- **Data Gaps**: Maintain backup providers and alerts when completeness < 90%.
- **API Outages**: Circuit breaker disables failing source after repeated errors and logs recommended manual follow-up.
- **Key Leakage**: `.env.example` supplied; `config` references env var names only; repository `.gitignore` updated to exclude secrets.

## Deliverables Summary
- Automated collection agent class with modular API connectors.
- Config template and instructions for injecting credentials.
- Daily collection log, metadata JSON, HTML quality report, Markdown collection summary.
- Tests covering request handling, quality metrics, and adaptive retry logic using mock responses.
