**Part 1: Your Scenario (20 points)**

Main Objective: Build an automated monitor that tracks volatility in the University Tech Equity portfolio by combining daily price data with macroeconomic indicators.
Data Sources: Alpha Vantage (equities), FRED (macro indicators), Financial Modeling Prep (backup fundamentals).
Data Types: Time-series OHLC prices, macroeconomic observations, derived percent change metrics, fetch metadata.
Geographic Scope: Global equities with U.S. macroeconomic context.
Time Range: 90-day historical backfill plus daily updates at 22:00 UTC.

**Part 2: Learning about API (15 points)**  
Write a brief reflection (1 paragraph) on what you learned about APIs

_(Add your paragraph reflection here once you finish the write-up.)_

**Part 3: Setting Up Free API Access (10 points)**  
Screenshot showing successful API key creation

- Capture the confirmation screens for: (1) Alpha Vantage key dashboard showing the issued key, (2) Financial Modeling Prep profile page with the generated key, and (3) FRED application approval page listing the project description.

**Part 4: Build Your AI Data Collection Agent (35 points)**  
Screenshots of your agent running

- Terminal run of `FinancialDataCollectionAgent` showing task execution and (on your network) the successful completion summary.
- Updated `logs/collection.log` tail after the run to demonstrate respectful rate limiting and retries.
- (Optional) Explorer/Finder view of new files in `ai_agent_willchristophersander/data/processed/` created by the run.

**Part 5:Documentation (20 pts)**  
Quality assessment report  and Collection Summary

Quality assessment report should contain:

- [x] total number of records — **1,144** (AAPL 100, MSFT 100, CPIAUCSL 944)  
- [x] collection success rate — **100%** completion on successful run (Alpha Vantage tasks completed; FRED succeeded but flagged staleness)  
- [x] quality score — **0.999** overall; per-source highlights: AAPL/MSFT score 1.00, FRED score 0.66 because latest observation predates the freshness threshold

Collection Summary should contain:

- [x] Total data points collected — **1,144**  
- [x] Success/failure rates by API — Alpha Vantage 2/2 successful tasks; FRED 1/1 completed but freshness warning triggered (data lag > 48h)  
- [x] Quality metrics and trends — Equity feeds scoring 1.00 quality; FRED feed quality 0.66 because of staleness; overall score 0.999 with freshness dominated by equities  
- [x] Any issues encountered — FRED series older than freshness threshold; recommend manual check or supplemental source. (Note: sandbox rerun later failed because outbound DNS blocked—documented in logs.)  
- [x] Recommendations for future collection — Schedule daily run post-market close, monitor FRED freshness or add backup, maintain Alpha Vantage delay at ≥12s, rerun in network-permitted environment if failures appear
