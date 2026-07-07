"""
CLI entry points for the stock price forecasting data pipeline.

Pipeline stages
---------------
1. download       – Download raw SP500 price + fundamental data via yfinance,
                    compute fundamentals-based weighted adjacency, save to
                    data/sp500/raw/.
                    Entry point: stock-download

2. update_stocks  – Re-fetch S&P 500 sector metadata from Wikipedia and update
                    stocks.csv in an existing raw data directory.
                    Entry point: stock-update-stocks

3. analyze_dates  – Analyse date coverage and missing-data distribution across
                    stocks.  Helps choose a cleaning strategy and threshold.
                    Entry point: stock-analyze-dates

4. clean          – Clean values.csv (common_range / drop_incomplete /
                    forward_fill), recompute adjacency and graph diagnostics,
                    and write a parameterised cleaned dataset directory under
                    data/sp500/.
                    Entry point: stock-clean

5. sweep_static_graph   – Sweep (threshold, top_k, min_degree) combinations on
                          a cleaned static adjacency and plot diagnostics.
                          Entry point: stock-sweep-static-graph

6. sweep_dynamic_graph  – Sweep the same sparsification parameters for dynamic
                          (periodic correlation) graph construction.
                          Entry point: stock-sweep-dynamic-graph

Training, evaluation and testing
---------------------------------
After the data pipeline the standard CLI modules are used:

    python -m graph_signal_diffusion.cli.train  task=stock_price_forecasting_v2
    python -m graph_signal_diffusion.cli.evaluate  dataset=sp500_cleaned ...
    python -m graph_signal_diffusion.cli.compare_baselines --config-name compare_baselines_sp500
    python -m graph_signal_diffusion.cli.test  ...

Shell launchers live in scripts/stock/sp500/ and scripts/stock/sp100/.
"""
