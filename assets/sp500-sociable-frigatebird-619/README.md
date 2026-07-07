# S&P 500 forecasting — `sociable-frigatebird-619`

Probabilistic multi-horizon forecasting of S&P 500 daily log-returns, cast as conditional graph signal
generation. The panel covers N = 468 stocks across 11 GICS sectors over 2353 trading days (2016-09-06
to 2026-02-13, via `yfinance`), on a static graph built from long-term company fundamentals (rank
correlation of fundamental profiles plus a same-sector bonus, thresholded and spectrally normalized).
Each sample conditions on a T<sub>h</sub> = 20-day history of 12 market features per stock and diffuses
the next T<sub>p</sub> = 5 days of log-returns; the 2328 windows are partitioned by an interleaved
chronological scheme (10 chunks, 80/10/10 train/val/test within each). The U-GNN diffusion model
produces a forecast ensemble of 100 trajectories per window, benchmarked against a geometric random
walk (GRW) whose per-stock i.i.d. Gaussian log-returns are fit once on the training set.

## Representative forecast trajectories

![Forecast trajectories — nine held-out windows](fig1_forecast_grid_3x3.png)

Nine stocks across three held-out test windows: Historical (black), realized Target (blue), and the
model's ensemble mean + 90% CI (orange) over the 5-day horizon. The sampled trajectories spread out
from the last observed price, with each sample tracking its stock's recent trend and volatility rather
than flattening or spreading evenly; occasional paths break from the bundle to trace larger excursions.
The columns span a range of behaviour — clean tracked uptrends (AXP/CBRE/COP), a target that escapes
the band upward (BIIB), and mixed up/down/flat (C/INCY/ZTS). Individual panels:
[w00](fig1_forecast_w00_t208.png) · [w12](fig1_forecast_w12_t220.png) ·
[w13](fig1_forecast_w13_t221.png).

## Structural & distributional comparison vs. GRW

![Structural comparison](fig2_structural_comparison.png)

Temporal and cross-sectional structure. U-GNN matches the autocorrelation signature of returns —
near-zero for r, persistent for r² (volatility clustering) — and tracks the leading eigenvalues of the
return covariance that capture the dominant market directions, both of which the GRW baseline misses.

![Distributional comparison](fig3_nll_comparison.png)

Score-based distributional fidelity: per-window scores under the GRW's analytic likelihood (NLL) and
under the trained U-GNN (ELBO estimates), compared to the real data through the Wasserstein-1 distance
between empirical CDFs. U-GNN lands closer to the real distribution under both scorers — even under the
GRW's own likelihood (W<sub>1</sub> = 0.14 vs. 0.23), and nearly coinciding with the real data under
its own score (W<sub>1</sub> = 0.04 vs. 0.14).

## Reproduce

- End-to-end pipeline (data → clean → checkpoint eval → baseline comparison):
  **[reproduce/sp500-sociable-frigatebird-619/](../../reproduce/sp500-sociable-frigatebird-619/)**
- Figure provenance (exact source run, tickers, regeneration): **[SOURCE.md](SOURCE.md)**
