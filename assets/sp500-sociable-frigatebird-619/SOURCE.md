# Source & provenance — `sociable-frigatebird-619` figures

Figures isolated for the top-level README. PNG (200 dpi), versioned in **regular
git** (see `../.gitattributes`); the print-quality PDF masters stay in `outputs/`.

## Source run

```
outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/comparison/
  2026-06-19/18-39-05_sociable-frigatebird-619_epoch-4500_full-test-split_100-samples/
  eval_viz/paper/
```

The `paper/` dir keeps an `.npz` sidecar next to each PDF; the figures are
regenerated from those sidecars (paper-figure replot step). Reproduce the parent
comparison run with `reproduce/sp500-sociable-frigatebird-619/06_compare_baselines.sh`
(ep4500 checkpoint, native 10-chunk test, 100 samples).

## Figures here

| PNG | Source PDF | What it shows |
|---|---|---|
| `fig1_forecast_w00_t208.png` | `fig1_forecast_column_test_w00_t208.pdf` | Trajectories: AEP (tracked uptrend), BIIB (target escapes CI upward), ORLY (sideways) |
| `fig1_forecast_w12_t220.png` | `fig1_forecast_column_test_w12_t220.pdf` | Trajectories: C / INCY / ZTS (mixed up / down / flat) |
| `fig1_forecast_w13_t221.png` | `fig1_forecast_column_test_w13_t221.pdf` | Trajectories: AXP / CBRE / COP (clean, well-calibrated uptrends) |
| `fig2_structural_comparison.png` | `fig2_structural_comparison.pdf` | Return/vol autocorrelation + top-5 correlation-matrix eigenvalues (Real vs GRW vs U-GNN) |
| `fig3_nll_comparison.png` | `fig3_nll_comparison.pdf` | NLL & ELBO histograms/CDFs with Wasserstein-1 distances (Real vs GRW vs U-GNN) |

Each `fig1` column plots Historical (black) / Target (blue) / Predicted mean +
ensemble + 90% CI (orange) for 3 stocks over the forecast horizon. All 20 test
windows (`w00`–`w19`) are available in the source `paper/` dir if a different
representative set is preferred.
