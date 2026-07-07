# Source & provenance — `rigorous-quoll` figures

Figures isolated for the top-level README. PNG (rendered with `pdftoppm`, downsized to
2200 px wide, 256-color), versioned in **regular git** (see `../.gitattributes`); the
print-quality PDF masters stay in `outputs/`.

## Source run

```
outputs/wireless_resource_allocation-wra/
  ugnn_wra_v3_ds8_norm_act_head-ddim_wra-gdm_wra_medium-large_outdoor_all_density/
  rigorous-quoll-131/
```

TSP WRA experiment `rigorous_quoll_131`, U-GNN diffusion model
(`ugnn_wra_v3_ds8_norm_act_head`) trained on the `wra_medium-large_outdoor_all_density`
dataset (ultra-low / low / mid / high density, expert = primal-dual). Reproducibility
anchor: **`repro/wra-tsp` @ `f53f7b0`** (== the retired `wra-local-wip`). Evaluation used
the best checkpoint `best_model_epoch_1600.pt` (DDIM, 100 steps, `ddim_eta=0.2`, 50
samples/input, 500 channel realizations).

The `network_panels/` and `per_density_*` PDFs are regenerated with
`scripts/wra/diffusion/rigorous_quoll_131/visualize_network_panels.sh` from the shipped
checkpoint; wired into the reproduction package as `reproduce/wra-rigorous-quoll/`.

## Figures here

| PNG | Source PDF (under the run dir) | What it shows |
|---|---|---|
| `fig1_network_anatomy_low.png` | `network_panels/low_net0_r0.6/network_panels_full.pdf` | Low-density network 0 in five panels: (i) physical TX–RX deployment, (ii) channel gains (interference edges + direct self-loops), (iii) PyG graph abstraction (node features + edge weights), (iv) PD-expert power allocation as a graph signal, (v) resulting ergodic rates (r_min=0.6, 86.5% feasible). 400 links, top-k=10 interference graph, P_max=0.01 W. |
| `fig1b_network_anatomy_high.png` | `network_panels/high_net0_r0.6/network_panels_full.pdf` | Same five-panel anatomy for a **high-density** network (77.2% feasible) — harder interference regime, lower mean power. |
| `fig2_power_timesharing.png` | `network_panels/low_net0_r0.6/power_allocation_realizations.pdf` | The generative payoff: the diffusion model draws multiple power-allocation realizations (time slots s−1, s, s+1) whose per-slot feasibility is only 56–86%, but **stochastic time-sharing** over the ensemble (mean allocation E[p]) reaches **100% feasibility**. |
| `fig2_power_timesharing_crop.png` | *(same PDF as above)* | The README §3 hero — `fig2_power_timesharing.png` with the figure title and white margins cropped off. |
| `fig2_power_timesharing_zoom.png` | `network_panels/low_net0_r0.6/power_allocation_realizations_zoom_camera.pdf` | Camera-zoom close-up of the same time-sharing figure (19 receivers in view, cross-boundary edges trailed off) — resolves individual per-link powers and rates. Title + margins cropped. |
| `fig3_rate_percentiles.png` | `per_density_rate_percentiles.pdf` | Tail-rate curves over training (min, 0.1-pct, 1-pct, 5-pct bits/s/Hz): generated (solid) vs expert (faded) across the four densities. The dotted line marks the ep-1600 best checkpoint. |

Other densities / networks / r_min values are available under `network_panels/*` in the
source run dir if a different representative set is preferred.
