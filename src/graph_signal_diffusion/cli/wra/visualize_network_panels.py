#!/usr/bin/env python3
"""Render a 5-panel "anatomy of a WRA network" figure for one network.

For a chosen (density, network_id, r_min) of the medium-large outdoor
all-density / all-r_min experiment (sophisticated-oarfish-9), this CLI produces
a combined figure with five panels:

    (i)   Physical TX-RX deployment.
    (ii)  Channel-gain graph: circular nodes at RX positions, directed
          interference edges + direct-gain self-loops (large-scale gains, dB).
    (iii) PyG abstraction: same layout, nodes coloured by the *derived* node
          features [direct_signal, interference_potential], edges by the
          processed ``edge_weight`` the GNN consumes.
    (iv)  An example PD-expert power allocation drawn as a graph signal.
    (v)   The resulting ergodic rates drawn as a graph signal (RXs below r_min
          outlined in red).

A second "zoom" figure mirrors the same five panels over a small spatial
neighbourhood (the N nearest links to a focus point) with per-node labels for
legibility; the zoom region is marked with a dashed box on the full view.

Data sources (all verified to align by ``network_seed``):
    - Channel deployment + large-scale fading + associations come from the
      content-addressed channel cache under ``data/wra_channel_cache/``.
    - The example allocation comes from the primal-dual expert's collected
      ``power_samples`` (``power_per_transmitter`` / ``power_per_receiver``,
      watts in ``[0, P_max]``) under ``data/wra/<scenario>/wrpc_v1_*``.
    - Ergodic rates are recomputed from the chosen allocation and a fresh
      instantaneous-channel realization via ``compute_ergodic_rates``.

Usage:
    python -m graph_signal_diffusion.cli.wra.visualize_network_panels \
        --config-name=network_panels/sophisticated_oarfish_9 \
        density=low network_id=0 r_min=0.6
"""

import json
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

# Importing the channel module registers the WirelessChannel* classes so the
# pickled cache objects unpickle cleanly.
from graph_signal_diffusion.datasets.wra import channel as _wra_channel  # noqa: F401,E402
from graph_signal_diffusion.utils.graph_builder import build_interference_graph  # noqa: E402
from graph_signal_diffusion.utils.rate_calculator import (  # noqa: E402
    compute_ergodic_rates,
    compute_jains_fairness_index,
)

log = logging.getLogger(__name__)

# Marker / colour conventions shared across panels.
TX_COLOR, RX_COLOR, LINK_COLOR = "red", "tab:blue", "green"
# Circular-node marker areas (pt^2). The full network is crowded, so its nodes are
# small; the zoom views show few labelled nodes, so they stay large for legibility.
NODE_SIZE = 180         # zoom (labelled) circular nodes
NODE_SIZE_FULL = 65     # full-view circular nodes (panels ii–v)
NODE_SIZE_REAL = 16     # realizations-figure nodes (panels are ~half the full size, so
                        # ~1/4 the marker area keeps the on-panel node size matched)
NODE_SIZE_REAL_ZOOM = 50  # realizations-figure nodes in the camera-zoom variant (few
                        # nodes in view -> larger, ~1/4 of NODE_SIZE like the full match)
# TX/RX deployment markers (panel i).
DEP_MARKER = 70         # zoom deployment markers
DEP_MARKER_FULL = 32    # full-view deployment markers

# Roomier spacing between each axes and its title / axis labels.
plt.rcParams.update({"axes.titlepad": 18, "axes.labelpad": 12})


class _CenterLogNorm(Normalize):
    """Diverging normalize for the rate panel: **linear** from ``vmin`` to ``vcenter``
    over the lower half [0, 0.5] of the colormap, and **logarithmic** from ``vcenter``
    to ``vmax`` over the upper half [0.5, 1]. This preserves colour resolution for
    rates at/around the ``r_min`` constraint (= ``vcenter``) while compressing the long
    tail of much-larger rates. Requires ``0 <= vmin < vcenter < vmax`` and ``vcenter>0``.
    """

    def __init__(self, vmin, vcenter, vmax, clip=False):
        super().__init__(vmin, vmax, clip)
        self.vcenter = float(vcenter)

    def __call__(self, value, clip=None):
        is_scalar = np.isscalar(value)
        v = np.asarray(value, dtype=float)
        c, lo, hi = self.vcenter, float(self.vmin), float(self.vmax)
        out = np.where(
            v <= c,
            0.5 * (v - lo) / (c - lo),
            0.5 + 0.5 * np.log(np.maximum(v, 1e-12) / c) / np.log(hi / c),
        )
        out = np.clip(out, 0.0, 1.0)
        return float(out) if is_scalar else np.ma.asarray(out)

    def inverse(self, value):
        v = np.asarray(value, dtype=float)
        c, lo, hi = self.vcenter, float(self.vmin), float(self.vmax)
        out = np.where(
            v <= 0.5,
            lo + (v / 0.5) * (c - lo),
            c * (hi / c) ** ((np.clip(v, 0.0, 1.0) - 0.5) / 0.5),
        )
        return out


def _fmt_rate(v: float) -> str:
    v = float(v)
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


class _LogPowerNorm(Normalize):
    """Log normalize for power/P_max: maps [floor, vmax] -> [0, 1] logarithmically and
    clips anything below ``floor`` to 0 (the 'off' end). The PD allocations are heavily
    low-skewed (mean << 0.5, ~quarter of TXs near 0), so a log scale resolves the active
    low-power bulk far better than a linear one. ``floor`` defaults to 1% of P_max — the
    same threshold the meta panel uses to call a TX 'active'."""

    def __init__(self, floor=0.01, vmax=1.0):
        super().__init__(vmin=float(floor), vmax=float(vmax))
        self.floor = float(floor)

    def __call__(self, value, clip=None):
        is_scalar = np.isscalar(value)
        v = np.asarray(value, dtype=float)
        lf, lv = np.log(self.floor), np.log(self.vmax)
        out = np.clip((np.log(np.clip(v, self.floor, self.vmax)) - lf) / (lv - lf), 0.0, 1.0)
        return float(out) if is_scalar else np.ma.asarray(out)

    def inverse(self, value):
        v = np.asarray(value, dtype=float)
        lf, lv = np.log(self.floor), np.log(self.vmax)
        return np.exp(lf + np.clip(v, 0.0, 1.0) * (lv - lf))


def _power_norm_ticks(floor=0.01, vmax=1.0, n=7):
    """(_LogPowerNorm, tick_values, tick_labels) with ticks at *uniform* colour
    positions (so the bar still reads evenly despite the log scale)."""
    norm = _LogPowerNorm(floor, vmax)
    ticks = norm.inverse(np.linspace(0.0, 1.0, n))
    labels = [f"{t:.2g}" for t in ticks]
    labels[0] = f"≤{ticks[0]:.2g}"          # everything below floor is 'off'
    return norm, ticks, labels


def _diverse_indices(feas, n_show):
    """Indices of the ``n_show`` samples with the most diverse feasible/infeasible
    patterns — greedy farthest-point on Hamming distance, anchored at the most-feasible
    sample — so each shown allocation satisfies a different RX subset (the case for
    time-sharing)."""
    K = len(feas)
    if K <= n_show:
        return np.arange(K)
    F = feas.astype(np.int16)
    sel = [int(np.argmax(F.mean(axis=1)))]                  # anchor: most feasible
    mind = (F != F[sel[0]]).sum(axis=1).astype(float)
    for _ in range(n_show - 1):
        mind[sel] = -1.0
        nxt = int(np.argmax(mind))
        sel.append(nxt)
        mind = np.minimum(mind, (F != F[nxt]).sum(axis=1))
    return np.asarray(sel)


def _timestep_labels(n):
    """Illustrative 'time slot s-1 / s / s+1 / ...' labels (mathtext), centred on s.

    The shown allocations are the most *diverse* PD samples, not consecutive iterates,
    so we label them as generic successive time slots of the stochastic policy rather
    than with their (arbitrary) source-step numbers."""
    half = (n - 1) // 2
    return ["time slot $s$" if off == 0 else f"time slot $s{off:+d}$"
            for off in range(-half, n - half)]


# ---------------------------------------------------------------------------
# Path resolution (density + r_min -> on-disk files, no hardcoded hashes)
# ---------------------------------------------------------------------------

def _scenario_name(density: str) -> str:
    return f"medium-large_outdoor_{density}_density"


def _resolve_channel_cache(density: str, root: Path) -> Path:
    """Return the single channel-cache ``.pt`` for *density*."""
    cache_dir = root / "data" / "wra_channel_cache" / f"wra_{_scenario_name(density)}"
    hits = sorted(cache_dir.glob("*.pt"))
    if not hits:
        raise FileNotFoundError(f"No channel cache .pt under {cache_dir}")
    if len(hits) > 1:
        log.warning("Multiple channel caches in %s; using %s", cache_dir, hits[0].name)
    return hits[0]


def _resolve_pd_dir(density: str, r_min: float, root: Path) -> Path:
    """Find the primal-history dir for *density* whose config r_min matches."""
    scenario_root = root / "data" / "wra" / _scenario_name(density)
    candidates = sorted(scenario_root.glob("wrpc_v1_primal_history_k200_*"))
    for cand in candidates:
        cfg_path = cand / "raw" / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        cand_r = float(cfg.get("training", {}).get("r_min", -1))
        if abs(cand_r - float(r_min)) < 1e-9:
            return cand
    avail = []
    for cand in candidates:
        cfg_path = cand / "raw" / "config.yaml"
        if cfg_path.exists():
            avail.append(yaml.safe_load(cfg_path.read_text()).get("training", {}).get("r_min"))
    raise FileNotFoundError(
        f"No primal-history dir for density={density} r_min={r_min} under "
        f"{scenario_root}. Available r_min: {sorted(set(avail))}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_channel(cache_path: Path, network_id: int):
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    channels = blob["channels"]
    if not 0 <= network_id < len(channels):
        raise IndexError(f"network_id={network_id} out of range (cache has {len(channels)})")
    return channels[network_id]


def _system_params(pd_dir: Path) -> dict:
    cfg = yaml.safe_load((pd_dir / "raw" / "config.yaml").read_text())
    sysd = cfg["system"]
    p_max = 10 ** ((float(sysd["P_max_dBm"]) - 30) / 10)
    noise_var = 10 ** ((float(sysd["noise_psd_dBm_Hz"]) + 10 * np.log10(float(sysd["bandwidth_Hz"])) - 30) / 10)
    return {
        "P_max": p_max,
        "noise_var": noise_var,
        "r_min": float(cfg["training"]["r_min"]),
        "num_timesteps_collected": int(cfg["dataset"].get("num_timesteps", 0)),
        # Max TX-RX pairing distance (a deployment-config quantity) — the interference
        # disk radius is derived from this (e.g. 2x = 200 m).
        "max_tx_rx_distance": float(cfg.get("channel", {}).get("max_tx_rx_distance", 0.0)),
    }


def _load_power_samples(pd_dir: Path, network_id: int, channel_seed: int):
    """Return the *full* set of PD-expert samples for one network:
    (p_tx (K,m), p_rx (K,n) | None, steps (K,), stored_rates (K,n) | None)."""
    npz = np.load(pd_dir / "raw" / "collected_samples.npz", allow_pickle=True)
    net = npz[f"network_{network_id}"].item()
    if int(net.get("network_seed", -1)) != int(channel_seed):
        # Fall back to a seed match if positional indexing diverges.
        for k in npz.files:
            if k.startswith("network_") and int(npz[k].item().get("network_seed", -2)) == int(channel_seed):
                net = npz[k].item()
                break
        else:
            raise ValueError(
                f"PD network seed {net.get('network_seed')} != channel seed {channel_seed}"
            )
    samples = net["power_samples"]  # list of dicts
    steps = np.asarray(net["sample_steps"])
    p_tx = np.stack([np.asarray(s["power_per_transmitter"], float) for s in samples])  # (K,m)
    p_rx = (np.stack([np.asarray(s["power_per_receiver"], float) for s in samples])
            if "power_per_receiver" in samples[0] else None)
    stored_rates = (np.stack([np.asarray(s["rates"], float) for s in samples])
                    if "rates" in samples[0] else None)
    return p_tx, p_rx, steps, stored_rates


def _load_allocation(pd_dir: Path, network_id: int, channel_seed: int,
                     selection: str, r_min: float):
    """Return (power_per_tx (m,), power_per_rx (n,), source_step, info)."""
    p_tx, p_rx, steps, stored_rates = _load_power_samples(pd_dir, network_id, channel_seed)

    sel = str(selection)
    if sel == "mean":
        idx, src = None, "mean"
        chosen_tx = p_tx.mean(0)
        chosen_rx = p_rx.mean(0) if p_rx is not None else None
    else:
        if sel == "best":
            if stored_rates is None:
                idx = int(np.argmax(steps))
            else:
                idx = int(np.argmax((stored_rates >= r_min).mean(axis=1)))
        elif sel == "last":
            idx = int(np.argmax(steps))
        else:
            idx = int(sel)
        chosen_tx = p_tx[idx]
        chosen_rx = p_rx[idx] if p_rx is not None else None
        src = int(steps[idx])

    info = {
        "num_samples": len(steps),
        "selection": sel,
        "selected_index": idx,
        "source_step": src,
        "step_range": [int(steps.min()), int(steps.max())],
    }
    return chosen_tx, chosen_rx, src, info


# ---------------------------------------------------------------------------
# Graph / signal derivation
# ---------------------------------------------------------------------------

def _edges_split(graph):
    """Return (interf_edges (2,E), interf_idx, self_node_idx) from edge_index."""
    ei = graph.edge_index.numpy()
    self_mask = ei[0] == ei[1]
    interf = ei[:, ~self_mask]
    interf_idx = np.where(~self_mask)[0]
    self_nodes = ei[0, self_mask]
    return interf, interf_idx, self_nodes


def _sample_channel(channel, num_timesteps):
    """Instantaneous-channel realization as a torch tensor of shape (T, m, n)."""
    H = np.asarray(channel.sample_realization(num_timesteps=num_timesteps)["H"], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(H.transpose(2, 0, 1)))  # (T, m, n)


def _rates_from_H(power_per_tx, Ht, associations, noise_var):
    """Ergodic (time-averaged) per-RX rates for a *constant* power vector over Ht."""
    return compute_ergodic_rates(
        torch.as_tensor(power_per_tx, dtype=torch.float32), Ht,
        torch.as_tensor(associations, dtype=torch.float32), float(noise_var)).numpy()


def _ergodic_rates(channel, power_per_tx, associations, noise_var, num_timesteps):
    return _rates_from_H(power_per_tx, _sample_channel(channel, num_timesteps),
                         associations, noise_var)


def _per_sample_ergodic_rates(p_tx_all, Ht, associations, noise_var):
    """(K, n) — the ergodic rate of *each* constant allocation, all evaluated over the
    same instantaneous-channel realization ``Ht`` so they are directly comparable. The
    mean over the K rows is the ergodic rate of the **stochastic time-sharing policy**
    (allocation drawn from the K samples independently of the fading)."""
    assoc_t = torch.as_tensor(associations, dtype=torch.float32)
    return np.stack([
        compute_ergodic_rates(torch.as_tensor(p, dtype=torch.float32), Ht, assoc_t,
                              float(noise_var)).numpy()
        for p in p_tx_all
    ])


def _select_zoom(rx_locs, graph, focus, n_links, interior_frac=0.15):
    """Return (node_indices, center_xy) for the spatial-crop zoom.

    For the default ``highest_interference`` focus, the candidate is restricted to the
    network *interior* (inset by ``interior_frac`` of the extent on each side), so the
    zoom neighbourhood is surrounded by nodes and the window doesn't hang off a graph
    edge/corner into empty space. An explicit RX index / xy is honoured as given.
    """
    if isinstance(focus, str) and focus == "highest_interference":
        interf_pot = graph.x[:, 1].numpy()
        lo, hi = rx_locs.min(axis=0), rx_locs.max(axis=0)
        margin = interior_frac * (hi - lo)
        interior = np.all((rx_locs >= lo + margin) & (rx_locs <= hi - margin), axis=1)
        cand = np.where(interior)[0]
        if len(cand) == 0:
            cand = np.arange(len(rx_locs))
        center = rx_locs[cand[int(np.argmax(interf_pot[cand]))]]
    elif isinstance(focus, (list, tuple)) and len(focus) == 2:
        center = np.asarray(focus, dtype=float)
    else:  # explicit RX index
        center = rx_locs[int(focus)]
    d = np.linalg.norm(rx_locs - center[None, :], axis=1)
    idx = np.argsort(d)[: int(n_links)]
    return np.sort(idx), center


# ---------------------------------------------------------------------------
# Panel renderers (each draws onto a given Axes for a given node subset)
# ---------------------------------------------------------------------------

def _faint_edges(ax, node_pos, interf, weights=None, mask=None):
    """Draw the interference edges as gray context. With *weights*, each edge's **gray
    level, opacity and width** all scale with its weight, so strong edges read as dark
    thick lines and weak ones recede — the colour stays achromatic (gray) so it never
    competes with the node colormap. Edge weights span orders of magnitude, so they are
    mapped on a **log scale** with robust 5–95th-percentile bounds over the *drawn*
    edges (keeping contrast in both the full plot and a zoom), plus a mild gamma so
    weaker edges stay visible while the strong tail still stands out."""
    ds, dt, dw = [], [], []
    for k, (s, t) in enumerate(zip(interf[0], interf[1])):
        if mask is not None and not (mask[s] and mask[t]):
            continue
        ds.append(s); dt.append(t)
        if weights is not None:
            dw.append(float(weights[k]))
    if not ds:
        return
    if weights is not None and len(dw):
        lw_log = np.log(np.maximum(np.asarray(dw), 1e-12))
        lo, hi = np.percentile(lw_log, 5), np.percentile(lw_log, 95)
        vv = np.clip((lw_log - lo) / ((hi - lo) or 1.0), 0.0, 1.0) ** 1.5
    segs, cols, lws = [], [], []
    for i, (s, t) in enumerate(zip(ds, dt)):
        if weights is not None:
            v = float(vv[i])
            lum = 0.70 * (1.0 - v)              # medium-light gray (weak) -> black (strong)
            alpha = 0.32 + 0.60 * v             # weak edges stay clearly visible
            lw = 0.5 + 2.5 * v
        else:
            lum, alpha, lw = 0.30, 0.5, 0.3
        segs.append([node_pos[s], node_pos[t]])
        cols.append((lum, lum, lum, alpha))
        lws.append(lw)
    lc = LineCollection(segs, colors=cols, linewidths=lws, zorder=1)
    lc.set_in_layout(False)
    ax.add_collection(lc)


def _draw_interf_edges(ax, node_pos, interf, values, cmap, mask=None, directed=False,
                       log_scale=False, gamma=1.0, cmin=0.12):
    """Draw interference edges with **colour + width + opacity** all scaled by *values*,
    so strong edges read dark/thick/opaque and weak ones light/thin/faint. *values* are
    mapped on **robust 5–95th-percentile bounds** over the drawn edges (so a few outliers
    don't compress the bulk into one colour), optionally in **log space** (``log_scale``,
    for raw weights that span orders of magnitude) and with an emphasis ``gamma``.
    Returns a Normalize over the original value range for an optional colorbar."""
    drawn = [k for k, (s, t) in enumerate(zip(interf[0], interf[1]))
             if mask is None or (mask[s] and mask[t])]
    if not drawn:
        return Normalize(0, 1)
    raw = np.asarray([values[k] for k in drawn], dtype=float)
    tv = np.log(np.maximum(raw, 1e-12)) if log_scale else raw
    lo, hi = float(np.percentile(tv, 5)), float(np.percentile(tv, 95))
    rng = (hi - lo) or 1.0
    vmap = np.clip((tv - lo) / rng, 0.0, 1.0) ** gamma
    segs, cols, lws = [], [], []
    for j, k in enumerate(drawn):
        s, t = int(interf[0][k]), int(interf[1][k])
        v = float(vmap[j])
        r, g, b = cmap(cmin + (1.0 - cmin) * v)[:3]
        if directed:
            ann = ax.annotate("", xy=node_pos[t], xytext=node_pos[s],
                              arrowprops=dict(arrowstyle="->", color=(r, g, b),
                                              alpha=0.22 + 0.73 * v, lw=0.4 + 3.4 * v,
                                              connectionstyle="arc3,rad=0.12"), zorder=2)
            # Clip the arrow to the axes box so cross-boundary edges (camera zoom)
            # trail off cleanly instead of painting into the margins / colorbars —
            # matching how the LineCollection edges in the other panels behave.
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_clip_on(True)
                ann.arrow_patch.set_clip_box(ax.bbox)
            # Exclude from layout/tight-bbox: an arrow whose far endpoint sits far
            # outside the (camera-zoom) view would otherwise balloon the figure's
            # tight bbox and shrink every panel. The nodes define the bbox.
            ann.set_in_layout(False)
        else:
            segs.append([node_pos[s], node_pos[t]])
            cols.append((r, g, b, 0.20 + 0.70 * v))   # per-edge opacity via RGBA
            lws.append(0.3 + 3.0 * v)
    if segs:
        lc = LineCollection(segs, colors=cols, linewidths=lws, zorder=2)
        lc.set_in_layout(False)
        ax.add_collection(lc)
    if log_scale:
        return Normalize(vmin=float(np.exp(lo)), vmax=float(np.exp(hi)))
    return Normalize(vmin=lo, vmax=hi)


def _self_loop_rings(ax, node_pos, self_nodes, values, cmap, mask=None, span=20.0):
    norm = Normalize(vmin=np.min(values), vmax=np.max(values)) if len(values) else Normalize(0, 1)
    for j in self_nodes:
        if mask is not None and not mask[j]:
            continue
        s = norm(values[j])
        ax.add_patch(plt.Circle((node_pos[j, 0], node_pos[j, 1] + span), radius=span * 0.6,
                                 color=cmap(0.4 + 0.6 * s), fill=False, lw=1.5 + 2 * s, zorder=3))
    return norm


def _labels(ax, node_pos, idx):
    for j in idx:
        ax.annotate(str(int(j)), node_pos[j], ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def _panel_deployment(ax, channel, draw_mask=None, label_idx=None,
                      interf_radius=None, disk_alpha=0.10, marker_size=DEP_MARKER):
    tx, rx = np.asarray(channel.tx_locations), np.asarray(channel.rx_locations)
    pairs = np.asarray(channel.tx_rx_pairs)
    # map drawn RX -> their serving TX
    if draw_mask is None:
        tshow = np.ones(len(tx), bool); rshow = np.ones(len(rx), bool)
    else:
        rshow = draw_mask
        tshow = np.zeros(len(tx), bool)
        for ti, ri in pairs:
            if draw_mask[ri]:
                tshow[ti] = True
    # Transparent interference disks centred on each TX (constant physical radius,
    # default = the path-loss dual-slope breakpoint). Overlaps / RXs falling inside a
    # disk make interference coupling legible from the deployment alone. Drawn first.
    if interf_radius and interf_radius > 0:
        for ti in np.where(tshow)[0]:
            ax.add_patch(plt.Circle((tx[ti, 0], tx[ti, 1]), radius=float(interf_radius),
                                    facecolor=TX_COLOR, edgecolor=TX_COLOR, alpha=disk_alpha,
                                    lw=0.6, zorder=0))
    for ti, ri in pairs:
        if draw_mask is not None and not draw_mask[ri]:
            continue
        ax.plot([tx[ti, 0], rx[ri, 0]], [tx[ti, 1], rx[ri, 1]], "-",
                color=LINK_COLOR, lw=1.8, alpha=0.6, zorder=2)
    ax.scatter(tx[tshow, 0], tx[tshow, 1], c=TX_COLOR, s=marker_size, marker="^",
               edgecolors="black", linewidths=1.0, zorder=4)
    ax.scatter(rx[rshow, 0], rx[rshow, 1], c=RX_COLOR, s=marker_size, marker="o",
               edgecolors="black", linewidths=1.0, zorder=4)
    if label_idx is not None:
        _labels(ax, rx, label_idx)
    # Legend lists only the markers; the interference-range disks are self-evident
    # from the plot and are intentionally left out of the legend.
    handles = [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=TX_COLOR,
                   markeredgecolor="black", markersize=9, linestyle="None", label="Transmitters"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RX_COLOR,
                   markeredgecolor="black", markersize=9, linestyle="None", label="Receivers"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_title("(i) Physical TX–RX deployment", fontweight="bold")


def _panel_gain_graph(ax, channel, graph, rx, gains, draw_mask=None, label_idx=None,
                      span=20.0, directed=False, node_size=NODE_SIZE):
    interf, interf_idx, self_nodes = graph["interf"], graph["interf_idx"], graph["self_nodes"]
    n_int = _draw_interf_edges(ax, rx, interf, gains["interf_db"], plt.cm.Reds,
                               mask=draw_mask, directed=directed)
    n_dir = _self_loop_rings(ax, rx, self_nodes, gains["direct_db"], plt.cm.Blues,
                             mask=draw_mask, span=span)
    show = np.ones(len(rx), bool) if draw_mask is None else draw_mask
    ax.scatter(rx[show, 0], rx[show, 1], c="0.75", s=node_size, marker="o",
               edgecolors="black", linewidths=1.2, zorder=4)
    if label_idx is not None:
        _labels(ax, rx, label_idx)
    _add_cbar(ax, plt.cm.Reds, n_int, "Interference gain (dB)", pad=0.03)
    _add_cbar(ax, plt.cm.Blues, n_dir, "Direct gain (dB)", pad=0.16, label_side="left")
    ax.set_title("(ii) Channel gains: interference edges + direct self-loops", fontweight="bold")


def _panel_pyg(ax, graph_data, graph, rx, draw_mask=None, label_idx=None,
               span=20.0, directed=False, node_size=NODE_SIZE):
    # Match the classic "*_deployment" right-panel aesthetic: red *curved directed*
    # interference arrows (always, regardless of the figure's directed flag), hollow
    # orange nodes tinted by interference potential, blue direct-signal rings.
    interf, self_nodes = graph["interf"], graph["self_nodes"]
    ew = graph["edge_weight_interf"]
    # Raw edge weights span orders of magnitude -> log scale so the modulation is visible.
    _draw_interf_edges(ax, rx, interf, ew, plt.cm.Reds, mask=draw_mask, directed=True,
                       log_scale=True)
    direct_feat = graph_data.x[:, 0].numpy()
    interf_feat = graph_data.x[:, 1].numpy()
    n_dir = _self_loop_rings(ax, rx, self_nodes, direct_feat, plt.cm.Blues, mask=draw_mask, span=span)
    show = np.ones(len(rx), bool) if draw_mask is None else draw_mask
    sc = ax.scatter(rx[show, 0], rx[show, 1], c=interf_feat[show], cmap="Oranges",
                    s=node_size, marker="o", edgecolors="black", linewidths=1.2, zorder=4)
    if label_idx is not None:
        _labels(ax, rx, label_idx)
    _add_cbar(ax, "Oranges", Normalize(interf_feat[show].min(), interf_feat[show].max()),
              "Node: interference potential", mappable=sc, pad=0.03)
    _add_cbar(ax, plt.cm.Blues, n_dir, "Ring: direct signal", pad=0.16, label_side="left")
    ax.set_title("(iii) PyG abstraction: node features + edge weights", fontweight="bold")


def _panel_signal(ax, rx, graph, values, cmap, title, cbar_label,
                  draw_mask=None, label_idx=None, threshold=None, vmin=None, vmax=None,
                  norm=None, below_edgecolor="red", below_lw=2.4, node_size=NODE_SIZE,
                  center_line=None, cbar_ticks=None, cbar_ticklabels=None):
    """Draw a per-RX graph signal. ``norm`` (e.g. a diverging/log norm) takes
    precedence over vmin/vmax. ``threshold`` outlines nodes below it (e.g. infeasible
    RXs) in ``below_edgecolor``; ``center_line`` marks a reference value on the bar
    (positioned by colour, so it is correct for any norm). ``cbar_ticks`` /
    ``cbar_ticklabels`` override the colorbar ticks (data values + their labels)."""
    _faint_edges(ax, rx, graph["interf"], weights=graph.get("edge_weight_interf"), mask=draw_mask)
    show = np.ones(len(rx), bool) if draw_mask is None else draw_mask
    edgecol = "black"
    lw = 1.2
    if threshold is not None:
        below = values < threshold
        edgecol = np.where(below[show], below_edgecolor, "black")
        lw = np.where(below[show], below_lw, 0.9)
    skw = dict(c=values[show], cmap=cmap, s=node_size, marker="o",
               edgecolors=edgecol, linewidths=lw, zorder=4)
    if norm is not None:
        skw["norm"] = norm
    else:
        skw["vmin"] = values[show].min() if vmin is None else vmin
        skw["vmax"] = values[show].max() if vmax is None else vmax
    sc = ax.scatter(rx[show, 0], rx[show, 1], **skw)
    if label_idx is not None:
        _labels(ax, rx, label_idx)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)
    if cbar_ticks is not None:
        cb.set_ticks(list(cbar_ticks))
        if cbar_ticklabels is not None:
            cb.set_ticklabels(list(cbar_ticklabels))
    if center_line is not None:
        # Position the reference line by colour-fraction so it lands on r_min for
        # any norm (linear, two-slope, or the linear-below/log-above rate norm).
        cpos = float(norm(center_line)) if norm is not None else None
        if cpos is not None:
            cb.ax.plot([0, 1], [cpos, cpos], transform=cb.ax.transAxes,
                       color="black", lw=1.4, ls="--", clip_on=False)
        else:
            cb.ax.axhline(center_line, color="black", lw=1.4, ls="--")
    ax.set_title(title, fontweight="bold")


def _add_cbar(ax, cmap, norm, label, pad=0.04, mappable=None, label_side="right",
              fraction=0.04):
    if mappable is None:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
    cb = plt.colorbar(mappable, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=9)
    if label_side == "left":
        # Inner (plot-adjacent) colorbar: put its ticks + label on the LEFT so the
        # label doesn't float over the outer colorbar and the two bars can sit close.
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position("left")
    return cb


def _style_spatial(ax, xlim, ylim):
    ax.set(xlabel="X (m)", ylabel="Y (m)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)


def _dashed_box(ax, bbox):
    (x0, x1, y0, y1) = bbox
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="purple", linestyle="--", lw=2, zorder=7))


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _build_figure(channel, graph_data, graph, rx, gains, power_rx, rates, meta,
                  draw_mask=None, label_idx=None, view_bbox=None, directed=False,
                  mark_bbox=None, span=20.0, scope="full",
                  node_size=NODE_SIZE_FULL, dep_marker=DEP_MARKER_FULL):
    """Render the 5-panel figure.

    draw_mask  : bool array of nodes to draw (None = whole network).
    label_idx  : node indices to annotate (None = none).
    view_bbox  : (xlim, ylim) axis window (None = fit to drawn nodes).
    directed   : draw interference edges as arrows (True) or lines (False).
    mark_bbox  : (x0, x1, y0, y1) dashed rectangle to overlay (None = none).
    node_size  : circular-node marker area (small for full, large for zoom).
    dep_marker : TX/RX marker area in the deployment panel.
    """
    # Width 29 (not 24): the 2-colorbar panels (ii)/(iii) then have enough room — even
    # with the inner colorbar's ticks/label on its left and a clear margin from the plot
    # — that their equal-aspect square is limited by height (like the 0/1-colorbar
    # panels), so all five plots come out the same size, and the colorbars (also
    # cell-height) line up with the plot instead of overshooting it.
    fig, axes = plt.subplots(2, 3, figsize=(29, 15))
    P_max, r_min = meta["P_max"], meta["r_min"]

    if view_bbox is not None:
        xlim, ylim = view_bbox
    else:
        pts = rx if draw_mask is None else rx[draw_mask]
        pad = 0.04 * (pts.max(0) - pts.min(0))
        xlim = (pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
        ylim = (pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])

    _panel_deployment(axes[0, 0], channel, draw_mask, label_idx,
                      meta.get("interf_radius"), meta.get("disk_alpha", 0.10),
                      marker_size=dep_marker)
    _panel_gain_graph(axes[0, 1], channel, graph, rx, gains, draw_mask, label_idx, span,
                      directed, node_size=node_size)
    _panel_pyg(axes[0, 2], graph_data, graph, rx, draw_mask, label_idx, span,
               directed, node_size=node_size)
    # (iv) PD-expert allocation — plasma on a LOG power scale (allocations are heavily
    # low-skewed; a linear scale crushes the active low-power bulk into one dark shade).
    # Ticks stay at uniform colour positions despite the log scale.
    pnorm, ptick, plabels = _power_norm_ticks()
    _panel_signal(axes[1, 0], rx, graph, power_rx / P_max, "plasma",
                  "(iv) PD-expert allocation (graph signal)", "Tx power / P_max",
                  draw_mask, label_idx, norm=pnorm, node_size=node_size,
                  cbar_ticks=ptick, cbar_ticklabels=plabels)
    # (v) Ergodic rates — diverging red→neutral→green centred on the r_min constraint:
    # dark red well below r_min, neutral (pale) at r_min, deep green far above it. The
    # upper (> r_min) half of the colour scale is LOGARITHMIC, so resolution stays high
    # at/around the constraint instead of being eaten by the long high-rate tail.
    frac = float(np.mean(rates >= r_min))
    show = np.ones(len(rx), bool) if draw_mask is None else draw_mask
    rv = rates[show]
    c = float(r_min)
    lo, hi = float(rv.min()), float(rv.max())
    span_r = max(hi - lo, 1e-3)
    lo = max(0.0, min(lo, c - 0.02 * span_r))    # 0 <= vmin < r_min
    hi = max(hi, c * 1.05)                        # vmax > r_min (needed for the log half)
    rate_norm = _CenterLogNorm(vmin=lo, vcenter=c, vmax=hi)
    # Ticks uniform in colour-position: 3 below r_min, r_min, 3 above (lower ones linear
    # in rate, upper ones log-spaced) — symmetric resolution on both sides of r_min.
    tick_vals = rate_norm.inverse(np.linspace(0.0, 1.0, 7))
    tick_vals[3] = c                             # exact r_min at the centre tick
    _panel_signal(axes[1, 1], rx, graph, rates, "RdYlGn",
                  f"(v) Ergodic rates  (r_min={r_min}, {frac:.0%} feasible)",
                  "Rate (bits/s/Hz)", draw_mask, label_idx, threshold=r_min,
                  norm=rate_norm, below_edgecolor="black", below_lw=2.2,
                  node_size=node_size, center_line=c,
                  cbar_ticks=tick_vals, cbar_ticklabels=[_fmt_rate(t) for t in tick_vals])

    for ax in axes.flat[:5]:
        _style_spatial(ax, xlim, ylim)
        if mark_bbox is not None:
            _dashed_box(ax, mark_bbox)

    # Full view: widen the deployment panel so TX outside the RX bbox stay visible.
    if view_bbox is None and draw_mask is None:
        tx = np.asarray(channel.tx_locations)
        allp = np.vstack([tx, rx])
        pad = 0.04 * (allp.max(0) - allp.min(0))
        axes[0, 0].set_xlim(allp[:, 0].min() - pad[0], allp[:, 0].max() + pad[0])
        axes[0, 0].set_ylim(allp[:, 1].min() - pad[1], allp[:, 1].max() + pad[1])

    _meta_panel(axes[1, 2], meta, rates, power_rx)
    fig.suptitle(
        f"WRA network anatomy — {meta['density']} density, network {meta['network_id']}, "
        f"r_min={r_min}  [{scope}]", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5, w_pad=3.5, h_pad=3.5)
    return fig


def _meta_panel(ax, meta, rates, power_rx):
    ax.axis("off")
    P_max, r_min = meta["P_max"], meta["r_min"]
    lines = [
        "RUN",
        f"  experiment      : {meta['experiment']}",
        f"  density / net   : {meta['density']} / {meta['network_id']}  (seed {meta['seed']})",
        f"  r_min           : {r_min} bits/s/Hz",
        f"  allocation      : {meta['selection']} (idx {meta['selected_index']}, step {meta['source_step']})",
        "",
        "NETWORK",
        f"  links (TX-RX)   : {meta['n_links']}",
        f"  interf. range   : {meta.get('interf_radius', 0):.0f} m  ("
        f"{meta.get('interf_radius_mult', 0):.0f}x max TX-RX sep = {meta.get('max_tx_rx_distance', 0):.0f} m)",
        f"  interf. edges   : {meta['n_interf_edges']}  (top_k={meta['top_k']})",
        f"  P_max           : {P_max:.4g} W   noise_var: {meta['noise_var']:.3g} W",
        "",
        "RATES (ergodic, bits/s/Hz)",
        f"  mean / min      : {rates.mean():.3f} / {rates.min():.3f}",
        f"  p5  / median    : {np.percentile(rates, 5):.3f} / {np.median(rates):.3f}",
        f"  feasible (>=r_min): {np.mean(rates >= r_min):.1%}",
        f"  Jain fairness   : {compute_jains_fairness_index(rates):.3f}",
        "",
        "ALLOCATION",
        f"  mean power/P_max: {np.mean(power_rx) / P_max:.3f}",
        f"  active (>1% Pmax): {np.mean(power_rx > 0.01 * P_max):.1%}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=11, transform=ax.transAxes)


def _panel_graph_signal(ax, rx, graph, values, cmap, norm, title, node_size,
                        threshold=None):
    """Compact spatial graph-signal panel (no per-panel colorbar) for the
    realizations figure: faint weight-modulated edges + nodes coloured by *values*."""
    _faint_edges(ax, rx, graph["interf"], weights=graph.get("edge_weight_interf"))
    lw = 0.3
    if threshold is not None:
        lw = np.where(values < threshold, 1.0, 0.3)   # outline infeasible RXs thicker
    sc = ax.scatter(rx[:, 0], rx[:, 1], c=values, cmap=cmap, norm=norm, s=node_size,
                    marker="o", edgecolors="black", linewidths=lw, zorder=4)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    return sc


def _build_realizations_figure(rx, graph, p_rx_show, rates_show, labels_show,
                               p_rx_mean, rate_timeshare, meta, node_size,
                               view_bbox=None):
    """Compare several *constant* policies with the stochastic time-sharing policy.

    A 2 x (N+1) grid: top row = power-allocation vectors (plasma), bottom row = the
    ergodic rates each one yields *if held constant for all time* (RdYlGn, centred on
    r_min). The last column is the stochastic policy: the mean allocation E[p] (top)
    and the **time-sharing ergodic rate** (bottom) = the average over all collected
    allocations of their individual ergodic rates.

    ``view_bbox`` ((xlim, ylim)) crops every panel to that window (camera-zoom variant):
    the full network is still drawn and coloured on the same scales, so cross-boundary
    edges trail off-panel and the colours match the full figure 1:1.
    """
    P_max, r_min = meta["P_max"], meta["r_min"]
    ncol = len(p_rx_show) + 1
    # Width coefficient per column is kept close to the square panel width (~2.8in) so
    # tight_layout doesn't pad each equal-aspect square out into a wide cell — otherwise
    # most of each column becomes horizontal gap. +1.2 reserves the right colorbar margin.
    # Height (8.2in for 2 rows) is likewise sized to the squares: extra height would just
    # become vertical slack between the allocation and rate rows.
    fig, axes = plt.subplots(2, ncol, figsize=(3.3 * ncol + 1.2, 8.2), squeeze=False)

    # Shared diverging/log rate norm over every displayed rate, so panels compare 1:1.
    all_rates = np.concatenate(list(rates_show) + [rate_timeshare])
    c = float(r_min)
    span = max(float(all_rates.max() - all_rates.min()), 1e-3)
    lo = max(0.0, min(float(all_rates.min()), c - 0.02 * span))
    hi = max(float(all_rates.max()), c * 1.05)
    rate_norm = _CenterLogNorm(vmin=lo, vcenter=c, vmax=hi)
    tick_vals = rate_norm.inverse(np.linspace(0.0, 1.0, 7)); tick_vals[3] = c

    allocs = list(p_rx_show) + [p_rx_mean]
    alloc_titles = [f"allocation · {lbl}" for lbl in labels_show] + ["mean allocation  E[p]"]
    rates_all = list(rates_show) + [rate_timeshare]
    rate_titles = [f"rate · {lbl}  ({np.mean(r >= r_min):.0%} feas)"
                   for lbl, r in zip(labels_show, rates_show)]
    rate_titles.append(f"TIME-SHARING rate  ({np.mean(rate_timeshare >= r_min):.0%} feas)")

    pnorm, ptick, plabels = _power_norm_ticks()        # shared log power scale, top row
    sc_top = sc_bot = None
    for j in range(ncol):
        sc_top = _panel_graph_signal(axes[0, j], rx, graph, allocs[j] / P_max, "plasma",
                                     pnorm, alloc_titles[j], node_size)
        sc_bot = _panel_graph_signal(axes[1, j], rx, graph, rates_all[j], "RdYlGn",
                                     rate_norm, rate_titles[j], node_size, threshold=r_min)

    if view_bbox is not None:
        xlim, ylim = view_bbox                      # camera crop (same window as panels zoom)
    else:
        px = 0.04 * np.ptp(rx, axis=0)
        xlim = (rx[:, 0].min() - px[0], rx[:, 0].max() + px[0])
        ylim = (rx[:, 1].min() - px[1], rx[:, 1].max() + px[1])
    for i in range(2):
        for j in range(ncol):
            ax = axes[i, j]
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
            # Label every panel (not just the edges) with X/Y + ticks, so each panel
            # stays self-contained when cropped out on its own.
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            # Visually separate the stochastic-policy column.
            if j == ncol - 1:
                for sp in ax.spines.values():
                    sp.set_edgecolor("purple"); sp.set_linewidth(2.0)

    if view_bbox is not None:
        in_view = int(((rx[:, 0] >= xlim[0]) & (rx[:, 0] <= xlim[1]) &
                       (rx[:, 1] >= ylim[0]) & (rx[:, 1] <= ylim[1])).sum())
        scope_txt = f"camera zoom · {in_view} RXs in view, cross-boundary edges trail off"
    else:
        scope_txt = f"{meta.get('num_samples', '?')} PD samples, time-share over all"
    fig.suptitle(
        f"WRA power-allocation realizations vs stochastic time-sharing — "
        f"{meta['density']} density, network {meta['network_id']}, r_min={r_min}  "
        f"({scope_txt})",
        fontsize=15, fontweight="bold")
    # Lay out the panels first (reserving the right margin), then add one shared
    # colorbar per row in that margin — fig.colorbar(ax=row) misplaces itself against
    # equal-aspect axes, so we position dedicated colorbar axes by hand.
    fig.tight_layout(rect=[0, 0, 0.93, 0.95], pad=2.0, w_pad=1.0, h_pad=1.0)
    fig.canvas.draw()                              # finalize equal-aspect positions
    p_top, p_bot = axes[0, 0].get_position(), axes[1, 0].get_position()
    cb1 = fig.colorbar(sc_top, cax=fig.add_axes([0.94, p_top.y0, 0.009, p_top.height]))
    cb1.set_label(r"Normalized tx power $\mathbf{p} / P_{\max}$", fontsize=9)
    cb1.set_ticks(list(ptick))
    cb1.set_ticklabels(plabels)
    cb2 = fig.colorbar(sc_bot, cax=fig.add_axes([0.94, p_bot.y0, 0.009, p_bot.height]))
    cb2.set_label("Rate (bits/s/Hz)", fontsize=9)
    cb2.set_ticks(list(tick_vals))
    cb2.set_ticklabels([_fmt_rate(t) for t in tick_vals])
    cb2.ax.plot([0, 1], [0.5, 0.5], transform=cb2.ax.transAxes, color="black",
                lw=1.4, ls="--", clip_on=False)   # r_min reference at colour-centre
    return fig


def _save(fig, stem: Path, root: Path):
    """Save *fig* as PDF + PNG under *stem* and close it."""
    fig.savefig(stem.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("✓ Saved %s", stem.relative_to(root).with_suffix(".pdf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf/wra_generation",
            config_name="network_panels/sophisticated_oarfish_9")
def main(cfg: DictConfig):
    root = Path(get_original_cwd())
    density = str(cfg.density)
    network_id = int(cfg.network_id)
    r_min = float(cfg.r_min)
    gcfg = cfg.graph

    log.info("Resolving data for density=%s network=%d r_min=%s", density, network_id, r_min)
    cache_path = _resolve_channel_cache(density, root)
    pd_dir = _resolve_pd_dir(density, r_min, root)
    log.info("  channel cache : %s", cache_path.relative_to(root))
    log.info("  pd dataset    : %s", pd_dir.relative_to(root))

    channel = _load_channel(cache_path, network_id)
    sysp = _system_params(pd_dir)
    assoc = np.asarray(channel.associations, dtype=np.float64)
    H_ls = np.asarray(channel.large_scale_fading, dtype=np.float64)
    rx = np.asarray(channel.rx_locations, dtype=np.float64)
    tx_assoc = np.argmax(assoc, axis=0)  # per-RX -> serving TX

    # ----- PyG graph + derived edge quantities --------------------------------
    graph_data = build_interference_graph(
        H_ls=H_ls, associations=assoc, top_k=int(gcfg.top_k),
        min_degree=int(gcfg.min_degree), normalize_method=str(gcfg.normalize_method))
    interf, interf_idx, self_nodes = _edges_split(graph_data)
    ew = graph_data.edge_weight.numpy()
    graph = {"interf": interf, "interf_idx": interf_idx, "self_nodes": self_nodes,
             "edge_weight_interf": ew[interf_idx]}

    # Raw large-scale gains (dB) for panel (ii): src=interferer-RX, tgt=victim-RX.
    eps = 1e-30
    interf_gain = H_ls[tx_assoc[interf[0]], interf[1]]
    direct_gain = H_ls[tx_assoc[np.arange(len(rx))], np.arange(len(rx))]
    gains = {"interf_db": 10 * np.log10(interf_gain + eps),
             "direct_db": 10 * np.log10(direct_gain + eps)}

    # ----- Allocation + ergodic rates -----------------------------------------
    power_tx, power_rx, source_step, ainfo = _load_allocation(
        pd_dir, network_id, int(channel.seed), str(cfg.sample_selection), r_min)
    if power_rx is None:  # derive RX-indexed power from TX-indexed if absent
        power_rx = power_tx[tx_assoc]
    rates = _ergodic_rates(channel, power_tx, assoc, sysp["noise_var"], int(cfg.num_timesteps))

    dcfg = cfg.get("deployment", {})
    # Interference-disk radius is a multiple of the deployment config's max TX-RX
    # pairing distance (default 2x = 200 m): two TXs within 2x the max link length can
    # have near-colliding RXs, i.e. strong mutual interference. Falls back to an explicit
    # interference_radius_m if the max separation isn't available in the config.
    max_sep = float(sysp.get("max_tx_rx_distance", 0.0))
    radius_mult = float(dcfg.get("interference_radius_multiplier", 2.0))
    interf_radius = (radius_mult * max_sep if max_sep > 0
                     else float(dcfg.get("interference_radius_m", 0.0)))
    disk_alpha = float(dcfg.get("interference_disk_alpha", 0.10))

    meta = {
        "experiment": str(cfg.get("experiment_name", "sophisticated-oarfish-9")),
        "density": density, "network_id": network_id, "seed": int(channel.seed),
        "n_links": int(channel.n_links), "n_interf_edges": int(interf.shape[1]),
        "top_k": int(gcfg.top_k), "P_max": sysp["P_max"], "noise_var": sysp["noise_var"],
        "r_min": r_min, "interf_radius": interf_radius, "disk_alpha": disk_alpha,
        "interf_radius_mult": radius_mult, "max_tx_rx_distance": max_sep, **ainfo,
    }

    # ----- Output dir ---------------------------------------------------------
    out_dir = root / str(cfg.output.base_dir) / f"{density}_net{network_id}_r{r_min}"
    out_dir.mkdir(parents=True, exist_ok=True)

    span = float(np.ptp(rx, axis=0).mean()) * 0.01  # ring radius ~1% of extent

    # ----- Shared zoom crop window (N nearest links to the focus) -------------
    zoom_idx, center = _select_zoom(rx, graph_data, OmegaConf.to_object(cfg.zoom.focus)
                                    if OmegaConf.is_config(cfg.zoom.focus) else cfg.zoom.focus,
                                    int(cfg.zoom.n_links))
    zsel = rx[zoom_idx]
    xmin, ymin = zsel.min(axis=0)
    xmax, ymax = zsel.max(axis=0)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    # Square crop centred on the selected-node bounding box, so the dashed box and
    # every zoom panel stay square instead of being stretched to the cluster's aspect
    # ratio (which in dense scenarios is an arbitrary thin rectangle). Half-side = half
    # the larger bbox dimension + ~12% margin + the self-loop ring radius.
    half = 0.5 * float(max(xmax - xmin, ymax - ymin)) * 1.12 + span
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    # Clamp the square window into the node bounding box so it never crops into empty
    # space past a graph edge/corner (shift inward; centre if wider than the network).
    def _clamp(a0, a1, lo, hi):
        side = a1 - a0
        if side >= hi - lo:
            mid = 0.5 * (lo + hi)
            return mid - side / 2.0, mid + side / 2.0
        if a0 < lo:
            return lo, lo + side
        if a1 > hi:
            return hi - side, hi
        return a0, a1
    nlo, nhi = rx.min(axis=0), rx.max(axis=0)
    x0, x1 = _clamp(x0, x1, float(nlo[0]), float(nhi[0]))
    y0, y1 = _clamp(y0, y1, float(nlo[1]), float(nhi[1]))
    crop_bbox = (x0, x1, y0, y1)               # for the dashed overlay (square)
    crop_view = ((x0, x1), (y0, y1))           # axis limits for both zoom modes
    in_box = (rx[:, 0] >= x0) & (rx[:, 0] <= x1) & (rx[:, 1] >= y0) & (rx[:, 1] <= y1)
    sub_mask = np.zeros(len(rx), bool); sub_mask[zoom_idx] = True
    # Master switch: show_node_labels=false strips all node indices everywhere.
    label_on = bool(cfg.get("show_node_labels", True)) and bool(cfg.zoom.label_nodes)
    mode = str(cfg.zoom.mode)

    # ----- Full figure (marks the crop window) --------------------------------
    fig_full = _build_figure(channel, graph_data, graph, rx, gains, power_rx, rates, meta,
                             draw_mask=None, label_idx=None, view_bbox=None, directed=False,
                             mark_bbox=crop_bbox if cfg.zoom.enabled else None,
                             span=span, scope="full",
                             node_size=NODE_SIZE_FULL, dep_marker=DEP_MARKER_FULL)
    _save(fig_full, out_dir / "network_panels_full", root)

    # ----- Zoom figures -------------------------------------------------------
    if cfg.zoom.enabled:
        if mode not in ("subgraph", "camera", "both"):
            log.warning("Unknown zoom.mode=%r; expected subgraph|camera|both. Skipping zoom.", mode)
        if mode in ("subgraph", "both"):
            # Induced subgraph: only the selected nodes + edges with BOTH ends selected.
            fig = _build_figure(channel, graph_data, graph, rx, gains, power_rx, rates, meta,
                                draw_mask=sub_mask, label_idx=(zoom_idx if label_on else None),
                                view_bbox=crop_view, directed=True, mark_bbox=None, span=span,
                                scope=f"zoom · subgraph ({len(zoom_idx)} links, intra-subset edges)",
                                node_size=NODE_SIZE, dep_marker=DEP_MARKER)
            _save(fig, out_dir / "network_panels_zoom_subgraph", root)
            log.info("  subgraph: %d nodes (edges restricted to the subset)", len(zoom_idx))
        if mode in ("camera", "both"):
            # Camera magnify: render the full network, crop the viewport to the window.
            cam_labels = np.where(in_box)[0] if label_on else None
            fig = _build_figure(channel, graph_data, graph, rx, gains, power_rx, rates, meta,
                                draw_mask=None, label_idx=cam_labels, view_bbox=crop_view,
                                directed=False, mark_bbox=None, span=span,
                                scope=f"zoom · camera ({int(in_box.sum())} nodes in view, full network)",
                                node_size=NODE_SIZE, dep_marker=DEP_MARKER)
            _save(fig, out_dir / "network_panels_zoom_camera", root)
            log.info("  camera: %d nodes in view (cross-boundary edges trail off-panel)", int(in_box.sum()))

    # ----- Realizations vs stochastic time-sharing figure ---------------------
    # Panels (iv)/(v) show ONE allocation held constant. This companion figure shows
    # several allocation realizations + their constant-policy ergodic rates, and the
    # ergodic rate of the stochastic time-sharing policy (mean over all collected
    # allocations of their individual ergodic rates).
    rcfg = cfg.get("realizations", {})
    ts_feasible = None
    if bool(rcfg.get("enabled", True)):
        p_tx_all, p_rx_all, _steps_all, _ = _load_power_samples(pd_dir, network_id, int(channel.seed))
        if p_rx_all is None:
            p_rx_all = p_tx_all[:, tx_assoc]               # derive (K,n) per-RX power
        T_pol = int(rcfg.get("num_timesteps", 100))
        Ht = _sample_channel(channel, T_pol)               # shared realization (T,m,n)
        rates_all = _per_sample_ergodic_rates(p_tx_all, Ht, assoc, sysp["noise_var"])  # (K,n)
        rate_timeshare = rates_all.mean(axis=0)            # stochastic time-sharing rate
        ts_feasible = float(np.mean(rate_timeshare >= r_min))
        # N realizations with the most diverse feasibility patterns (each satisfies a
        # different RX subset — the case for time-sharing), shown worst->best feasibility.
        n_show = int(rcfg.get("n_show", 3))
        feas = rates_all >= r_min
        pick = _diverse_indices(feas, n_show)
        pick = pick[np.argsort(feas[pick].mean(axis=1))]
        p_show = [p_rx_all[k] for k in pick]
        r_show = [rates_all[k] for k in pick]
        lbls = _timestep_labels(len(pick))
        p_mean = p_rx_all.mean(axis=0)
        fig_r = _build_realizations_figure(
            rx, graph, p_show, r_show, lbls, p_mean, rate_timeshare, meta, NODE_SIZE_REAL)
        _save(fig_r, out_dir / "power_allocation_realizations", root)
        log.info("  realizations: %d shown; time-sharing feasible=%.1f%% "
                 "(per-sample feasible range %.1f-%.1f%%)", len(pick), 100 * ts_feasible,
                 100 * (rates_all >= r_min).mean(axis=1).min(),
                 100 * (rates_all >= r_min).mean(axis=1).max())
        # Camera-zoom variant: full network drawn, cropped to the same window as
        # network_panels_zoom_camera (so the two figures are directly comparable).
        if cfg.zoom.enabled and mode in ("camera", "both"):
            fig_rz = _build_realizations_figure(
                rx, graph, p_show, r_show, lbls, p_mean, rate_timeshare, meta,
                NODE_SIZE_REAL_ZOOM, view_bbox=crop_view)
            _save(fig_rz, out_dir / "power_allocation_realizations_zoom_camera", root)
            log.info("  realizations (camera zoom): %d RXs in view", int(in_box.sum()))

    # ----- Metadata sidecar ---------------------------------------------------
    meta_out = {**meta, "channel_cache": str(cache_path.relative_to(root)),
                "pd_dir": str(pd_dir.relative_to(root)),
                "zoom_mode": mode if cfg.zoom.enabled else None,
                "zoom_focus_xy": np.round(center, 2).tolist() if cfg.zoom.enabled else None,
                "zoom_n_links": int(cfg.zoom.n_links) if cfg.zoom.enabled else None,
                "zoom_subgraph_nodes": int(len(zoom_idx)) if cfg.zoom.enabled else None,
                "zoom_camera_nodes": int(in_box.sum()) if cfg.zoom.enabled else None,
                "zoom_link_indices": zoom_idx.tolist() if cfg.zoom.enabled else None,
                "rate_mean": float(rates.mean()), "rate_min": float(rates.min()),
                "rate_p5": float(np.percentile(rates, 5)),
                "frac_feasible": float(np.mean(rates >= r_min)),
                "timeshare_frac_feasible": ts_feasible}
    (out_dir / "metadata.json").write_text(json.dumps(meta_out, indent=2))
    log.info("✓ Saved metadata: %s", (out_dir / "metadata.json").relative_to(root))
    print(f"\nDone. Figures written to: {out_dir}")


if __name__ == "__main__":
    main()
