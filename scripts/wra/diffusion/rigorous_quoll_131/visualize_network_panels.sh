#!/bin/bash
# Render the 5-panel "WRA network anatomy" figure (+ spatial-crop zoom + power-
# allocation realizations) for one network of the rigorous-quoll-131 experiment
# (ugnn_wra_v3_ds8_norm_act_head, all-density @ single r_min=0.6).
#
# Panels: (i) physical TX-RX deployment, (ii) channel-gain graph (interference
# edges + direct self-loops), (iii) PyG abstraction (node features + edge
# weights), (iv) PD-expert allocation as a graph signal, (v) ergodic rates as a
# graph signal.  Figures are written under the experiment dir:
#   outputs/.../ugnn_wra_v3_ds8_norm_act_head-.../rigorous-quoll-131/network_panels/<density>_net<id>_r<rmin>/
#
# These panels visualize the PD-EXPERT scenario data (not the diffusion model), so
# at a given (density, r_min) they match sophisticated-oarfish-9's — only the output
# location and experiment label differ. This arm is trained at a single r_min=0.6.
#
# Usage:
#   bash scripts/wra/diffusion/rigorous_quoll_131/visualize_network_panels.sh
#
# Examples (any cfg field is overridable):
#   # mid density, network 3, final sample, larger zoom:
#   bash scripts/wra/diffusion/rigorous_quoll_131/visualize_network_panels.sh \
#     density=mid network_id=3 sample_selection=last zoom.n_links=20
#
#   # focus the zoom on a specific receiver index, disable labels:
#   bash scripts/wra/diffusion/rigorous_quoll_131/visualize_network_panels.sh \
#     zoom.focus=17 zoom.label_nodes=false

set -euo pipefail
export HYDRA_FULL_ERROR=1

# Ensure the *env* libstdc++ is found (provides CXXABI_1.3.15 needed by matplotlib).
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/graph-signal-diffusion/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"
CONFIG_NAME="${CONFIG_NAME:-network_panels/rigorous_quoll_131}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -m graph_signal_diffusion.cli.wra.visualize_network_panels
  --config-name="${CONFIG_NAME}"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Rendering WRA network-anatomy panels (config: ${CONFIG_NAME})"
"${cmd[@]}"
