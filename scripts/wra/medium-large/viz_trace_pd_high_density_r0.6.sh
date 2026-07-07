#!/bin/bash

config_name=wra_medium-large_outdoor_high_density
collection_window=2000
top_pairs=2       # scatter pairs from top quantile group
mid_pairs=1       # scatter pairs from mid quantile group
bot_pairs=1       # scatter pairs from bottom quantile group

# Diffusion overlay (merged net0+net1 NPZ for r0.6 high-density)
OVERLAY_PATH="${OVERLAY_PATH:-outputs/wireless_resource_allocation-wra/comparison/2026-04-04/20-54-50/overlay/generated_powers_medium-large_outdoor_high_density_ha6c7c432ee13_merged.npz}"
top_k=$((2 * $top_pairs))
mid_k=$((2 * $mid_pairs))
bot_k=$((2 * $bot_pairs))
n_seeds=100          # number of figures to generate per r_min (one per seed)
base_seed=42

export HYDRA_FULL_ERROR=1

# All r_min configurations: "run_id|run_timestamp"
# Comment out lines with # to skip them.
while IFS='|' read -r run_id run_timestamp; do
    [[ "$run_id" =~ ^#.*$ || -z "$run_id" ]] && continue
    input_dir="outputs/${config_name}/${run_id}/${run_timestamp}"
    output_dir="${input_dir}/trace-viz"
    rmin=$(echo "$run_id" | grep -oP 'r\d+\.\d+')
    echo "====== ${config_name} | ${rmin} ======"

    for i in $(seq 0 $((n_seeds - 1))); do
        seed=$((base_seed + i))
        echo "=== seed=${seed} (${i}/${n_seeds}) ==="
        python -m graph_signal_diffusion.cli.wra.visualize_trace \
            --config-name=pd_trace_visualization/${config_name} \
            ++input_dir=${input_dir} \
            ++collection_window=${collection_window} \
            ++panels.scatter.transient_display=both ++panels.scatter.max_collection_points=200 ++panels.scatter.max_transient_points=200 ++panels.scatter.transient_scope=early ++panels.scatter.axis_range=unit \
            ++panels.dual.xscale=compressed ++panels.slack.xscale=compressed \
            ++panels.dual.ymin=0.0 ++panels.dual.ymax=50.0 \
            ++panels.slack.convention=max ++panels.slack.worst_percentile=1 \
            ++output.dir=${output_dir} ++output.format=pdf \
            ++visualization_mode=row \
            ++receiver_selection.auto_strategy=neg_corr_pair \
            ++receiver_selection.seed=${seed} \
            ++receiver_selection.neg_corr_pair.quantile_source=["bottom_rate_mean","dual_mean"] ++receiver_selection.neg_corr_pair.variance_quantile=0.05 \
            '++receiver_selection.groups=[{quantile:[0.96,1.00],count:'"${top_k}",'dual_count:'"${top_k}"',palette:["#dc143c","#8b5cf6"],scatter_pairs:'"${top_pairs}"'},{quantile:[0.80,0.98],count:'"${mid_k}"',dual_count:0,palette:["#10b981","#f59e0b"],scatter_pairs:'"${mid_pairs}"'},{quantile:[0.0,0.80],count:'"${bot_k}"',dual_count:0,palette:["#3b82f6","#14b8a6"],scatter_pairs:'"${bot_pairs}"'}]' \
            ++overlay.generated_powers_path="${OVERLAY_PATH}" \
            ++overlay.marker=x '++overlay.color="#ff6600"' ++overlay.alpha=0.4 ++overlay.marker_size=50 ++overlay.label=Diffusion \
            ++panels.scatter.legend=off \
            ++panels.dual.xlabel=Iteration ++panels.slack.xlabel=Iteration \
            ++panels.slack.ylabel=explicit '++panels.slack.ylabel_text=Constraint Slack' '++panels.slack.ylabel_color="#1f5faa"' \
            ++panels.dual.title=off ++panels.slack.title=off \
            '++panels.slack.infeasibility_label="Per-iteration infeasible %"' \
            '++panels.slack.worst_avg_label="Iteration-averaged worst slack"' \
            ++panels.slack.infeasibility_ylabel=explicit '++panels.slack.infeasibility_ylabel_text="Constraint violations (%)"' \
            ++panels.slack.feasible_alpha=0.10 ++panels.slack.infeasible_alpha=0.10 \
            ++panels.slack.cw_shade_alpha=0.20 \
            ++panels.slack.target_nticks=6 \

    done
done <<'RUNS'
# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.4_a0.2_haeeb262dc512|2026-03-17/00-34-17
# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.5_a0.2_haf3bf83302a0|2026-03-17/12-07-32
wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.6_a0.2_h9361961eadf0|2026-03-16/16-41-22
# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.7_a0.2_h4f814ddbf57b|2026-03-17/22-59-38
# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.8_a0.2_h310269b3a34c|2026-03-18/16-09-37
RUNS

####### Comments for refining the select run's visualizations #######
# For r0.6: 21K convergence, structured points but large multipliers. Trace-viz quality is poor/ok.
# For r0.7: 41K convergence, large dual multipliers and concentrated points. Trace-viz quality is poor.
# For r0.8: 60K convergence, large dual multipliers and concentrated points. Trace-viz quality is poor.

# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.6_a0.2_h9361961eadf0
    # /2026-03-16/16-41-22/trace-viz/trace_row_network_0_20260323_125938.pdf
    # -> Good cyclostability of duals, good slack convergence panel.

# wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.7_a0.2_h4f814ddbf57b
    # /2026-03-17/22-59-38/trace-viz/trace_row_network_0_20260323_130526.pdf
    # -> Good cyclostability of duals, good slack convergence panel.

