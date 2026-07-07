#!/bin/bash

config_name=wra_medium-large_outdoor_ultra-low_density
collection_window=2000
top_pairs=2       # scatter pairs from top quantile group
mid_pairs=1       # scatter pairs from mid quantile group
bot_pairs=1       # scatter pairs from bottom quantile group
top_k=$((2 * $top_pairs))
mid_k=$((2 * $mid_pairs))
bot_k=$((2 * $bot_pairs))
n_seeds=5          # number of figures to generate per r_min (one per seed)
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
            ++panels.slack.convention=max ++panels.slack.worst_percentile=0.5 \
            ++output.dir=${output_dir} ++output.format=pdf \
            ++visualization_mode=row \
            ++receiver_selection.auto_strategy=neg_corr_pair \
            ++receiver_selection.seed=${seed} \
            ++receiver_selection.neg_corr_pair.quantile_source=["bottom_rate_mean","dual_mean"] ++receiver_selection.neg_corr_pair.variance_quantile=0.05 \
            '++receiver_selection.groups=[{quantile:[0.95,1.00],count:'"${top_k}"',palette:["#dc143c","#8b5cf6"],scatter_pairs:'"${top_pairs}"'},{quantile:[0.75,0.95],count:'"${mid_k}"',palette:["#f59e0b","#10b981"],scatter_pairs:'"${mid_pairs}"'},{quantile:[0.0,0.75],count:'"${bot_k}"',palette:["#3b82f6","#14b8a6"],scatter_pairs:'"${bot_pairs}"'}]' \

    done
done <<'RUNS'
# wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.4_a0.2_hee178290f551|2026-03-15/15-34-39
# wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.5_a0.2_h7afcdff41256|2026-03-15/18-52-45
# wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.6_a0.2_ha8f90c3f236f|2026-03-15/11-54-13
wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.7_a0.2_hd71a1624c374|2026-03-15/22-16-54
wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.8_a0.2_h99b12fea8846|2026-03-16/05-37-41
RUNS



####### Comments for refining the select run's visualizations #######
# For r0.4: Early convergence, too scattered points. Trace-viz quality is poor.
# For r0.5: Early convergence, too scattered points. Trace-viz quality is poor.
# For r0.6: Early convergence, too scattered points. Trace-viz quality is poor.
# For r0.7: 20K convergence, structured points. Trace-viz quality is very good.
# For r0.8: 20K convergence, structured points. Trace-viz quality is very good.


# wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.8_a0.2_h99b12fea8846
    # /2026-03-16/05-37-41/trace-viz/trace_row_network_1_20260323_131029.pdf
    # -> Good example of late convergence of dual multipliers.