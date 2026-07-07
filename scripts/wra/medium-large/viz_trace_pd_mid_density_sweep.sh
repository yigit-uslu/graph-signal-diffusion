#!/bin/bash

config_name=wra_medium-large_outdoor_mid_density
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
wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.4_a0.2_h278d2d05ae5c|2026-03-15/18-54-57
wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.5_a0.2_h694589f8a766|2026-03-16/00-07-27
wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.6_a0.2_hf183faf6339f|2026-03-14/19-04-11
# wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.7_a0.2_h1cbee1e461e9|2026-03-16/07-33-42
# wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.8_a0.2_h4a47f7f0dbca|2026-03-16/23-50-48
RUNS


####### Comments for refining the select run's visualizations #######
# For r0.4: 20K convergence, structured points. Trace-viz quality is good.
# For r0.5: 20K convergence, structured points. Trace-viz quality is good.
# For r0.6: 20K convergence, structured points. Trace-viz quality is good.
# For r0.7: 40K convergence, large dual multipliers and concentrated points. Trace-viz quality is poor.
# For r0.8: 40K convergence, large dual multipliers and concentrated points. Trace-viz quality is poor.



