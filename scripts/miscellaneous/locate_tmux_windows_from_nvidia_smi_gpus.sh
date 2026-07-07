#!/usr/bin/env bash
# Run this script inside a tmux session so that tmux list-panes works. 
# It will locate which tmux windows correspond to the GPU compute processes shown in nvidia-smi. 
# It cross-references the TTY of the processes with the TTY of tmux panes to find matches.
set -euo pipefail

normalize_tty() {
  local t="${1:-}"
  t="${t#/dev/}"       # /dev/pts/7 -> pts/7
  t="${t#tty}"         # tty7 -> 7 (keeps pts/7 unchanged)
  echo "$t"
}

# Map GPU UUID -> index (used to label GPUs)
declare -A uuid_to_index
while IFS=, read -r idx uuid; do
  idx="$(echo "$idx" | xargs)"
  uuid="$(echo "$uuid" | xargs)"
  if [ -n "$idx" ] && [ -n "$uuid" ]; then
    uuid_to_index["$uuid"]="$idx"
  fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)

# Collect compute PIDs with GPU UUID when available
declare -A pid_to_gpu
mapfile -t pid_gpu_lines < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk 'NF')

if [ ${#pid_gpu_lines[@]} -eq 0 ]; then
  echo "No GPU compute processes found."
  exit 0
fi

pids=()
for line in "${pid_gpu_lines[@]}"; do
  pid="$(echo "$line" | awk -F, '{print $1}' | xargs)"
  gpu_uuid="$(echo "$line" | awk -F, '{print $2}' | xargs)"
  if [ -n "$pid" ]; then
    pids+=("$pid")
    if [ -n "$gpu_uuid" ] && [ -n "${uuid_to_index["$gpu_uuid"]:-}" ]; then
      pid_to_gpu["$pid"]="${uuid_to_index["$gpu_uuid"]}"
    elif [ -n "$gpu_uuid" ]; then
      pid_to_gpu["$pid"]="$gpu_uuid"
    else
      pid_to_gpu["$pid"]="?"
    fi
  fi
done

# Build a map of TTY -> tmux pane label (across all tmux servers for this user)
declare -A tty_to_pane
uid="$(id -u)"
sock_dir="/tmp/tmux-$uid"
mapfile -t sockets < <(find "$sock_dir" -maxdepth 1 -type s 2>/dev/null || true)

if [ ${#sockets[@]} -eq 0 ]; then
  sockets=("default")
fi

for sock in "${sockets[@]}"; do
  server_label="$(basename "$sock")"
  while read -r line; do
    pane_label=$(awk '{print $1}' <<<"$line")
    pane_tty=$(awk '{print $3}' <<<"$line")
    norm_tty="$(normalize_tty "$pane_tty")"
    if [ -n "$norm_tty" ] && [ "$norm_tty" != "?" ]; then
      tty_to_pane["$norm_tty"]="$server_label:$pane_label"
    fi
  done < <(tmux -S "$sock" list-panes -a -F "#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_tty}" 2>/dev/null || true)
done

printf "%-8s %-6s %-10s %-30s %s\n" PID GPU TTY PROC TMUX_PANE
for pid in "${pids[@]}"; do
  if ps_out=$(ps -o pid=,tty=,comm= -p "$pid" 2>/dev/null); then
    pid_f=$(awk '{print $1}' <<<"$ps_out")
    tty_f=$(awk '{print $2}' <<<"$ps_out")
    comm_f=$(awk '{print $3}' <<<"$ps_out")
    norm_tty="$(normalize_tty "$tty_f")"
    pane=${tty_to_pane["$norm_tty"]:-"-"}
    gpu="${pid_to_gpu["$pid_f"]:-?}"
    printf "%-8s %-6s %-10s %-30s %s\n" "$pid_f" "$gpu" "$tty_f" "$comm_f" "$pane"
  fi
done
