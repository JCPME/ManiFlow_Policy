#!/usr/bin/env bash
# kill_training.sh — terminate all ManiFlow training processes (and their
# dataloader/worker children + wandb), gracefully then forcefully.
#
# Usage:
#   bash kill_training.sh        # SIGTERM, wait 10s, then SIGKILL survivors (prompts)
#   bash kill_training.sh -y     # don't ask for confirmation
#   bash kill_training.sh -9     # skip graceful TERM, go straight to SIGKILL
#   bash kill_training.sh -9 -y  # nuke immediately, no prompt
#
set -uo pipefail

FORCE=0; YES=0
for a in "$@"; do case "$a" in
  -9|--force) FORCE=1 ;;
  -y|--yes)   YES=1 ;;
  -h|--help)  sed -n '2,11p' "$0"; exit 0 ;;
  *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

# What counts as "training": the python workspaces + the shell launchers.
PATTERN='train_maniflow.*workspace\.py|train_eval_[a-z]*\.sh'

SELF=$$
MYPGID=$(ps -o pgid= -p "$SELF" | tr -d ' ')

# PIDs whose full command line matches — excluding THIS script, its parent
# shell, and pgrep itself, so we never signal ourselves.
list_pids() {
  pgrep -af "$PATTERN" 2>/dev/null \
    | grep -vE 'kill_training|[ /]pgrep( |$)' \
    | awk -v s="$SELF" -v pp="$PPID" '$1!=s && $1!=pp {print $1}'
}

mapfile -t PIDS < <(list_pids)
if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No training processes running."
  exit 0
fi

CSV=$(IFS=,; echo "${PIDS[*]}")
echo "Found ${#PIDS[@]} training process(es):"
ps -o pid,pgid,etime,cmd -p "$CSV" 2>/dev/null | cut -c1-150

if [ "$YES" -ne 1 ] && [ -t 0 ]; then
  read -r -p $'\nKill these and their child workers? [y/N] ' ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# Process groups to signal (catches dataloader workers); never our own group.
mapfile -t PGIDS < <(ps -o pgid= -p "$CSV" 2>/dev/null | tr -d ' ' | sort -u | grep -vx "$MYPGID")

send() {  # send <SIGNAL>
  for g in "${PGIDS[@]:-}"; do [ -n "$g" ] && kill -s "$1" "-$g" 2>/dev/null; done
  for p in "${PIDS[@]}";     do kill -s "$1" "$p"  2>/dev/null; done
}
alive() { ps -o pid= -p "$CSV" 2>/dev/null | grep -c '[0-9]'; }

if [ "$FORCE" -ne 1 ]; then
  echo "Sending SIGTERM..."
  send TERM
  for _ in $(seq 10); do [ "$(alive)" -eq 0 ] && break; sleep 1; done
fi

if [ "$(alive)" -ne 0 ]; then
  echo "Survivors remain -> SIGKILL."
  send KILL
  sleep 1
fi

# Report.
if [ "$(list_pids | wc -l)" -ne 0 ]; then
  echo "WARNING — still alive:"
  list_pids
  exit 1
fi
echo "✓ All training processes terminated."
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU now:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
fi
