#!/usr/bin/env bash
#
# Resumable batch runner for the VSI demos/experiments.
#
# Runs each (domain, scenario) job once, writing a per-job log and a
# ".done" sentinel on success.  Re-running SKIPS completed jobs, so an
# interrupted run resumes where it left off instead of starting over.
#
# Jobs are grouped into batches by domain (the two MILP domains first,
# then the cheaper LP/QP singleton domains). A single failing job does
# not abort the queue; it is reported at the end and retried on the next
# run.
#
# Usage:
#   ./scripts/run_experiments.sh                 # run all pending jobs
#   ./scripts/run_experiments.sh scmdp           # run only one batch (domain)
#   ./scripts/run_experiments.sh markovgame_singleton
#   FORCE=1 ./scripts/run_experiments.sh         # ignore sentinels, re-run all
#
# Resume a single job: delete its sentinel, e.g.
#   rm results/_runlogs/scmdp__scaling_states.done
#
# Run it under tmux (or nohup) so it survives SSH disconnects:
#   tmux new -s vsi
#   source .venv/bin/activate         # the server's virtualenv
#   ./scripts/run_experiments.sh
#   # detach with Ctrl-b d ; reattach with: tmux attach -t vsi
#
set -u  # NOT -e: one failing job must not kill the whole queue

PY=${PY:-python3}
LOG_DIR=${LOG_DIR:-results/_runlogs}
mkdir -p "$LOG_DIR"

# --- Job catalogue, grouped into batches by domain -------------------------
JOBS=(
  # batch 1: scmdp (full value-set, MILP — heaviest)
  "scmdp single_run"
  "scmdp bound_sweep"
  "scmdp scaling_states"
  "scmdp scaling_actions"
  # batch 2: markovgame (full value-set, MILP)
  "markovgame single_run"
  "markovgame scaling_states"
  "markovgame scaling_actions"
  # batch 3: scmdp_singleton (LP/QP, cheaper)
  "scmdp_singleton single_run"
  "scmdp_singleton scaling_states"
  "scmdp_singleton scaling_actions"
  "scmdp_singleton scaling_reward_dim"
  # batch 4: markovgame_singleton (LP/QP, cheaper)
  "markovgame_singleton single_run"
  "markovgame_singleton scaling_states"
  "markovgame_singleton scaling_actions"
  "markovgame_singleton scaling_num_agents"
)

filter=${1:-}

run_one() {
  local domain=$1 scenario=$2
  local tag="${domain}__${scenario}"
  local log="${LOG_DIR}/${tag}.log"
  local done="${LOG_DIR}/${tag}.done"

  if [[ -z "${FORCE:-}" && -f "$done" ]]; then
    echo "[skip] $tag (already done)"
    return 0
  fi

  # A forced or interrupted rerun is incomplete until it succeeds again.
  rm -f "$done"
  echo "[run ] $tag  ->  $log"
  local start; start=$(date +%s)
  if PYTHONPATH=. "$PY" main.py "$domain" "$scenario" >"$log" 2>&1; then
    local secs=$(( $(date +%s) - start ))
    date -u +"%Y-%m-%dT%H:%M:%SZ ${secs}s ok" > "$done"
    echo "[ok  ] $tag (${secs}s)"
  else
    echo "[FAIL] $tag — see $log" >&2
    return 1
  fi
}

fails=0
for job in "${JOBS[@]}"; do
  read -r domain scenario <<<"$job"
  [[ -n "$filter" && "$domain" != "$filter" ]] && continue
  run_one "$domain" "$scenario" || fails=$((fails + 1))
done

echo "----"
if (( fails )); then
  echo "Done with $fails failed job(s). Fix the cause and re-run to resume."
  exit 1
fi
echo "All selected jobs complete."
