#!/bin/bash
# watcher_s5.sh — Monitor s5_explicit_pd_16k training and auto-launch s6
#
# Runs via nohup on RTX6000. Survives local computer shutdown.
# Log: /tmp/watcher_s5.log
#
# Lifecycle:
#   1. Monitor PID 678911 (s5 torchrun parent)
#   2. When PID dies → wait 60s for TensorBoard flush
#   3. Run train_monitor.py to find best checkpoint
#   4. Create training plan for s6 (rough terrain)
#   5. Launch orchestrator for s6 via nohup
#   6. Orchestrator monitors s6, handles overfitting/crash, saves best

set -euo pipefail

# --- Configuration ---
PID=678911
RUN_DIR="/home/phh/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/2026-05-05_17-27-25_s5_explicit_pd_16k"
PROJECT_ROOT="/home/phh/magiclab_rl_lab"
PLAN_PATH="$PROJECT_ROOT/training_plans/z1_s5_s6_plan.yaml"
LOG_FILE="/tmp/watcher_s5.log"
NUM_GPUS=4
NUM_ENVS=16384
MAX_ITERS=50000

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# --- Phase 1: Wait for PID to exit ---
log "=== Watcher started ==="
log "Monitoring PID $PID (s5_explicit_pd_16k)"
log "Run dir: $RUN_DIR"
log "Polling every 120s"

while kill -0 "$PID" 2>/dev/null; do
    LATEST=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
    LATEST_NAME=$(basename "${LATEST:-none}")
    log "PID $PID alive | latest: $LATEST_NAME"
    sleep 120
done

log "=== PID $PID exited! ==="
log "Waiting 60s for TensorBoard and checkpoints to flush..."
sleep 60

# --- Phase 2: Find best checkpoint ---
cd "$PROJECT_ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

log "Running train_monitor.py to find best checkpoint..."
python -u scripts/train_monitor.py --once --terrain gentle --run_dir "$RUN_DIR" > /tmp/watcher_monitor_output.txt 2>&1 || true

# Try to extract best model iteration from monitor output
BEST_ITER=""
if [ -f /tmp/watcher_monitor_output.txt ]; then
    # Look for "best_model_iter" or "Best model" in output
    BEST_ITER=$(grep -aPoi 'm\d+' /tmp/watcher_monitor_output.txt | head -1 | grep -Po '\d+' || true)
fi

if [ -n "$BEST_ITER" ] && [ -f "$RUN_DIR/model_${BEST_ITER}.pt" ]; then
    BEST_CKPT="$RUN_DIR/model_${BEST_ITER}.pt"
    log "Best checkpoint (monitor): model_${BEST_ITER}.pt"
else
    # Fallback: latest checkpoint (highest iteration number)
    BEST_CKPT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        log "ERROR: No checkpoints found in $RUN_DIR! Exiting."
        exit 1
    fi
    log "Best checkpoint (fallback=latest): $(basename "$BEST_CKPT")"
fi

# --- Phase 3: Create training plan ---
mkdir -p "$(dirname "$PLAN_PATH")"
cat > "$PLAN_PATH" << YAML
# Auto-generated training plan — s5 → s6
# Created: $(date '+%Y-%m-%d %H:%M:%S')
# s5 best checkpoint: $BEST_CKPT

stages:
  - id: s6_rough_resume
    terrain: rough
    env_config: source/magiclab_rl_lab/magiclab_rl_lab/tasks/locomotion/robots/z1/12dof/velocity_env_cfg_s4_full_terrain.py
    max_iterations: $MAX_ITERS
    num_envs: $NUM_ENVS
    initial_checkpoint: $BEST_CKPT
    monitor:
      action_rate_threshold: -1.5
      min_iterations: 2000

retry_policy:
  max_retries: 2
  nan:
    learning_rate_multiplier: 0.5
  oom:
    num_envs_divisor: 2
YAML

log "Created plan: $PLAN_PATH"
log "  s6 initial_checkpoint: $(basename "$BEST_CKPT")"

# --- Phase 4: Launch orchestrator ---
log "Launching orchestrator for s6_rough_resume..."
nohup python -u -m automation.orchestrator \
    --plan "$PLAN_PATH" \
    --project-root "$PROJECT_ROOT" \
    --device cuda:0 \
    --num-gpus "$NUM_GPUS" \
    --poll-interval 120 \
    >> "$LOG_FILE" 2>&1 &

ORCH_PID=$!
log "=== Orchestrator launched (PID $ORCH_PID) ==="
log "Orchestrator log: $LOG_FILE"
log "Also: tail -f $PROJECT_ROOT/logs/orchestrator.log"
log "Watcher exiting. Goodbye!"
