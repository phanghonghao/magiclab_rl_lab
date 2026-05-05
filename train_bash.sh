#!/bin/bash

# switch to script directory
cd "$(dirname "$0")"

# choose python interpreter
PYTHON="/home/ubuntu/miniconda3/envs/mmckit/bin/python"

# script path
TRAIN_SCRIPT="scripts/rsl_rl/train.py"

# === Smoke Test (10 iterations) ===
# Uncomment the following lines for a quick smoke test:
# $PYTHON $TRAIN_SCRIPT \
#     --task=Magiclab-Z1-12dof-Velocity \
#     --run_name=z1_smoke_test \
#     --headless \
#     --max_iterations=10 \
#     --num_envs=64 \
#     --device=cuda:0

# === Formal Training ===
# Use nohup for background training, logs saved to train_z1.log
nohup $PYTHON $TRAIN_SCRIPT \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=z1_locomotion_s1 \
    --headless \
    --max_iterations=50000 \
    --num_envs=8192 \
    --device=cuda:0 \
    > train_z1.log 2>&1 &

echo "Training started in background. PID: $!"
echo "Monitor with: tail -f train_z1.log"
echo "TensorBoard:  tensorboard --logdir logs/rsl_rl/ --port 6006 --bind_all"

# === Resume Training (uncomment to resume from checkpoint) ===
# $PYTHON $TRAIN_SCRIPT \
#     --task=Magiclab-Z1-12dof-Velocity \
#     --run_name=z1_locomotion_s1_resume \
#     --headless \
#     --max_iterations=50000 \
#     --num_envs=8192 \
#     --device=cuda:0 \
#     --resume \
#     --load_run=<timestamp>_z1_locomotion_v1 \
#     --checkpoint=model_<N>.pt