#!/bin/bash

# ============================================================
# Mario RL Training Script - FAST CONVERGENCE Configuration
# ============================================================
# Optimized for fastest convergence with aggressive n-step
# Use this for quick experiments and proof-of-concept

MODEL_NAME="ImpalaDueling"  # Best performance/speed ratio
EPISODES=3000               # Fewer episodes needed with n-step
MAX_STEPS=4000

# Aggressive N-Step for faster learning
N_STEP=5  # Higher n-step = faster credit assignment

# Frame Processing
FRAME_STACK=4
FRAME_SKIP=4

# Larger batches for stability
BATCH_SIZE=256
BUFFER_SIZE=100000  # Larger buffer for more diversity
LEARNING_RATE=0.00025  # Slightly higher LR for faster learning

# Faster epsilon decay
EPSILON_DECAY=0.99  # Decay faster to exploit learned policy

# Optimizer
OPTIMIZER="adamw"
WEIGHT_DECAY=0.00005  # Lower weight decay
SCHEDULER="cosine"    # Cosine annealing for smooth decay
SCHEDULER_GAMMA=0.999

# Environment
MOVESET="balanced"

# More frequent testing
TEST_EVERY=25
SAVE_EVERY=25

# W&B
WANDB_ENABLED="--wandb"
WANDB_PROJECT="mario-rl-fast"

COMMAND="python3 train.py \
    --model-name ${MODEL_NAME} \
    --episodes ${EPISODES} \
    --max-steps ${MAX_STEPS} \
    --n-step ${N_STEP} \
    --frame-stack ${FRAME_STACK} \
    --frame-skip ${FRAME_SKIP} \
    --batch-size ${BATCH_SIZE} \
    --buffer-size ${BUFFER_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --epsilon-decay ${EPSILON_DECAY} \
    --optimizer ${OPTIMIZER} \
    --weight-decay ${WEIGHT_DECAY} \
    --scheduler ${SCHEDULER} \
    --scheduler-gamma ${SCHEDULER_GAMMA} \
    --moveset ${MOVESET} \
    --test-every ${TEST_EVERY} \
    --save-every ${SAVE_EVERY} \
    ${WANDB_ENABLED} \
    --wandb-project ${WANDB_PROJECT}"

echo "============================================================"
echo "🚀 FAST CONVERGENCE Training Mode"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Episodes: ${EPISODES} (reduced for faster convergence)"
echo "N-Step: ${N_STEP} (aggressive credit assignment)"
echo "Batch Size: ${BATCH_SIZE} (large batches)"
echo "Learning Rate: ${LEARNING_RATE}"
echo "============================================================"
echo ""

sudo docker run -it --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"
