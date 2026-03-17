#!/bin/bash

# ============================================================
# Mario RL Training Script - STABLE/SAFE Configuration
# ============================================================
# Conservative settings for stable, reliable training
# Use this for production runs and benchmark comparisons

MODEL_NAME="ImpalaDueling"
EPISODES=7000          # More episodes for thorough learning
MAX_STEPS=6000         # Longer episodes for exploration

# Conservative N-Step
N_STEP=3  # Balanced bias/variance

# Frame Processing
FRAME_STACK=4
FRAME_SKIP=4

# Conservative batch size
BATCH_SIZE=64
BUFFER_SIZE=75000
LEARNING_RATE=0.00005  # Lower LR for stability

# Slow epsilon decay for more exploration
EPSILON_DECAY=0.997

# Optimizer with stronger regularization
OPTIMIZER="adamw"
WEIGHT_DECAY=0.0002    # Higher weight decay for regularization
SCHEDULER="plateau"    # Adaptive LR based on performance
SCHEDULER_GAMMA=0.999

# Environment
MOVESET="balanced"

# Testing & Saving
TEST_EVERY=100
SAVE_EVERY=100

# W&B
WANDB_ENABLED="--wandb"
WANDB_PROJECT="mario-rl-stable"

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
echo "🛡️  STABLE/CONSERVATIVE Training Mode"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Episodes: ${EPISODES} (extended training)"
echo "N-Step: ${N_STEP} (balanced)"
echo "Batch Size: ${BATCH_SIZE} (conservative)"
echo "Learning Rate: ${LEARNING_RATE} (low, stable)"
echo "Scheduler: ${SCHEDULER} (adaptive)"
echo "============================================================"
echo ""

sudo docker run -it --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"
