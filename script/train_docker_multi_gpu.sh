#!/bin/bash

# ============================================================
# Mario RL Training Script - MULTI-GPU Configuration
# ============================================================
# Distributed training across multiple GPUs
# Requires 2+ CUDA GPUs

MODEL_NAME="ImpalaDueling"
EPISODES=5000
MAX_STEPS=5000

# Optimized N-Step for distributed training
N_STEP=3

# Frame Processing
FRAME_STACK=4
FRAME_SKIP=4

# Scale batch size with number of GPUs
# Each GPU will process (BATCH_SIZE / WORLD_SIZE) samples
BATCH_SIZE=256  # 128 per GPU if WORLD_SIZE=2
BUFFER_SIZE=100000
LEARNING_RATE=0.0002  # Higher LR for larger effective batch

# Epsilon
EPSILON_DECAY=0.995

# Optimizer
OPTIMIZER="adamw"
WEIGHT_DECAY=0.0001
SCHEDULER="exponential"
SCHEDULER_GAMMA=0.999

# Environment
MOVESET="balanced"

# Testing & Saving
TEST_EVERY=50
SAVE_EVERY=50

# Distributed Training Configuration
DISTRIBUTED="--distributed"
WORLD_SIZE=2  # Number of GPUs (change to match your setup)

# W&B
WANDB_ENABLED="--wandb"
WANDB_PROJECT="mario-rl-distributed"

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
    ${DISTRIBUTED} \
    --world-size ${WORLD_SIZE} \
    ${WANDB_ENABLED} \
    --wandb-project ${WANDB_PROJECT}"

echo "============================================================"
echo "🌐 MULTI-GPU DISTRIBUTED Training Mode"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${WORLD_SIZE}"
echo "Episodes: ${EPISODES}"
echo "N-Step: ${N_STEP}"
echo "Batch Size: ${BATCH_SIZE} (${BATCH_SIZE}/${WORLD_SIZE} per GPU)"
echo "Learning Rate: ${LEARNING_RATE}"
echo "============================================================"
echo ""

# Check GPU availability
echo "Checking available GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

sudo docker run -it --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"
