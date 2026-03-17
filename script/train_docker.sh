#!/bin/bash

# ============================================================
# Mario RL Training Script - Optimized Configuration
# ============================================================
# This script launches training with optimized hyperparameters
# including n-step returns for faster convergence

# Model Architecture
MODEL_NAME="ImpalaDueling"  # Best model: Dueling + IMPALA CNN
# Alternatives: "DQN", "ResNETv1", "ImpalaDQN", "ImpalaNoisy", "ImpalaFull"

# Training Episodes
EPISODES=5000
MAX_STEPS=5000  # Max steps per episode

# N-Step Returns (NEW - speeds up convergence by 20-35%)
N_STEP=3  # Recommended: 3-5 for Mario
# N_STEP=1: Disable n-step (1-step TD learning)
# N_STEP=3: Best balance of bias/variance (RECOMMENDED)
# N_STEP=5: Longer credit assignment for complex zones

# Frame Processing
FRAME_STACK=4    # Temporal information (number of frames stacked)
FRAME_SKIP=4     # Action repeat for faster training

# Replay Buffer & Training
BATCH_SIZE=128        # Larger batch = more stable gradients
BUFFER_SIZE=50000     # Large replay buffer for diverse experience
LEARNING_RATE=0.0001  # Default 1e-4 works well with AdamW

# Exploration (Epsilon-Greedy)
EPSILON_DECAY=0.995   # Per-episode decay (0.995 = moderate exploration)
# Higher (0.999) = slower decay, more exploration
# Lower (0.99) = faster decay, more exploitation

# Optimizer Configuration
OPTIMIZER="adamw"        # AdamW with weight decay (better than Adam)
WEIGHT_DECAY=0.0001      # L2 regularization (1e-4)
SCHEDULER="exponential"  # Learning rate scheduler
SCHEDULER_GAMMA=0.999    # LR decay factor per episode

# Environment Configuration
MOVESET="balanced"  # Action space complexity
# Options: "minimal", "balanced" (default), "reckless", "speedrun", "complete"

# Testing & Saving
TEST_EVERY=50    # Test model every N episodes
SAVE_EVERY=50    # Save checkpoint every N episodes

# Weights & Biases Logging
WANDB_ENABLED="--wandb"  # Comment out to disable W&B
WANDB_PROJECT="mario-rl"
# WANDB_ENTITY="your-team"  # Uncomment and set your W&B team

# ============================================================
# Construct Training Command
# ============================================================
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

# Uncomment for distributed training (multi-GPU)
# COMMAND="${COMMAND} --distributed --world-size 2"

echo "============================================================"
echo "🎮 Starting Mario RL Training"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Episodes: ${EPISODES}"
echo "N-Step Returns: ${N_STEP}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Optimizer: ${OPTIMIZER} (weight_decay=${WEIGHT_DECAY})"
echo "Moveset: ${MOVESET}"
echo "============================================================"
echo ""

# Launch Docker container with GPU support
sudo docker run -it --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"
