#!/bin/bash

# ============================================================
# A/B Testing Script: Compare N-Step Values
# ============================================================
# This script runs multiple training sessions with different
# n-step values to compare convergence speed

echo "============================================================"
echo "🧪 N-Step A/B Testing Experiment"
echo "============================================================"
echo ""
echo "This script will run 3 training sessions:"
echo "  1. Baseline (n=1) - Standard DQN"
echo "  2. Optimized (n=3) - Recommended setting"
echo "  3. Aggressive (n=5) - Fast convergence"
echo ""
echo "Each run will train for 2000 episodes."
echo "Results will be logged to separate W&B projects."
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."
echo ""

# Common settings
MODEL_NAME="ImpalaDueling"
EPISODES=2000
MAX_STEPS=4000
FRAME_STACK=4
FRAME_SKIP=4
BATCH_SIZE=128
BUFFER_SIZE=50000
LEARNING_RATE=0.0001
EPSILON_DECAY=0.995
OPTIMIZER="adamw"
WEIGHT_DECAY=0.0001
SCHEDULER="exponential"
SCHEDULER_GAMMA=0.999
MOVESET="balanced"
TEST_EVERY=50
SAVE_EVERY=50

# Experiment timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ============================================================
# Experiment 1: Baseline (n=1)
# ============================================================
echo "============================================================"
echo "📊 Experiment 1/3: Baseline (n=1)"
echo "============================================================"

N_STEP=1
WANDB_PROJECT="mario-nstep-baseline-${TIMESTAMP}"

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
    --wandb \
    --wandb-project ${WANDB_PROJECT}"

echo "Starting baseline training (n=1)..."
sudo docker run --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"

echo ""
echo "✅ Baseline experiment complete!"
echo ""
sleep 3

# ============================================================
# Experiment 2: Optimized (n=3)
# ============================================================
echo "============================================================"
echo "📊 Experiment 2/3: Optimized (n=3)"
echo "============================================================"

N_STEP=3
WANDB_PROJECT="mario-nstep-optimized-${TIMESTAMP}"

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
    --wandb \
    --wandb-project ${WANDB_PROJECT}"

echo "Starting optimized training (n=3)..."
sudo docker run --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"

echo ""
echo "✅ Optimized experiment complete!"
echo ""
sleep 3

# ============================================================
# Experiment 3: Aggressive (n=5)
# ============================================================
echo "============================================================"
echo "📊 Experiment 3/3: Aggressive (n=5)"
echo "============================================================"

N_STEP=5
WANDB_PROJECT="mario-nstep-aggressive-${TIMESTAMP}"

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
    --wandb \
    --wandb-project ${WANDB_PROJECT}"

echo "Starting aggressive training (n=5)..."
sudo docker run --gpus all -v $(pwd):/workspace light_mario-rl "$COMMAND"

echo ""
echo "✅ Aggressive experiment complete!"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "🎉 All Experiments Complete!"
echo "============================================================"
echo ""
echo "Results logged to W&B:"
echo "  • Baseline (n=1):    mario-nstep-baseline-${TIMESTAMP}"
echo "  • Optimized (n=3):   mario-nstep-optimized-${TIMESTAMP}"
echo "  • Aggressive (n=5):  mario-nstep-aggressive-${TIMESTAMP}"
echo ""
echo "Compare results at: https://wandb.ai"
echo ""
echo "Key metrics to compare:"
echo "  • Episodes to first completion"
echo "  • Final average score"
echo "  • Test completion rate"
echo "  • Training time"
echo ""
echo "Expected results:"
echo "  • n=1: Baseline performance"
echo "  • n=3: ~25% faster convergence ⚡"
echo "  • n=5: ~35% faster convergence ⚡⚡"
echo ""
echo "============================================================"
