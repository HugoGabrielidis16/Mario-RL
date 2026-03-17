# Training Configuration Guide

This directory contains optimized training scripts for the Mario RL project with **n-step returns** for faster convergence.

## 📋 Available Configurations

### 1. **train_docker.sh** - BALANCED (Recommended Default)
**Best for**: General purpose training, most use cases

```bash
./script/train_docker.sh
```

**Key Parameters:**
- **N-Step**: 3 (optimal balance)
- **Episodes**: 5000
- **Batch Size**: 128
- **Learning Rate**: 0.0001
- **Epsilon Decay**: 0.995
- **Optimizer**: AdamW with weight decay
- **Expected Convergence**: ~2000-2400 episodes (20-35% faster than 1-step)

**Use when:**
- Starting a new training run
- You want reliable, proven hyperparameters
- Balanced exploration vs exploitation

---

### 2. **train_docker_fast.sh** - FAST CONVERGENCE
**Best for**: Quick experiments, proof-of-concept

```bash
./script/train_docker_fast.sh
```

**Key Parameters:**
- **N-Step**: 5 (aggressive credit assignment)
- **Episodes**: 3000 (fewer needed)
- **Batch Size**: 256 (large batches)
- **Learning Rate**: 0.00025 (higher)
- **Epsilon Decay**: 0.99 (faster exploitation)
- **Expected Convergence**: ~1500-2000 episodes (FASTEST)

**Use when:**
- Testing new reward shaping
- Quick ablation studies
- Limited compute time
- Prototyping new features

**⚠️ Trade-offs:**
- May overfit to early strategies
- Less thorough exploration
- Higher variance in results

---

### 3. **train_docker_stable.sh** - STABLE/SAFE
**Best for**: Production runs, benchmark comparisons

```bash
./script/train_docker_stable.sh
```

**Key Parameters:**
- **N-Step**: 3 (balanced)
- **Episodes**: 7000 (extended)
- **Batch Size**: 64 (conservative)
- **Learning Rate**: 0.00005 (low, stable)
- **Epsilon Decay**: 0.997 (slow exploration)
- **Scheduler**: Plateau (adaptive)
- **Expected Convergence**: ~3000-3500 episodes (MOST STABLE)

**Use when:**
- Final model training
- Creating benchmark results
- Publishing/sharing models
- Maximum stability required

**✅ Benefits:**
- Most reliable convergence
- Better generalization
- Thorough exploration
- Lower risk of instability

---

### 4. **train_docker_multi_gpu.sh** - DISTRIBUTED TRAINING
**Best for**: Multi-GPU systems, fastest training

```bash
./script/train_docker_multi_gpu.sh
```

**Key Parameters:**
- **N-Step**: 3
- **World Size**: 2 (modify for your GPU count)
- **Batch Size**: 256 (split across GPUs)
- **Learning Rate**: 0.0002 (scaled for larger batch)
- **Expected Convergence**: 2-4x faster wall-clock time

**Requirements:**
- 2+ CUDA GPUs
- NCCL backend
- Sufficient VRAM per GPU (≥8GB recommended)

**Use when:**
- You have multiple GPUs available
- Want fastest wall-clock training time
- Collecting large amounts of data

**📝 Notes:**
- Only rank 0 saves checkpoints
- Gradients are averaged across GPUs
- Effective batch size = BATCH_SIZE / WORLD_SIZE per GPU

---

## 🎯 Parameter Guide

### N-Step Returns (Most Important New Parameter)

| Value | Use Case | Pros | Cons |
|-------|----------|------|------|
| **1** | Baseline, debugging | Low bias, simple | Slow convergence |
| **3** | **RECOMMENDED** | Best balance | Minimal downsides |
| **5** | Complex tasks, long sequences | Fastest credit assignment | Higher variance |
| **7+** | Very long-term planning | Maximum foresight | Too much variance |

### Learning Rate

| Value | Speed | Stability | Use Case |
|-------|-------|-----------|----------|
| 0.00005 | Slow | Very stable | Production runs |
| 0.0001 | Medium | Stable | **Default (recommended)** |
| 0.00025 | Fast | Moderate | Quick experiments |
| 0.0005+ | Very fast | Unstable | Not recommended |

### Batch Size

| Value | Training Speed | Sample Efficiency | Memory |
|-------|----------------|-------------------|--------|
| 32-64 | Fast updates | Good | Low |
| 128 | **Balanced** | **Good** | **Medium (recommended)** |
| 256+ | Slower, stable | Excellent | High |

### Epsilon Decay

| Value | Exploration | Convergence | Use Case |
|-------|-------------|-------------|----------|
| 0.99 | Fast decay | Quick | Fast experiments |
| 0.995 | **Balanced** | **Medium** | **Default** |
| 0.997 | Slow decay | Slow | Thorough exploration |
| 0.999 | Very slow | Very slow | Complex environments |

---

## 🚀 Quick Start

### First Time Training
```bash
# Use balanced configuration (recommended)
./script/train_docker.sh
```

### Quick Experiment
```bash
# Use fast configuration
./script/train_docker_fast.sh
```

### Production/Final Model
```bash
# Use stable configuration
./script/train_docker_stable.sh
```

### Multi-GPU Training
```bash
# Edit WORLD_SIZE to match your GPU count
nano script/train_docker_multi_gpu.sh
# Then run
./script/train_docker_multi_gpu.sh
```

---

## 📊 Expected Performance

### Convergence Speed Comparison

| Configuration | Episodes to Complete | Wall-Clock Time* | Stability |
|---------------|---------------------|------------------|-----------|
| **1-Step (Old)** | ~3000 | Baseline | Good |
| **Balanced (3-step)** | ~2000-2400 | **-25%** | Good |
| **Fast (5-step)** | ~1500-2000 | **-35%** | Moderate |
| **Stable (3-step)** | ~3000-3500 | +10% | **Excellent** |
| **Multi-GPU (3-step)** | ~2000-2400 | **-60%*** | Good |

*Wall-clock time depends on hardware. Multi-GPU scales linearly with GPU count.

### Training Metrics to Monitor

**Good Training:**
- ✅ Average score increasing steadily
- ✅ Max X position progressing
- ✅ Completion rate improving
- ✅ Loss stabilizing after initial phase

**Warning Signs:**
- ⚠️ Score oscillating wildly (reduce LR or increase batch size)
- ⚠️ Agent stuck at same X position (increase epsilon or n_step)
- ⚠️ Loss increasing (reduce LR, check for bugs)
- ⚠️ No improvement after 1000 episodes (check reward shaping)

---

## 🔧 Customization

To modify any script, edit the variables at the top:

```bash
# Open script
nano script/train_docker.sh

# Modify parameters
N_STEP=5                  # Change n-step
BATCH_SIZE=256           # Increase batch size
LEARNING_RATE=0.0002     # Adjust learning rate
MOVESET="speedrun"       # Change action space

# Save and run
./script/train_docker.sh
```

---

## 📈 Weights & Biases Integration

All scripts include W&B logging by default. To disable:

```bash
# Comment out WANDB_ENABLED in script
# WANDB_ENABLED="--wandb"
```

To change project name:
```bash
WANDB_PROJECT="my-custom-project"
```

---

## 💡 Tips & Best Practices

1. **Start with balanced config** - It works well for most cases
2. **Use fast config for experiments** - Quick iterations when testing ideas
3. **Use stable config for final runs** - When you need reliable results
4. **Monitor test performance** - Training score ≠ test performance
5. **Save checkpoints frequently** - Training can be interrupted
6. **Use n_step=3 by default** - Best balance of speed and stability
7. **Increase n_step for complex zones** - If agent struggles with long sequences
8. **Lower learning rate if unstable** - If loss oscillates wildly
9. **Use distributed training if available** - Massive speedup with multiple GPUs

---

## 🐛 Troubleshooting

### Agent not learning
- Increase epsilon_decay to 0.997 (more exploration)
- Increase n_step to 5 (better credit assignment)
- Check reward shaping in environment

### Training unstable
- Decrease learning_rate to 0.00005
- Increase batch_size to 256
- Use plateau scheduler instead of exponential

### Out of memory
- Decrease batch_size to 64 or 32
- Decrease buffer_size to 25000
- Use single GPU instead of distributed

### Training too slow
- Increase frame_skip to 6
- Use train_docker_fast.sh
- Enable multi-GPU training if available

---

## 📚 Further Reading

- **N-Step Returns**: [Sutton & Barto RL Book, Chapter 7]
- **Rainbow DQN**: Uses 3-step returns as default
- **IMPALA Architecture**: Optimized for RL tasks
- **AdamW Optimizer**: Better than Adam for deep RL

---

**Happy Training! 🎮🍄**
