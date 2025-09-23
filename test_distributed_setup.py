#!/usr/bin/env python3
"""
Test script to validate distributed training setup without running full training
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from model.agent import Agent
from model.DQN import DQN
from model.ResNET import MultiFrameResNet
import os

def test_ddp_setup(rank, world_size):
    """Test DDP initialization"""
    try:
        # Setup environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize process group
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"✅ Rank {rank}: DDP initialized successfully")

        # Test device assignment
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"✅ Rank {rank}: Device set to {device}")

        # Test model creation with DDP
        state_shape = (4, 84, 84)  # frame_stack, H, W
        n_actions = 7

        # Test DQN model
        agent_dqn = Agent(
            model=DQN,
            state_shape=state_shape,
            n_actions=n_actions,
            rank=rank,
            world_size=world_size
        )
        print(f"✅ Rank {rank}: DQN Agent created with DDP")

        # Test ResNet model
        agent_resnet = Agent(
            model=MultiFrameResNet,
            state_shape=state_shape,
            n_actions=n_actions,
            rank=rank,
            world_size=world_size
        )
        print(f"✅ Rank {rank}: ResNet Agent created with DDP")

        # Test forward pass
        dummy_input = torch.randn(1, 4, 84, 84).cuda(rank)
        with torch.no_grad():
            output = agent_dqn.q_network(dummy_input)
            print(f"✅ Rank {rank}: Forward pass successful, output shape: {output.shape}")

        # Test synchronization
        dist.barrier()
        print(f"✅ Rank {rank}: Synchronization successful")

    except Exception as e:
        print(f"❌ Rank {rank}: Error - {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def test_single_gpu():
    """Test single GPU setup"""
    try:
        print("🧪 Testing single GPU setup...")

        state_shape = (4, 84, 84)
        n_actions = 7

        # Test without DDP
        agent = Agent(
            model=DQN,
            state_shape=state_shape,
            n_actions=n_actions
        )
        print("✅ Single GPU: DQN Agent created successfully")

        # Test forward pass
        dummy_input = torch.randn(1, 4, 84, 84)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()

        with torch.no_grad():
            output = agent.q_network(dummy_input)
            print(f"✅ Single GPU: Forward pass successful, output shape: {output.shape}")

        print("✅ Single GPU setup test passed!")

    except Exception as e:
        print(f"❌ Single GPU test failed: {e}")

def main():
    print("🧪 Testing Mario RL Distributed Setup")
    print("=" * 40)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping distributed tests.")
        test_single_gpu()
        return

    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} GPU(s)")

    # Test single GPU first
    test_single_gpu()

    if gpu_count >= 2:
        print(f"\n🚀 Testing distributed setup with {min(2, gpu_count)} GPUs...")
        try:
            mp.spawn(test_ddp_setup,
                     args=(min(2, gpu_count),),
                     nprocs=min(2, gpu_count),
                     join=True)
            print("✅ Distributed test completed!")
        except Exception as e:
            print(f"❌ Distributed test failed: {e}")
    else:
        print("⚠️ Need at least 2 GPUs for distributed testing. Skipping distributed tests.")

    print("\n📋 Test Summary:")
    print("   • Single GPU setup: Tested")
    if gpu_count >= 2:
        print("   • Distributed setup: Tested")
    else:
        print("   • Distributed setup: Skipped (insufficient GPUs)")

if __name__ == "__main__":
    main()