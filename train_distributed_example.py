#!/usr/bin/env python3
"""
Example script demonstrating how to use distributed training for Mario RL

This script shows different ways to run distributed training:
1. Command line with arguments
2. Programmatic execution
"""

import subprocess
import sys
import torch

def check_distributed_requirements():
    """Check if system supports distributed training"""
    print("🔍 Checking distributed training requirements...")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    print(f"   • CUDA available: {cuda_available}")
    print(f"   • GPU count: {gpu_count}")

    if not cuda_available:
        print("❌ CUDA not available. Distributed training requires CUDA.")
        return False

    if gpu_count < 2:
        print("⚠️ Only 1 GPU detected. Distributed training works best with 2+ GPUs.")
        print("   You can still run distributed training for testing purposes.")

    return True

def run_single_gpu_training():
    """Run standard single GPU training"""
    print("\n🚀 Running single GPU training...")

    cmd = [
        sys.executable, "train.py",
        "--model-name", "ResNETv1",
        "--episodes", "100",  # Reduced for demo
        "--batch-size", "32",
        "--learning-rate", "0.0001"
    ]

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_distributed_training():
    """Run distributed training on multiple GPUs"""
    print("\n🚀 Running distributed training...")

    gpu_count = torch.cuda.device_count()

    cmd = [
        sys.executable, "train.py",
        "--distributed",
        "--world-size", str(gpu_count),
        "--model-name", "ResNETv1",
        "--episodes", "100",  # Reduced for demo
        "--batch-size", "64",  # Larger batch for distributed
        "--learning-rate", "0.0001"
    ]

    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    print("🎮 Mario RL Distributed Training Demo")
    print("=" * 50)

    if not check_distributed_requirements():
        print("\n❌ Requirements not met for distributed training.")
        return

    print("\nSelect training mode:")
    print("1. Single GPU training")
    print("2. Distributed training (multi-GPU)")
    print("3. Show command examples")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        run_single_gpu_training()
    elif choice == "2":
        run_distributed_training()
    elif choice == "3":
        show_command_examples()
    else:
        print("Invalid choice.")

def show_command_examples():
    """Show example commands for different training scenarios"""
    print("\n📝 Command Examples:")
    print("=" * 30)

    print("\n1. Single GPU training:")
    print("   python train.py --model-name ResNETv1 --episodes 5000")

    print("\n2. Distributed training on 2 GPUs:")
    print("   python train.py --distributed --world-size 2 --model-name ResNETv1 --episodes 5000")

    print("\n3. Distributed training with custom batch size:")
    print("   python train.py --distributed --world-size 4 --batch-size 128 --learning-rate 0.0002")

    print("\n4. Quick test run (distributed):")
    print("   python train.py --distributed --episodes 10 --test-every 5")

    print("\n5. Full parameter customization:")
    print("   python train.py --distributed \\")
    print("     --world-size 4 \\")
    print("     --model-name ResNETv1 \\")
    print("     --episodes 10000 \\")
    print("     --batch-size 128 \\")
    print("     --buffer-size 100000 \\")
    print("     --learning-rate 0.0002 \\")
    print("     --frame-stack 4 \\")
    print("     --moveset balanced")

if __name__ == "__main__":
    main()