#!/bin/bash
# Mario RL Training Environment Setup Script - FIXED VERSION

echo "🍄 Setting up Mario RL Training Environment 🍄"

# Check if conda/mamba is available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "❌ Error: conda or mamba not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "📦 Using $CONDA_CMD for package management"

# Create conda environment
echo "🔨 Creating conda environment 'mario-rl'..."
$CONDA_CMD create -n mario-rl python=3.10 -y

# Activate environment
echo "🎯 Activating environment..."
source $($CONDA_CMD info --base)/etc/profile.d/conda.sh
$CONDA_CMD activate mario-rl

# Upgrade pip and setuptools first
echo "🔧 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools==68.2.2 wheel==0.41.2

# Install PyTorch first (CPU version - change if you have GPU)
echo "🧠 Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Install compatibility layer first
echo "🔗 Installing compatibility layer..."
pip install shimmy[gym-v21]==1.3.0

# Install stable-baselines3 and gymnasium
echo "🤖 Installing RL framework..."
pip install gymnasium==0.29.1
pip install stable-baselines3==2.2.1

# Install Mario environment
echo "🎮 Installing Mario environment..."
pip install nes-py==8.2.1
pip install gym-super-mario-bros==7.4.0

# Install remaining dependencies
echo "📚 Installing other dependencies..."
pip install opencv-python==4.8.1.78
pip install matplotlib==3.8.0
pip install numpy==1.24.3
pip install scipy==1.11.3
pip install tqdm==4.66.1
pip install tensorboard==2.15.1
pip install cloudpickle==2.2.1
pip install psutil==5.9.6
pip install Pillow==10.0.0

echo "✅ Installation complete!"
echo ""
echo "🚀 To use the environment:"
echo "   conda activate mario-rl"
echo "   python mario.py"
echo ""
echo "📊 To view training logs:"
echo "   tensorboard --logdir ./mario_tensorboard/"
echo ""
echo "🎮 Happy training!"