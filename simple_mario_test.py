#!/usr/bin/env python3
"""
Simple test to verify Mario RL environment is working
Run this before training to make sure everything is set up correctly
"""

import gymnasium
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
import numpy as np

class GymToGymnasiumWrapper(gymnasium.Wrapper):
    """Convert old gym API to new gymnasium API"""
    
    def __init__(self, env):
        # Convert gym env to gymnasium-compatible
        self.unwrapped_env = env
        super().__init__(env)
    
    @property
    def observation_space(self):
        return self.unwrapped_env.observation_space
    
    @property
    def action_space(self):
        return self.unwrapped_env.action_space
    
    def reset(self, **kwargs):
        obs = self.unwrapped_env.reset()
        if isinstance(obs, tuple):
            return obs  # Already in new format
        else:
            return obs, {}  # Convert to new format
    
    def step(self, action):
        result = self.unwrapped_env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info  # Add truncated=False
        else:
            return result  # Already in new format
    
    def render(self, mode='human'):
        return self.unwrapped_env.render()
    
    def close(self):
        return self.unwrapped_env.close()

def test_mario_setup():
    """Test that Mario environment and PPO work together"""
    print("🍄 Testing Mario RL Setup 🍄")
    print("=" * 40)
    
    # Step 1: Create Mario environment
    print("1️⃣ Creating Mario environment...")
    try:
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GymToGymnasiumWrapper(env)  # Our custom wrapper
        print("✅ Mario environment created successfully!")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Step 2: Test environment reset and step
    print("\n2️⃣ Testing environment interaction...")
    try:
        obs, info = env.reset()
        print(f"✅ Environment reset successful!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        
        # Take a few random actions
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")
            
    except Exception as e:
        print(f"❌ Environment interaction failed: {e}")
        return False
    
    # Step 3: Test PPO model creation
    print("\n3️⃣ Testing PPO model creation...")
    try:
        model = PPO('CnnPolicy', env, verbose=0)
        print("✅ PPO model created successfully!")
        
        # Test prediction
        action, _states = model.predict(obs, deterministic=True)
        print(f"   Model prediction: action={action}")
        
    except Exception as e:
        print(f"❌ PPO model creation failed: {e}")
        return False
    
    # Step 4: Test short training
    print("\n4️⃣ Testing short training run...")
    try:
        print("   Training for 100 steps...")
        model.learn(total_timesteps=100, progress_bar=False)
        print("✅ Short training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    env.close()
    
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("🚀 Your environment is ready for Mario RL training!")
    print("\nTo start training, run:")
    print("   python mario.py")
    print("   Choose option 1 to test environment")
    print("   Choose option 2 or 3 to start training")
    
    return True

if __name__ == "__main__":
    success = test_mario_setup()
    if not success:
        print("\n❌ Setup verification failed!")
        print("💡 Try reinstalling dependencies or check the error messages above.")
    else:
        print("\n✅ Setup verification passed!")
        print("🎮 Ready to train Mario!")