import torch
import time
import datetime
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

from model.agent import Agent
from model.DQN import DQN

import warnings
warnings.filterwarnings("ignore")


# Environment wrapper for preprocessing
class MarioEnvironment:
    def __init__(self):
        """
        """
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, [["left"],["right"], ["right", "A"],])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        
    def reset(self):
        state = self.env.reset()
        # Ensure we return a copy of the state
        return self.preprocess(state), state.copy()
    
    def step(self, action):
        print("Action", action)
        print(self.env.step(action))
        next_state, reward, done, info = self.env.step(action)
        # Ensure we return a copy of the state
        return self.preprocess(next_state), reward, done, info, next_state.copy()
    
    def preprocess(self, state):
        # Convert to grayscale
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        # Normalize
        normalized = resized / 255.0
        return normalized
    
    def get_current_frame(self):
        """Get the current frame without stepping"""
        return self.env.render(mode='rgb_array')
    
    def close(self):
        self.env.close()

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size after convolutions
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out.view(conv_out.size(0), -1))

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_shape, n_actions, learning_rate=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Neural networks
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        print(f"Dones value :{dones}")
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def save_frames_as_gif(frames, episode, fps=30):
    """Save frames as a GIF for visualization"""
    os.makedirs(SAVING_FOLDER, exist_ok=True)
    
    if not frames:
        print(f"No frames to save for episode {episode}")
        return
    
    # Debug first frame
    print(f"Saving GIF with {len(frames)} frames")
    print(f"First frame shape: {frames[0].shape}, dtype: {frames[0].dtype}, min: {frames[0].min()}, max: {frames[0].max()}")
    
    # Convert frames to PIL Images
    pil_frames = []
    for i, frame in enumerate(frames[::3]):  # Take every 3rd frame to reduce size
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure frame is in correct format (RGB)
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # If RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
        # Resize frame for smaller file size
        resized = cv2.resize(frame, (256, 240))
        pil_frames.append(Image.fromarray(resized))
    
    # Save as GIF
    if pil_frames:
        pil_frames[0].save(
            f'{SAVING_FOLDER}/episode_{episode}.gif',
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000//fps,
            loop=0
        )
        print(f"Saved gameplay GIF: {SAVING_FOLDER}/episode_{episode}.gif")

def train_mario(episodes=1000, render_every=100, save_every=100):
    # Create environment
    env = MarioEnvironment()
    
    # Create agent
    state_shape = (84, 84)
    n_actions = env.action_space.n

    model = DQN(input_shape= state_shape,
                n_actions= n_actions)

    agent = DQNAgent(state_shape, n_actions)
    
    # Training metrics
    scores = []
    moving_avg = deque(maxlen=100)
    
    # Create save directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    
    for episode in range(episodes):
        state, raw_state = env.reset()
        total_reward = 0
        steps = 0
        
        # Store frames for visualization
        frames = []
        save_frames_this_episode = (episode % render_every == 0 and episode > 0)
        
        # Debug: Check initial frame
        if save_frames_this_episode:
            print(f"Episode {episode} - Initial frame shape: {raw_state.shape}, dtype: {raw_state.dtype}")
            if raw_state.shape[2] == 3:  # RGB frame
                frames.append(raw_state.copy())
        
        while True:
            # Select and perform action
            action = agent.act(state)
            next_state, reward, done, info, raw_next_state = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            if len(agent.replay_buffer) > 32:
                agent.replay()
            
            # Store frame for visualization - ensure it's the raw RGB frame
            if save_frames_this_episode and steps % 5 == 0:  # Save every 5th frame
                # Make sure we have a valid RGB frame
                if raw_next_state is not None and raw_next_state.shape[2] == 3:
                    frames.append(raw_next_state.copy())
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Save gameplay as GIF
        if save_frames_this_episode and frames:
            save_frames_as_gif(frames, episode)
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Track progress
        scores.append(total_reward)
        moving_avg.append(total_reward)
        avg_score = np.mean(moving_avg)
        
        # Print progress
        print(f"Episode: {episode + 1}/{episodes}, Score: {total_reward:.2f}, "
              f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, "
              f"Steps: {steps}, X Position: {info.get('x_pos', 0)}")
        
        # Save checkpoint
        if (episode + 1) % save_every == 0:
            checkpoint_path = f'checkpoints/mario_dqn_episode_{episode + 1}.pt'
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Plot progress every 100 episodes
        if (episode + 1) % 100 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(scores)
            if len(scores) > 100:
                plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid'))
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Score', 'Moving Avg (100)'])
            
            plt.subplot(1, 2, 2)
            x_positions = [s for s in scores]  # You could track actual x_pos here
            plt.plot(range(len(scores)), x_positions)
            plt.title('Episodes vs Score')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(f'training_plots/progress_episode_{episode + 1}.png')
            plt.close()
            print(f"Progress plot saved: training_plots/progress_episode_{episode + 1}.png")
    
    env.close()
    return agent, scores

def test_agent(agent, episodes=5, save_gif=True):
    """Test the trained agent"""
    env = MarioEnvironment()
    
    for episode in range(episodes):
        state, raw_state = env.reset()
        total_reward = 0
        frames = []
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info, raw_next_state = env.step(action)
            
            frames.append(raw_next_state)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Test Episode {episode + 1}: Score = {total_reward}, "
                      f"X Position = {info.get('x_pos', 0)}")
                if save_gif:
                    save_frames_as_gif(frames, f"test_{episode + 1}", saving_folder = SAVING_FOLDER)
                break
    env.close()

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    # Train the agent
    EPISODES = 1000
    RENDER_EVERY = 1
    SAVE_EVERY = 100
    SAVING_FOLDER = f"gameplay_gifs/{current_time.day}d_{current_time.hour}h_{current_time.minute}s"
    print("Starting Mario DQN Training...")
    print(f"Note: Every {RENDER_EVERY} episodes, a GIF will be saved showing Mario's gameplay")
    agent, scores = train_mario(episodes=EPISODES, render_every=RENDER_EVERY, save_every=SAVE_EVERY)
    # Test the trained agent
    print("\nTesting trained agent...")
    agent.epsilon = 0.01  # Use mostly learned policy
    test_agent(agent, episodes=3)