import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self,
                model, 
                state_shape, 
                n_actions, 
                learning_rate=1e-4, 
                gamma=0.99, 
                epsilon_start=1.0, 
                epsilon_end=0.01, 
                epsilon_decay=0.995,
                batch_size=32,
                buffer_size=10000):
        
        self.model = model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"🖥️  Using device: {self.device}")
        print(f"📊 State shape: {state_shape}")
        print(f"🎯 N_actions: {n_actions}")
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Determine if we're using frame stacking
        self.is_frame_stacked = len(state_shape) == 3  # (frame_stack, H, W)
        if self.is_frame_stacked:
            self.frame_stack = state_shape[0]
            print(f"🎬 Frame stacking detected: {self.frame_stack} frames")
        else:
            self.frame_stack = 1
            print(f"📷 Single frame mode")
        
        # Neural networks
        self.q_network = self.model(state_shape=state_shape, n_actions=n_actions).to(self.device)
        self.target_network = self.model(state_shape=state_shape, n_actions=n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.step_count = 0
        self.total_reward = 0

    def decay_epsilon(self):
        """Decay epsilon - call once per episode"""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Game state (frame_stack, H, W) for stacked frames or (H, W) for single frame
            training: Whether to use exploration (epsilon-greedy)
        """
        # Handle epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Convert state to tensor and ensure correct shape
        state_tensor = self._prepare_state_tensor(state)
        
        # Get Q-values and select best action
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def _prepare_state_tensor(self, state):
        """
        Convert state to properly shaped tensor for the network
        
        Args:
            state: numpy array or tensor
            
        Returns:
            torch.Tensor: Properly shaped tensor (1, frame_stack, H, W) or (1, 1, H, W)
        """
        # Convert to numpy if tensor
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Handle different input shapes
        if self.is_frame_stacked:
            # Expected: (frame_stack, H, W) -> (1, frame_stack, H, W)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            elif state_tensor.dim() == 4:
                # Already batched, ensure correct shape
                if state_tensor.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1, got {state_tensor.shape[0]}")
            else:
                raise ValueError(f"Invalid state shape: {state_tensor.shape}")
        else:
            # Single frame: (H, W) -> (1, 1, H, W)
            if state_tensor.dim() == 2:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            elif state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            elif state_tensor.dim() == 4:
                # Already batched
                if state_tensor.shape[0] != 1:
                    raise ValueError(f"Expected batch size 1, got {state_tensor.shape[0]}")
            else:
                raise ValueError(f"Invalid state shape: {state_tensor.shape}")
        
        return state_tensor.to(self.device)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Convert to numpy arrays for consistent storage
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Update metrics
        self.step_count += 1
        self.total_reward += reward
    
    def replay(self, batch_size=None):
        """
        Train the agent on a batch of experiences
        
        Args:
            batch_size: Size of training batch (uses default if None)
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors with proper shapes
        states_tensor = self._prepare_batch_tensor(states)
        next_states_tensor = self._prepare_batch_tensor(next_states)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Compute next Q-values (Double DQN style)
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Epsilon decay moved to training loop (per episode, not per step)
        
        return loss.item()
    
    def _prepare_batch_tensor(self, states_batch):
        """
        Convert batch of states to properly shaped tensor
        
        Args:
            states_batch: List of state arrays
            
        Returns:
            torch.Tensor: Properly shaped batch tensor
        """
        # Stack states into batch
        states_array = np.array(states_batch)
        states_tensor = torch.FloatTensor(states_array)
        
        # Handle different shapes
        if self.is_frame_stacked:
            # Expected: (batch_size, frame_stack, H, W)
            if states_tensor.dim() != 4:
                raise ValueError(f"Expected 4D tensor for frame stacked states, got {states_tensor.dim()}D")
        else:
            # Single frame: (batch_size, H, W) -> (batch_size, 1, H, W)
            if states_tensor.dim() == 3:
                states_tensor = states_tensor.unsqueeze(1)
            elif states_tensor.dim() != 4:
                raise ValueError(f"Expected 3D or 4D tensor for single frame states, got {states_tensor.dim()}D")
        
        return states_tensor.to(self.device)
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save agent state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions
        }, filepath)
        print(f"💾 Agent saved to {filepath}")
        
    def load(self, filepath):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint.get('step_count', 0)
        self.total_reward = checkpoint.get('total_reward', 0)
        print(f"📂 Agent loaded from {filepath}")
        print(f"   • Epsilon: {self.epsilon:.4f}")
        print(f"   • Steps: {self.step_count}")
        print(f"   • Total reward: {self.total_reward:.2f}")
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }
    
    def set_training_mode(self, training=True):
        """Set training/evaluation mode"""
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()