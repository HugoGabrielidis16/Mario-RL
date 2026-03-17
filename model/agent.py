import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
                q_network,
                target_network,
                state_shape,
                n_actions,
                learning_rate=1e-4,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                batch_size=32,
                buffer_size=10000,
                rank=None,
                world_size=None,
                optimizer_type='adamw',
                weight_decay=1e-4,
                scheduler_type='exponential',
                scheduler_gamma=0.999,
                scheduler_step_size=1000,
                total_episodes=5000,
                n_step=1):

        self.rank = rank
        self.world_size = world_size
        self.is_distributed = rank is not None and world_size is not None

        # Device selection for distributed training
        if self.is_distributed:
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(rank)
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        if self.is_distributed:
            print(f"🌐 Initializing agent on device: {self.device} (rank {rank}/{world_size})")
        else:
            print(f"🌐 Initializing agent on device: {self.device}")


        print(f"🖥️  Using device: {self.device}")
        print(f"📊 State shape: {state_shape}")
        print(f"🎯 N_actions: {n_actions}")
        print(f"🔢 N-step returns: {n_step}")

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_step = n_step

        # N-step buffer for accumulating multi-step returns
        self.n_step_buffer = deque(maxlen=n_step)

        # Determine if we're using frame stacking
        self.is_frame_stacked = len(state_shape) == 3  # (frame_stack, H, W)
        if self.is_frame_stacked:
            self.frame_stack = state_shape[0]
            print(f"🎬 Frame stacking detected: {self.frame_stack} frames")
        else:
            self.frame_stack = 1
            print(f"📷 Single frame mode")

        # Neural networks - receive instances and move to device
        self.q_network = q_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Wrap with DDP if distributed training is enabled
        if self.is_distributed:
            self.q_network = DDP(self.q_network, device_ids=[rank], output_device=rank)
            # Note: target_network is not wrapped with DDP as it's only used for inference
        
        # Optimizer and replay buffer
        # Get parameters from the wrapped model if using DDP
        params = self.q_network.module.parameters() if self.is_distributed else self.q_network.parameters()

        # Choose optimizer type
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(params,
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(params,
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Choose scheduler type
        if scheduler_type.lower() == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                            gamma=scheduler_gamma)
        elif scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                T_max=total_episodes,
                                                                eta_min=learning_rate * 0.01)
        elif scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                     step_size=scheduler_step_size,
                                                     gamma=scheduler_gamma)
        elif scheduler_type.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode='max',
                                                                factor=0.5,
                                                                patience=100,  # Reduced from 1000
                                                                verbose=True)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self.scheduler_type = scheduler_type.lower()

        print(f"🔧 Optimizer: {optimizer_type} (lr={learning_rate}, weight_decay={weight_decay})")
        print(f"📈 Scheduler: {scheduler_type} (gamma={scheduler_gamma if 'gamma' in locals() else 'N/A'})")
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.step_count = 0
        self.total_reward = 0

    def decay_epsilon(self):
        """Decay epsilon - call once per episode"""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def step_scheduler(self, metric=None):
        """Step the learning rate scheduler"""
        if self.scheduler_type == 'plateau':
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
        
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Game state (frame_stack, H, W) for stacked frames or (H, W) for single frame
            training: Whether to use exploration (epsilon-greedy)
        """
        # Reset noise for NoisyNet before action selection
        if hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
        elif hasattr(self.q_network, 'module') and hasattr(self.q_network.module, 'reset_noise'):
            self.q_network.module.reset_noise()

        # Handle epsilon-greedy exploration during training (skip if using NoisyNet)
        is_noisy = (hasattr(self.q_network, 'noisy') and self.q_network.noisy) or \
                   (hasattr(self.q_network, 'module') and hasattr(self.q_network.module, 'noisy') and self.q_network.module.noisy)

        if training and not is_noisy and random.random() < self.epsilon:
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
        Store experience in replay buffer with n-step returns

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

        # Add transition to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we have enough steps or episode ended, compute n-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # Get the first transition from the buffer
            first_state, first_action, _, _, _ = self.n_step_buffer[0]

            # Compute n-step return
            n_step_reward = 0
            n_step_state = next_state
            n_step_done = done

            # Sum discounted rewards
            for i, (_, _, r, s, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                n_step_state = s
                n_step_done = d
                if d:  # Episode ended, stop accumulating
                    break

            # Store n-step transition in replay buffer
            self.replay_buffer.push(first_state, first_action, n_step_reward, n_step_state, n_step_done)

        # If episode ended, flush remaining transitions in n-step buffer
        if done and len(self.n_step_buffer) > 1:
            # Process remaining transitions (they'll have fewer than n steps)
            buffer_list = list(self.n_step_buffer)
            for start_idx in range(1, len(buffer_list)):
                first_state, first_action, _, _, _ = buffer_list[start_idx]

                # Compute remaining steps return
                n_step_reward = 0
                n_step_state = next_state
                n_step_done = True  # Episode ended

                for i, (_, _, r, s, d) in enumerate(buffer_list[start_idx:]):
                    n_step_reward += (self.gamma ** i) * r
                    n_step_state = s

                self.replay_buffer.push(first_state, first_action, n_step_reward, n_step_state, n_step_done)

            # Clear n-step buffer at episode end
            self.n_step_buffer.clear()

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

        # Reset noise for NoisyNet before training
        if hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
        elif hasattr(self.q_network, 'module') and hasattr(self.q_network.module, 'reset_noise'):
            self.q_network.module.reset_noise()

        if hasattr(self.target_network, 'reset_noise'):
            self.target_network.reset_noise()

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

        # Compute next Q-values (Double DQN style) with n-step returns
        # Note: rewards_tensor already contains n-step cumulative rewards
        # So we need to use gamma^n for bootstrapping
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            # Use gamma^n_step for n-step bootstrapping
            gamma_n = self.gamma ** self.n_step
            target_q_values = rewards_tensor + (gamma_n * next_q_values * (1 - dones_tensor))

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability - handle DDP case
        if self.is_distributed:
            torch.nn.utils.clip_grad_norm_(self.q_network.module.parameters(), max_norm=1.0)
        else:
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
        # Get state dict from the wrapped model if using DDP
        if self.is_distributed:
            self.target_network.load_state_dict(self.q_network.module.state_dict())
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save agent state"""
        # Only save from rank 0 in distributed training
        if self.is_distributed and self.rank != 0:
            return

        # Get state dict from the wrapped model if using DDP
        q_network_state = self.q_network.module.state_dict() if self.is_distributed else self.q_network.state_dict()

        torch.save({
            'q_network_state_dict': q_network_state,
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

        # Load state dict into the wrapped model if using DDP
        if self.is_distributed:
            self.q_network.module.load_state_dict(checkpoint['q_network_state_dict'])
        else:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])

        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint.get('step_count', 0)
        self.total_reward = checkpoint.get('total_reward', 0)

        if not self.is_distributed or self.rank == 0:
            print(f"📂 Agent loaded from {filepath}")
            print(f"   • Epsilon: {self.epsilon:.4f}")
            print(f"   • Steps: {self.step_count}")
            print(f"   • Total reward: {self.total_reward:.2f}")
    
    def get_stats(self):
        """Get training statistics"""
        stats = {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }

        if self.is_distributed:
            stats.update({
                'rank': self.rank,
                'world_size': self.world_size,
                'distributed': True
            })

        return stats
    
    def set_training_mode(self, training=True):
        """Set training/evaluation mode"""
        if training:
            self.q_network.train()
            self.target_network.train()
        else:
            self.q_network.eval()
            self.target_network.eval()