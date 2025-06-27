import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from collections import deque

class MarioEnvironmentRL:
    def __init__(
        self,
        resize_shape=(84, 84),
        frame_stack=4,
        frame_skip=4,
    ):
        """
        RL-Optimized Mario Environment
        
        Args:
            resize_shape: Target size for frame preprocessing
            frame_stack: Number of frames to stack for temporal information
            frame_skip: Number of frames to skip (action repeat)
        """
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, [["left"], ["right"], ["right", "A"]])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.resized_shape = resize_shape
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        # Initialize frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)
        
        print("=" * 50)
        print("RL ENVIRONMENT INFORMATION".center(50))
        print("=" * 50)
        print(f"Action Space      : {self.action_space}")
        print(f"Observation Space : {self.observation_space}")
        print(f"Resized Shape     : {self.resized_shape}")
        print(f"Frame Stack       : {self.frame_stack}")
        print(f"Frame Skip        : {self.frame_skip}")
        print("=" * 50)
    
    def reset(self):
        """Reset environment and initialize frame stack"""
        state = self.env.reset()
        processed_frame = self.preprocess(state)
        
        # Fill frame buffer with initial frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        return self.get_stacked_frames(), state.copy()
    
    def step(self, action):
        """
        RL-optimized step with frame skipping and stacking
        """
        total_reward = 0
        done = False
        info = {}
        
        # Repeat action for frame_skip frames (action repeat)
        for i in range(self.frame_skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Always process the last frame, but for efficiency,
            # only process intermediate frames if we need them
            if i == self.frame_skip - 1 or done:
                processed_frame = self.preprocess(next_state)
                self.frames.append(processed_frame)
            
            if done:
                break
        
        return self.get_stacked_frames(), total_reward, done, info, next_state.copy()
    
    def preprocess(self, state):
        """Optimized preprocessing for RL"""
        # Convert to grayscale (faster than RGB)
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Resize to target shape
        resized = cv2.resize(gray, self.resized_shape, interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def get_stacked_frames(self):
        """Get current frame stack for RL agent"""
        return np.stack(self.frames, axis=0)
    
    def get_several_frames_rl(self, num_actions=4, action_sequence=None):
        """
        RL-specific method for collecting training data
        
        Args:
            num_actions: Number of actions to execute
            action_sequence: Specific sequence of actions, if None uses random
            
        Returns:
            tuple: (frame_sequence, rewards, done, final_info)
        """
        frame_sequence = []
        rewards = []
        
        for i in range(num_actions):
            if action_sequence is not None:
                action = action_sequence[i % len(action_sequence)]
            else:
                action = self.env.action_space.sample()
            
            stacked_frames, reward, done, info, _ = self.step(action)
            frame_sequence.append(stacked_frames.copy())
            rewards.append(reward)
            
            if done:
                break
        
        return np.array(frame_sequence), np.array(rewards), done, info
    
    def get_current_frame(self):
        """Get the current frame without stepping"""
        return self.env.render(mode='rgb_array')
    
    def close(self):
        self.env.close()

# Example usage for different RL scenarios
class RLFrameCollector:
    """Utility class for different RL frame collection strategies"""
    
    @staticmethod
    def collect_for_dqn(env, num_frames=4):
        """Optimized for DQN: Recent frames with temporal info"""
        return env.get_stacked_frames()
    
    @staticmethod
    def collect_for_a3c(env, sequence_length=20):
        """Optimized for A3C: Longer sequences for policy gradients"""
        frames, rewards, done, info = env.get_several_frames_rl(num_actions=sequence_length)
        return frames, rewards, done, info
    
    @staticmethod
    def collect_for_rainbow(env, n_step=3):
        """Optimized for Rainbow DQN: N-step returns"""
        frames, rewards, done, info = env.get_several_frames_rl(num_actions=n_step)
        return frames, rewards, done, info