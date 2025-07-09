import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from collections import deque


def get_mario_action_set(complexity="balanced"):
    """
    Get different action sets for Mario based on training complexity
    
    Args:
        complexity: "minimal", "balanced", "complete", "reckless", "speedrun"
    
    Returns:
        list: Action combinations for JoypadSpace
    """
    
    action_sets = {
        # 🔰 MINIMAL (Current - Too Limited!)
        "minimal": [
            ["right"],          # 0: Move right
            ["right", "A"]      # 1: Move right + jump
        ],
        
        # 🎯 BALANCED (Recommended for Learning)
        "balanced": [
            [],                 # 0: No action (important for timing!)
            ["right"],          # 1: Move right
            ["right", "A"],     # 2: Move right + jump
            ["A"],              # 3: Jump in place
            ["left"],           # 4: Move left (escape/positioning)
            ["down"],           # 5: Duck/crouch (avoid enemies, enter pipes)
            ["right", "B"],     # 6: Run right (speed)
            ["right", "A", "B"] # 7: Run right + jump (long jumps)
        ],
        
        # 🏃‍♂️ RECKLESS (Aggressive Movement)
        "reckless": [
            ["right"],              # 0: Move right
            ["right", "A"],         # 1: Jump right  
            ["right", "B"],         # 2: Run right
            ["right", "A", "B"],    # 3: Run + jump right
            ["A"],                  # 4: Jump in place
            ["right", "B", "A"],    # 5: Running jump (same as 3, for emphasis)
            ["B"],                  # 6: Run in place (build momentum)
        ],
        
        # 🚀 SPEEDRUN (Fast Completion)
        "speedrun": [
            ["right"],              # 0: Walk right
            ["right", "B"],         # 1: Run right  
            ["right", "A", "B"],    # 2: Running jump
            ["A", "B"],             # 3: High jump
            ["right", "A"],         # 4: Walk + jump
            [],                     # 5: Stop (rare but needed)
        ],
        
        # 🎮 COMPLETE (Advanced Players)
        "complete": [
            [],                     # 0: No action
            ["right"],              # 1: Walk right
            ["left"],               # 2: Walk left  
            ["down"],               # 3: Duck
            ["A"],                  # 4: Jump
            ["B"],                  # 5: Run/fire
            ["right", "A"],         # 6: Jump right
            ["left", "A"],          # 7: Jump left
            ["right", "B"],         # 8: Run right
            ["left", "B"],          # 9: Run left
            ["right", "A", "B"],    # 10: Running jump right
            ["left", "A", "B"],     # 11: Running jump left
            ["down", "A"],          # 12: Duck jump (rare but useful)
        ],
        "own" : [
            []
        ]
    }
    
    return action_sets.get(complexity, action_sets["balanced"])



class CUSTOMMarioEnvironmentRL:
    def __init__(
        self,
        resize_shape=(84, 84),
        frame_stack=4,
        frame_skip=4,
        reward_shaping=True,  # Enable custom reward shaping
        action_set = "balanced",
    ):
        """
        RL-Optimized Mario Environment with Custom Reward Shaping
        
        Args:
            resize_shape: Target size for frame preprocessing
            frame_stack: Number of frames to stack for temporal information
            frame_skip: Number of frames to skip (action repeat)
            reward_shaping: Whether to apply custom reward modifications
        """
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, get_mario_action_set(action_set))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.resized_shape = resize_shape
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.reward_shaping = reward_shaping
        
        # Initialize frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)
        
        # Reward shaping tracking variables
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_life = 3
        self.prev_status = "small"
        self.stuck_counter = 0
        self.max_x_pos = 0  # Track furthest progress
        
        print("=" * 50)
        print("RL ENVIRONMENT INFORMATION".center(50))
        print("=" * 50)
        print(f"Action Space      : {self.action_space}")
        print(f"Observation Space : {self.observation_space}")
        print(f"Resized Shape     : {self.resized_shape}")
        print(f"Frame Stack       : {self.frame_stack}")
        print(f"Frame Skip        : {self.frame_skip}")
        print(f"Reward Shaping    : {self.reward_shaping}")
        print("=" * 50)
    
    def reset(self):
        """Reset environment and initialize frame stack"""
        state = self.env.reset()
        processed_frame = self.preprocess(state)
        
        # Reset reward tracking variables
        self.prev_x_pos = 0
        self.prev_score = 0
        self.prev_coins = 0
        self.prev_life = 3
        self.prev_status = "small"
        self.stuck_counter = 0
        self.max_x_pos = 0

        
        # Fill frame buffer with initial frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        return self.get_stacked_frames(), state.copy()
    
    def step(self, action):
        """
        RL-optimized step with frame skipping, stacking, and reward shaping
        """
        total_reward = 0
        done = False
        info = {}
        
        # Repeat action for frame_skip frames (action repeat)
        for i in range(self.frame_skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward # When not using GYM 
            
            # Always process the last frame, but for efficiency,
            # only process intermediate frames if we need them
            if i == self.frame_skip - 1 or done:
                processed_frame = self.preprocess(next_state)
                self.frames.append(processed_frame)
            
            if done:
                break
        
        # Apply custom reward shaping
        if self.reward_shaping:
            shaped_reward = self.calculate_shaped_reward(total_reward, info, done)
            total_reward = shaped_reward
        
        return self.get_stacked_frames(), total_reward, done, info, next_state.copy()
    
    def calculate_shaped_reward(self, original_reward, info, done):
        """
        Custom reward shaping to improve learning
        
        Args:
            original_reward: Raw reward from gym environment
            info: Game state information
            done: Whether episode ended
            
        Returns:
            float: Shaped reward
        """
        reward = original_reward
        
        # Extract current game state
        current_x_pos = info.get('x_pos', 0)
        current_score = info.get('score', 0)
        current_coins = info.get('coins', 0)
        current_life = info.get('life', 3)
        current_status = info.get('status','small')
        current_time = info.get('time', 400)

        
        # 1. PROGRESS REWARDS
        # Reward forward movement
        x_progress = current_x_pos - self.prev_x_pos
        if x_progress > 0:
            reward += x_progress * 0.1  # Scale progress reward
            self.stuck_counter = 0
        elif x_progress < 0:
            reward -= 0.05  # Small penalty for moving backward
            self.stuck_counter += 1
        else:
            self.stuck_counter += 1
        
        # Bonus for reaching new furthest point
        if current_x_pos > self.max_x_pos:
            reward += (current_x_pos - self.max_x_pos) * 0.2
            self.max_x_pos = current_x_pos
        
        # 2. SCORE AND COLLECTIBLES
        # Reward score increases (enemy kills, coin collection)
        score_increase = current_score - self.prev_score
        if score_increase > 0:
            reward += score_increase * 0.01  # Scale score reward
        
        # Coin collection bonus
        coin_increase = current_coins - self.prev_coins
        if coin_increase > 0:
            reward += coin_increase * 10  # Coins are valuable
        
        # 3. POWER-UP Status REWARD
        status_reward = self.calculate_status_reward(current_status,self.prev_status)
        reward += status_reward

        # 3. SURVIVAL AND COMPLETION
        # Death penalty
        if current_life < self.prev_life:
            reward -= 50  # Significant death penalty
        
        # Time bonus (encourage faster completion)
        """
        if current_time > 350:  # Plenty of time left
            reward += 0.1
        elif current_time < 100:  # Running out of time
            reward -= 0.2
        """
        # 4. BEHAVIORAL SHAPING
        # Penalty for being stuck (not making progress)
        if self.stuck_counter > 30:  # Stuck for too long
            reward -= 1.0
            self.stuck_counter = 0  # Reset counter
        
        # 5. LEVEL COMPLETION BONUS
        # Check for level completion (flag get)
        if info.get('flag_get', False):
            reward += 1000  # Huge bonus for completing level
        
        # 6. SURVIVAL BONUS
        if not done:
            reward += 0.5  # Small bonus for staying alive
        
        # Update tracking variables
        self.prev_x_pos = current_x_pos
        self.prev_score = current_score
        self.prev_coins = current_coins
        self.prev_life = current_life
        
        return reward

    def calculate_status_reward(self, current_status, prev_status):
        """
        Calculate reward based on Mario's power-up status changes
        Encourages reckless behavior by only rewarding power-ups, no penalties for losing them
        
        Args:
            current_status: Current Mario status ('small', 'tall', 'fireball')
            prev_status: Previous Mario status
            
        Returns:
            float: Status change reward
        """
        # Define status hierarchy and rewards
        status_hierarchy = {
            'small': 0,     # Base status
            'tall': 1,      # Super Mario (mushroom)
            'fireball': 2   # Fire Mario (fire flower)
        }
        
        # Get numeric values for comparison
        current_level = status_hierarchy.get(current_status, 0)
        prev_level = status_hierarchy.get(prev_status, 0)
        
        # Calculate reward based on status change
        if current_level > prev_level:
            # Power-up gained! Big rewards to encourage seeking them
            if current_status == 'tall':
                return 50  # Mushroom power-up reward
            elif current_status == 'fireball':
                return 100  # Fire flower power-up reward (even better!)
        elif current_level < prev_level:
            # Power-up lost - NO PENALTY to encourage reckless play!
            # The agent should feel free to take risks without punishment
            return 0  # No penalty for losing power-ups
        
        # No status change
        return 0


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
    
    def get_reward_info(self):
        """Get current reward tracking information for debugging"""
        return {
            'x_pos': self.prev_x_pos,
            'max_x_pos': self.max_x_pos,
            'score': self.prev_score,
            'coins': self.prev_coins,
            'life': self.prev_life,
            'stuck_counter': self.stuck_counter
        }
    
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

# Example usage with different reward configurations
def create_mario_env(reward_type="shaped"):
    """
    Factory function to create Mario environment with different reward strategies
    
    Args:
        reward_type: "raw", "shaped", or "sparse"
    """
    if reward_type == "raw":
        return CUSTOMMarioEnvironmentRL(reward_shaping=False)
    elif reward_type == "shaped":
        return CUSTOMMarioEnvironmentRL(reward_shaping=True)
    elif reward_type == "sparse":
        # Could implement sparse rewards (only at level completion)
        env = CUSTOMMarioEnvironmentRL(reward_shaping=True)
        # Override reward function for sparse rewards
        return env
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

# Reward comparison utility
def compare_rewards(env, num_steps=100):
    """Compare raw vs shaped rewards"""
    env_raw = CUSTOMMarioEnvironmentRL(reward_shaping=False)
    env_shaped = CUSTOMMarioEnvironmentRL(reward_shaping=True)
    
    raw_rewards = []
    shaped_rewards = []
    
    for env_instance, reward_list in [(env_raw, raw_rewards), (env_shaped, shaped_rewards)]:
        env_instance.reset()
        for _ in range(num_steps):
            action = env_instance.action_space.sample()
            _, reward, done, _, _ = env_instance.step(action)
            reward_list.append(reward)
            if done:
                break
        env_instance.close()
    
    print(f"Raw rewards - Mean: {np.mean(raw_rewards):.2f}, Std: {np.std(raw_rewards):.2f}")
    print(f"Shaped rewards - Mean: {np.mean(shaped_rewards):.2f}, Std: {np.std(shaped_rewards):.2f}")
    
    return raw_rewards, shaped_rewards