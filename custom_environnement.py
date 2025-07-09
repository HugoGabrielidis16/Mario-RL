import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import cv2
import time
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
            ["right", "A", "B"], # 7: Run right + jump (long jumps)
            ["A", "B"],         # 8: High jump in place (optional)

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
            ["A","B"],
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
        self.prev_status = current_status
        
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





class MarioV2Environment:
    """
    Mario RL Environment V2 - Intelligent Multi-Zone Training System
    
    Features:
    - Adaptive difficulty zone detection and handling
    - Pattern-based reward shaping for specific challenges
    - Intelligent stuck detection with zone-specific thresholds
    - Enhanced temporal action sequence tracking
    - Breakthrough bonus system for zone completion
    - Comprehensive debugging and performance tracking
    """
    
    def __init__(
        self,
        resize_shape=(84, 84),
        frame_stack=5,
        frame_skip=4,
        reward_shaping=True,
        action_set="balanced",
        verbose=False
    ):
        """
        Initialize Mario V2 Environment
        
        Args:
            resize_shape: Frame resize dimensions (height, width)
            frame_stack: Number of frames to stack for temporal awareness
            frame_skip: Frame skipping for action repetition
            reward_shaping: Enable intelligent reward modifications
            action_set: Action set to use (uses existing moveset function)
            verbose: Enable detailed logging
        """
        # Base environment setup
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, self._get_action_set(action_set))
        
        # Environment properties
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.resize_shape = resize_shape
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.reward_shaping = reward_shaping
        self.verbose = verbose
        
        # Frame processing
        self.frames = deque(maxlen=frame_stack)
        
        # Multi-zone difficulty system
        self.difficulty_zones = self._initialize_difficulty_zones()
        
        # Advanced tracking systems
        self.game_state = self._initialize_game_state()
        self.action_tracker = self._initialize_action_tracker()
        self.zone_tracker = self._initialize_zone_tracker()
        self.performance_tracker = self._initialize_performance_tracker()
        
        # Episode statistics
        self.episode_count = 0
        self.total_training_time = 0
        self.start_time = time.time()
        
        if self.verbose:
            self._print_initialization_info()
    
    def _get_action_set(self, action_set_name):
        """Use existing action set function - no changes needed"""
        return get_mario_action_set(action_set_name)
    
    def _initialize_difficulty_zones(self):
        """Define intelligent difficulty zones with specific handling strategies"""
        return {
            'enemy_navigation_594': {
                'x_range': (590, 600),
                'challenge_type': 'enemy_timing',
                'description': 'Goomba/enemy navigation requiring timing',
                'max_stuck_frames': 50,
                'progress_multiplier': 3.0,
                'breakthrough_bonus': 25.0,
                'pattern_rewards': {
                    'enemy_jump_timing': {
                        'sequence': [1, 0, 2],  # right → pause → walk+jump
                        'reward': 4.0,
                        'description': 'Timed jump over enemy'
                    },
                    'running_enemy_clear': {
                        'sequence': [6, 7],     # run → run+jump
                        'reward': 5.0,
                        'description': 'Running jump over enemy'
                    },
                    'positioning_jump': {
                        'sequence': [1, 1, 3], # right → right → jump
                        'reward': 3.0,
                        'description': 'Positioned jump'
                    }
                },
                'exploration_bonus': 1.0,
                'survival_bonus': 0.5
            },
            
            'double_jump_722': {
                'x_range': (720, 730),
                'challenge_type': 'complex_platforming',
                'description': 'Double jump sequence for large gap',
                'max_stuck_frames': 80,
                'progress_multiplier': 6.0,
                'breakthrough_bonus': 50.0,
                'pattern_rewards': {
                    'classic_double_jump': {
                        'sequence': [7, 0, 3],     # run+jump → pause → jump
                        'reward': 15.0,
                        'description': 'Classic double jump sequence'
                    },
                    'extended_double_jump': {
                        'sequence': [7, 0, 0, 3], # run+jump → pause → pause → jump
                        'reward': 18.0,
                        'description': 'Extended timing double jump'
                    },
                    'walk_double_jump': {
                        'sequence': [2, 0, 3],     # walk+jump → pause → jump
                        'reward': 12.0,
                        'description': 'Walk-based double jump'
                    },
                    'precise_double_jump': {
                        'sequence': [6, 7, 0, 3], # run → run+jump → pause → jump
                        'reward': 20.0,
                        'description': 'High-momentum double jump'
                    }
                },
                'exploration_bonus': 2.0,
                'survival_bonus': 0.8
            },
            
            # Expandable for future zones
            'future_zone_template': {
                'x_range': (0, 0),  # Disabled
                'challenge_type': 'custom',
                'description': 'Template for new zones',
                'max_stuck_frames': 40,
                'progress_multiplier': 2.0,
                'breakthrough_bonus': 15.0,
                'pattern_rewards': {},
                'exploration_bonus': 0.5,
                'survival_bonus': 0.3
            }
        }
    
    def _initialize_game_state(self):
        """Initialize game state tracking"""
        return {
            'x_pos': 0,
            'max_x_pos': 0,
            'score': 0,
            'coins': 0,
            'life': 3,
            'status': 'small',
            'time': 400,
            'prev_x_pos': 0,
            'prev_score': 0,
            'prev_coins': 0,
            'prev_life': 3,
            'prev_status': 'small'
        }
    
    def _initialize_action_tracker(self):
        """Initialize action sequence tracking"""
        return {
            'recent_actions': deque(maxlen=20),
            'action_history': [],
            'last_action': 0,
            'action_patterns_detected': [],
            'successful_patterns': {}
        }
    
    def _initialize_zone_tracker(self):
        """Initialize zone-specific tracking"""
        return {
            'current_zone': None,
            'zone_entry_frame': 0,
            'zone_attempts': {},
            'zone_successes': {},
            'zone_best_progress': {},
            'zone_time_spent': {},
            'stuck_counter': 0,
            'last_progress_frame': 0
        }
    
    def _initialize_performance_tracker(self):
        """Initialize performance tracking"""
        return {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_max_x': [],
            'zone_performance': {},
            'breakthrough_history': [],
            'training_metrics': {
                'total_frames': 0,
                'total_rewards': 0,
                'successful_episodes': 0
            }
        }
    
    def reset(self):
        """Reset environment with comprehensive state initialization"""
        # Reset base environment
        state = self.env.reset()
        processed_frame = self._preprocess_frame(state)
        
        # Reset all tracking systems
        self._reset_game_state()
        self._reset_action_tracker()
        self._reset_zone_tracker()
        
        # Initialize frame stack
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        # Episode tracking
        self.episode_count += 1
        
        if self.verbose and self.episode_count % 100 == 0:
            self._print_training_progress()
        
        return self._get_stacked_frames(), state.copy()
    
    def step(self, action):
        """
        Enhanced step function with intelligent reward shaping
        """
        # Store action for pattern analysis
        self.action_tracker['last_action'] = action
        self.action_tracker['recent_actions'].append(action)
        self.action_tracker['action_history'].append(action)
        
        # Execute action with frame skipping
        total_reward = 0
        done = False
        info = {}
        
        for frame_idx in range(self.frame_skip):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            
            # Process frame on last iteration or when done
            if frame_idx == self.frame_skip - 1 or done:
                processed_frame = self._preprocess_frame(next_state)
                self.frames.append(processed_frame)
            
            if done:
                break
        
        # Update game state tracking
        self._update_game_state(info)
        
        # Apply intelligent reward shaping
        if self.reward_shaping:
            shaped_reward = self._calculate_intelligent_reward(total_reward, info, done)
            total_reward = shaped_reward
        
        # Update performance tracking
        self._update_performance_tracking(total_reward, done)
        
        return self._get_stacked_frames(), total_reward, done, info, next_state.copy()
    
    def _calculate_intelligent_reward(self, original_reward, info, done):
        """
        Intelligent multi-zone adaptive reward system
        """
        reward = original_reward
        current_x = self.game_state['x_pos']
        
        # Zone detection and management
        active_zone = self._detect_and_manage_zones(current_x)
        
        # Core progress rewards
        reward += self._calculate_progress_rewards(active_zone)
        
        # Pattern-based rewards
        reward += self._calculate_pattern_rewards(active_zone)
        
        # Zone-specific bonuses
        reward += self._calculate_zone_bonuses(active_zone)
        
        # Stuck detection and penalties
        reward += self._calculate_stuck_penalties(active_zone)
        
        # Game state rewards (score, coins, status)
        reward += self._calculate_game_state_rewards()
        
        # Survival and completion bonuses
        reward += self._calculate_survival_completion_rewards(done, active_zone)
        
        # Breakthrough detection
        reward += self._detect_breakthrough_bonuses(active_zone)
        
        return reward
    
    def _detect_and_manage_zones(self, x_pos):
        """Detect current difficulty zone and manage transitions"""
        active_zone = None
        active_zone_data = None
        
        # Check all zones for current position
        for zone_name, zone_data in self.difficulty_zones.items():
            if zone_data['x_range'][0] <= x_pos <= zone_data['x_range'][1]:
                active_zone = zone_name
                active_zone_data = zone_data
                break
        
        # Handle zone transitions
        if active_zone != self.zone_tracker['current_zone']:
            if active_zone:
                self._enter_zone(active_zone, x_pos)
            else:
                self._exit_zone(x_pos)
        
        # Update zone time tracking
        if active_zone:
            if active_zone not in self.zone_tracker['zone_time_spent']:
                self.zone_tracker['zone_time_spent'][active_zone] = 0
            self.zone_tracker['zone_time_spent'][active_zone] += 1
        
        return active_zone_data
    
    def _calculate_progress_rewards(self, zone_data):
        """Calculate progress-based rewards with zone multipliers"""
        x_progress = self.game_state['x_pos'] - self.game_state['prev_x_pos']
        reward = 0
        
        if x_progress > 0:
            # Base progress reward
            base_reward = x_progress * 0.1
            
            # Apply zone multiplier
            if zone_data:
                base_reward *= zone_data['progress_multiplier']
            
            reward += base_reward
            self.zone_tracker['stuck_counter'] = 0
            self.zone_tracker['last_progress_frame'] = self.performance_tracker['training_metrics']['total_frames']
            
        elif x_progress < 0:
            # Backward movement penalty (reduced in zones)
            penalty = 0.05
            if zone_data:
                penalty *= 0.2  # Reduced penalty in zones for repositioning
            reward -= penalty
            self.zone_tracker['stuck_counter'] += 1
        else:
            # No movement
            self.zone_tracker['stuck_counter'] += 1
        
        # New maximum progress bonus
        if self.game_state['x_pos'] > self.game_state['max_x_pos']:
            progress_bonus = (self.game_state['x_pos'] - self.game_state['max_x_pos']) * 0.3
            if zone_data:
                progress_bonus *= 2.0  # Double bonus in zones
            reward += progress_bonus
            self.game_state['max_x_pos'] = self.game_state['x_pos']
        
        return reward
    
    def _calculate_pattern_rewards(self, zone_data):
        """Calculate rewards for detected action patterns"""
        if not zone_data or len(self.action_tracker['recent_actions']) < 3:
            return 0
        
        reward = 0
        recent_actions = list(self.action_tracker['recent_actions'])
        
        # Check each pattern in the current zone
        for pattern_name, pattern_data in zone_data['pattern_rewards'].items():
            pattern_sequence = pattern_data['sequence']
            pattern_reward = pattern_data['reward']
            
            # Check if recent actions match this pattern
            if self._matches_pattern(recent_actions, pattern_sequence):
                reward += pattern_reward
                
                # Track successful patterns
                if pattern_name not in self.action_tracker['successful_patterns']:
                    self.action_tracker['successful_patterns'][pattern_name] = 0
                self.action_tracker['successful_patterns'][pattern_name] += 1
                
                if self.verbose:
                    print(f"Pattern detected: {pattern_name} (+{pattern_reward})")
        
        return reward
    
    def _calculate_zone_bonuses(self, zone_data):
        """Calculate zone-specific exploration and survival bonuses"""
        if not zone_data:
            return 0
        
        reward = 0
        
        # Exploration bonus for trying different actions
        recent_5_actions = list(self.action_tracker['recent_actions'])[-5:]
        unique_actions = len(set(recent_5_actions))
        if unique_actions >= 3:
            reward += zone_data['exploration_bonus']
        
        # Zone-specific survival bonus
        reward += zone_data['survival_bonus']
        
        return reward
    
    def _calculate_stuck_penalties(self, zone_data):
        """Calculate adaptive stuck penalties based on zone type"""
        if self.zone_tracker['stuck_counter'] == 0:
            return 0
        
        # Determine stuck threshold and penalty
        if zone_data:
            stuck_threshold = zone_data['max_stuck_frames']
            # Graduated penalty system based on challenge type
            if zone_data['challenge_type'] == 'enemy_timing':
                penalty_multiplier = 0.5  # Gentler for timing-based challenges
            elif zone_data['challenge_type'] == 'complex_platforming':
                penalty_multiplier = 0.8  # Moderate for complex sequences
            else:
                penalty_multiplier = 1.0
        else:
            stuck_threshold = 30
            penalty_multiplier = 1.0
        
        if self.zone_tracker['stuck_counter'] > stuck_threshold:
            penalty = 1.0 * penalty_multiplier
            self.zone_tracker['stuck_counter'] = 0
            
            # Track zone attempts
            if zone_data and self.zone_tracker['current_zone']:
                zone_name = self.zone_tracker['current_zone']
                if zone_name not in self.zone_tracker['zone_attempts']:
                    self.zone_tracker['zone_attempts'][zone_name] = 0
                self.zone_tracker['zone_attempts'][zone_name] += 1
            
            return -penalty
        
        return 0
    
    def _calculate_game_state_rewards(self):
        """Calculate rewards for score, coins, and status changes"""
        reward = 0
        
        # Score increase rewards
        score_increase = self.game_state['score'] - self.game_state['prev_score']
        if score_increase > 0:
            reward += score_increase * 0.01
        
        # Coin collection rewards
        coin_increase = self.game_state['coins'] - self.game_state['prev_coins']
        if coin_increase > 0:
            reward += coin_increase * 15  # Increased coin value
        
        # Status (power-up) rewards
        reward += self._calculate_status_rewards()
        
        # Death penalty
        if self.game_state['life'] < self.game_state['prev_life']:
            reward -= 50
        
        return reward
    
    def _calculate_survival_completion_rewards(self, done, zone_data):
        """Calculate survival bonuses and level completion rewards"""
        reward = 0
        
        # Level completion bonus
        if hasattr(self, '_last_info') and self._last_info.get('flag_get', False):
            reward += 1000
        
        # Adaptive survival bonus
        if not done:
            base_survival = 0.1
            if zone_data:
                base_survival = zone_data['survival_bonus']
            
            # Only give survival bonus if making some progress
            if self.zone_tracker['stuck_counter'] < 20:
                reward += base_survival
        
        return reward
    
    def _detect_breakthrough_bonuses(self, zone_data):
        """Detect and reward breakthrough moments"""
        if not zone_data or not self.zone_tracker['current_zone']:
            return 0
        
        current_x = self.game_state['x_pos']
        zone_end = zone_data['x_range'][1]
        
        # Check if we just passed the zone
        if (current_x > zone_end and 
            self.game_state['prev_x_pos'] <= zone_end):
            
            breakthrough_bonus = zone_data['breakthrough_bonus']
            zone_name = self.zone_tracker['current_zone']
            
            # Track successful breakthrough
            if zone_name not in self.zone_tracker['zone_successes']:
                self.zone_tracker['zone_successes'][zone_name] = 0
            self.zone_tracker['zone_successes'][zone_name] += 1
            
            # Record breakthrough
            self.performance_tracker['breakthrough_history'].append({
                'zone': zone_name,
                'episode': self.episode_count,
                'attempts': self.zone_tracker['zone_attempts'].get(zone_name, 0),
                'bonus': breakthrough_bonus
            })
            
            if self.verbose:
                print(f"🎉 BREAKTHROUGH! Passed {zone_name} (+{breakthrough_bonus})")
            
            return breakthrough_bonus
        
        return 0
    
    def _matches_pattern(self, recent_actions, pattern):
        """Check if recent actions match a specific pattern"""
        if len(recent_actions) < len(pattern):
            return False
        
        recent_slice = recent_actions[-len(pattern):]
        return recent_slice == pattern
    
    def _calculate_status_rewards(self):
        """Calculate power-up status change rewards"""
        status_hierarchy = {'small': 0, 'tall': 1, 'fireball': 2}
        
        current_level = status_hierarchy.get(self.game_state['status'], 0)
        prev_level = status_hierarchy.get(self.game_state['prev_status'], 0)
        
        if current_level > prev_level:
            if self.game_state['status'] == 'tall':
                return 75  # Mushroom bonus
            elif self.game_state['status'] == 'fireball':
                return 150  # Fire flower bonus
        
        return 0  # No penalty for losing power-ups
    
    def _update_game_state(self, info):
        """Update game state tracking from environment info"""
        # Store previous state
        self.game_state['prev_x_pos'] = self.game_state['x_pos']
        self.game_state['prev_score'] = self.game_state['score']
        self.game_state['prev_coins'] = self.game_state['coins']
        self.game_state['prev_life'] = self.game_state['life']
        self.game_state['prev_status'] = self.game_state['status']
        
        # Update current state
        self.game_state['x_pos'] = info.get('x_pos', self.game_state['x_pos'])
        self.game_state['score'] = info.get('score', self.game_state['score'])
        self.game_state['coins'] = info.get('coins', self.game_state['coins'])
        self.game_state['life'] = info.get('life', self.game_state['life'])
        self.game_state['status'] = info.get('status', self.game_state['status'])
        self.game_state['time'] = info.get('time', self.game_state['time'])
        
        # Store info for next cycle
        self._last_info = info
    
    def _enter_zone(self, zone_name, x_pos):
        """Handle entering a difficulty zone"""
        self.zone_tracker['current_zone'] = zone_name
        self.zone_tracker['zone_entry_frame'] = self.performance_tracker['training_metrics']['total_frames']
        
        if self.verbose:
            zone_info = self.difficulty_zones[zone_name]
            print(f"🎯 Entering {zone_name} at x={x_pos}: {zone_info['description']}")
    
    def _exit_zone(self, x_pos):
        """Handle exiting a difficulty zone"""
        if self.zone_tracker['current_zone'] and self.verbose:
            print(f"✅ Exiting {self.zone_tracker['current_zone']} at x={x_pos}")
        
        self.zone_tracker['current_zone'] = None
        self.zone_tracker['zone_entry_frame'] = 0
    
    def _update_performance_tracking(self, reward, done):
        """Update comprehensive performance metrics"""
        metrics = self.performance_tracker['training_metrics']
        metrics['total_frames'] += 1
        metrics['total_rewards'] += reward
        
        if done:
            # Episode completed
            episode_length = metrics['total_frames']
            episode_reward = sum(self.performance_tracker['episode_rewards'][-episode_length:]) if self.performance_tracker['episode_rewards'] else reward
            
            self.performance_tracker['episode_lengths'].append(episode_length)
            self.performance_tracker['episode_max_x'].append(self.game_state['max_x_pos'])
            
            if self.game_state['max_x_pos'] > 1000:  # Rough success threshold
                metrics['successful_episodes'] += 1
    
    def _reset_game_state(self):
        """Reset game state tracking"""
        self.game_state.update({
            'x_pos': 0, 'max_x_pos': 0, 'score': 0, 'coins': 0,
            'life': 3, 'status': 'small', 'time': 400,
            'prev_x_pos': 0, 'prev_score': 0, 'prev_coins': 0,
            'prev_life': 3, 'prev_status': 'small'
        })
    
    def _reset_action_tracker(self):
        """Reset action tracking"""
        self.action_tracker['recent_actions'].clear()
        self.action_tracker['last_action'] = 0
        self.action_tracker['action_patterns_detected'].clear()
    
    def _reset_zone_tracker(self):
        """Reset zone tracking for new episode"""
        self.zone_tracker.update({
            'current_zone': None,
            'zone_entry_frame': 0,
            'stuck_counter': 0,
            'last_progress_frame': 0
        })
    
    def _preprocess_frame(self, frame):
        """Optimized frame preprocessing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _get_stacked_frames(self):
        """Get current stacked frames"""
        return np.stack(self.frames, axis=0)
    
    def _print_initialization_info(self):
        """Print environment initialization information"""
        print("=" * 60)
        print("MARIO V2 ENVIRONMENT - INTELLIGENT TRAINING SYSTEM".center(60))
        print("=" * 60)
        print(f"Action Space      : {self.action_space}")
        print(f"Observation Space : {self.observation_space}")
        print(f"Frame Stack       : {self.frame_stack}")
        print(f"Frame Skip        : {self.frame_skip}")
        print(f"Resize Shape      : {self.resize_shape}")
        print(f"Reward Shaping    : {self.reward_shaping}")
        print(f"Difficulty Zones  : {len([z for z in self.difficulty_zones.values() if z['x_range'][0] > 0])}")
        print("=" * 60)
        
        for zone_name, zone_data in self.difficulty_zones.items():
            if zone_data['x_range'][0] > 0:  # Active zones
                print(f"📍 {zone_name}: x={zone_data['x_range']} - {zone_data['description']}")
        print("=" * 60)
    
    def _print_training_progress(self):
        """Print training progress summary"""
        metrics = self.performance_tracker['training_metrics']
        elapsed = time.time() - self.start_time
        
        print(f"\n📊 Training Progress - Episode {self.episode_count}")
        print(f"⏱️  Time: {elapsed/60:.1f} min | Frames: {metrics['total_frames']:,}")
        print(f"🎯 Max Progress: {self.game_state['max_x_pos']}")
        print(f"🏆 Breakthroughs: {len(self.performance_tracker['breakthrough_history'])}")
        
        # Zone performance
        for zone_name, attempts in self.zone_tracker['zone_attempts'].items():
            successes = self.zone_tracker['zone_successes'].get(zone_name, 0)
            success_rate = (successes / attempts * 100) if attempts > 0 else 0
            print(f"🎮 {zone_name}: {successes}/{attempts} ({success_rate:.1f}%)")
    
    # Public interface methods
    def get_debug_info(self):
        """Get comprehensive debugging information"""
        return {
            'game_state': self.game_state.copy(),
            'zone_tracker': self.zone_tracker.copy(),
            'action_tracker': {
                'recent_actions': list(self.action_tracker['recent_actions']),
                'successful_patterns': self.action_tracker['successful_patterns'].copy()
            },
            'performance': self.performance_tracker.copy()
        }
    
    def get_zone_performance(self):
        """Get zone-specific performance metrics"""
        return {
            'attempts': self.zone_tracker['zone_attempts'].copy(),
            'successes': self.zone_tracker['zone_successes'].copy(),
            'time_spent': self.zone_tracker['zone_time_spent'].copy(),
            'breakthroughs': self.performance_tracker['breakthrough_history'].copy()
        }
    
    def close(self):
        """Close environment and print final statistics"""
        if self.verbose:
            print("\n🏁 Training Session Complete!")
            self._print_training_progress()
        
        self.env.close()