import os
import numpy as np
import datetime
from tqdm import tqdm
from collections import deque
from environnement import MarioEnvironmentRL
from custom_environnement import CUSTOMMarioEnvironmentRL
import time

from model.DQN import DQN
from model.ResNET import MultiFrameResNet
from model.agent import Agent
from visualization import save_frames_as_gif, progress_logger

import warnings
warnings.filterwarnings("ignore")


def load_model(
            state_shape,
            n_actions,
            learning_rate,
            epsilon_start,
            epsilon_end,
            batch_size,
            buffer_size,
            epsilon_decay,
            model_name = "DQN",
            *args,
            **kwargs
            ):
    print("Using model: ",model_name)
    if model_name == "DQN":
        model = DQN
    if model_name == "ResNETv1":
        model = MultiFrameResNet
    else:
        raise ValueError("Model name Innapropriate")

    agent = Agent(
        model = model,
        state_shape = state_shape,
        n_actions = n_actions,
        learning_rate=learning_rate,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size
    )
    return agent



def train_mario(
        model_name="DQN",
        episodes=1000, 
        render_every=100, 
        save_every=100,
        frame_stack=4,
        frame_skip=2,  # Reduced from 4 for better reactivity
        learning_rate=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,  # Per episode decay
        batch_size=32,
        replay_frequency=4,  # Train every N steps
        target_update_frequency=10,  # Update target every N episodes
        buffer_size=500,
        state_shape = (64,64),
        custom_env = True):  # Larger buffer
    """
    Train Mario using RL-optimized environment with improved training strategy and TQDM progress bars
    
    Args:
        model_name: Type of RL model to use
        episodes: Number of training episodes
        render_every: Save GIF every N episodes
        save_every: Save model every N episodes
        frame_stack: Number of frames to stack for temporal info
        frame_skip: Number of frames to skip (action repeat)
        learning_rate: Learning rate for optimizer
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate per episode
        batch_size: Training batch size
        replay_frequency: Train every N steps
        target_update_frequency: Update target network every N episodes
        buffer_size: Replay buffer size
    """
    if custom_env:
       # Create RL-optimized environment
        env = CUSTOMMarioEnvironmentRL(
        resize_shape=state_shape,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        reward_shaping=True,
        action_set= "balanced"
        )
    else:
        # Use classical gym reward environment
        env = MarioEnvironmentRL(
        resize_shape=state_shape,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        ) 

    
    # Create agent - state shape now includes frame stack dimension
    n_actions = env.action_space.n
    stacked_state_shape = (frame_stack, state_shape[0], state_shape[1])
    
    agent = load_model(
        model_name=model_name,
        state_shape=stacked_state_shape,
        n_actions=n_actions,
        learning_rate=learning_rate,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size
    )
    
    # Training metrics
    scores = []
    moving_avg = deque(maxlen=100)
    losses = []
    epsilon_history = []
    episode_lengths = []
    
    # Create save directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('gameplay_gifs', exist_ok=True)
    
    print(f"\n🎮 Starting training with:")
    print(f"   • Frame Stack: {frame_stack}")
    print(f"   • Frame Skip: {frame_skip}")
    print(f"   • State Shape: {stacked_state_shape}")
    print(f"   • Action Space: {n_actions}")
    print(f"   • Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
    print(f"   • Buffer Size: {buffer_size}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Render Every: {render_every} ")
    print(f"   • Replay Frequency: Every {replay_frequency} steps")
    
    # Early stopping and adaptive training
    best_avg_score = -float('inf')
    patience_counter = 0
    patience = 50  # Episodes without improvement
    
    # Initialize TQDM progress bar for episodes
    episode_pbar = tqdm(
        range(episodes), 
        desc="🍄 Training Mario",
        unit="episode",
        colour="green",
        dynamic_ncols=True,
        leave=True
    )
    
    # Training start time for ETA calculation
    training_start_time = time.time()
    
    try:
        for episode in episode_pbar:
            episode_start_time = time.time()
            
            # Reset environment
            stacked_state, raw_state = env.reset()
            total_reward = 0
            steps = 0
            episode_losses = []
            
            # Store frames for visualization
            frames = []
            save_frames_this_episode = (episode % render_every == 0 and episode > 0)
            
            # Debug info
            if save_frames_this_episode:
                tqdm.write(f"📹 Episode {episode+1} - Recording GIF (Stacked: {stacked_state.shape}, Raw: {raw_state.shape})")
                if raw_state.shape[2] == 3:  # RGB frame
                    frames.append(raw_state.copy())
            
            # Create inner progress bar for steps within episode (optional, for long episodes)
            max_steps = 2000  # Reasonable maximum for Mario episodes
            step_pbar = tqdm(
                total=max_steps,
                desc=f"🏃 Episode {episode+1}",
                unit="step",
                leave=False,
                colour="blue",
                disable=False  # Set to True if you don't want step-level progress
            )
            
            # Episode loop
            while True:
                # Select action using current policy
                action = agent.act(stacked_state, training=True)
                
                # Step environment
                next_stacked_state, reward, done, info, raw_next_state = env.step(action)
                
                # Store experience
                agent.remember(stacked_state, action, reward, next_stacked_state, done)
                
                # Train agent at specified frequency
                if (len(agent.replay_buffer) >= batch_size and steps % replay_frequency == 0):
                    loss = agent.replay(batch_size)
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Store frames for visualization
                if save_frames_this_episode and steps % 5 == 0:
                    if raw_next_state is not None and raw_next_state.shape[2] == 3:
                        frames.append(raw_next_state.copy())
                
                # Update state
                stacked_state = next_stacked_state
                total_reward += reward
                steps += 1
                
                # Update step progress bar
                step_pbar.update(1)
                step_pbar.set_postfix({
                    'Score': f'{total_reward:.0f}',
                    'X_pos': f"{info.get('x_pos', 0)}",
                    'Y_pos' : f"{info.get('y_pos',0)}",
                    'Current_reward' : reward,
                    #'LIFE' : f"{info.get('life',0)}",
                    'Steps': steps
                })
                
                if done or steps >= max_steps:
                    break
            
            # Close step progress bar
            step_pbar.close()
            
            # Episode completed - decay epsilon once per episode
            if hasattr(agent, 'decay_epsilon'):
                agent.decay_epsilon()
            else:
                # Manual epsilon decay if method doesn't exist
                if agent.epsilon > agent.epsilon_end:
                    agent.epsilon *= agent.epsilon_decay
            
            # Update target network
            if episode % target_update_frequency == 0:
                agent.update_target_network()
            
            # Track metrics
            scores.append(total_reward)
            moving_avg.append(total_reward)
            avg_score = np.mean(moving_avg)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            # Store training history
            epsilon_history.append(agent.epsilon)
            episode_lengths.append(steps)
            losses.append(avg_loss)
            
            # Check for improvement
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                patience_counter = 0
                # Save best model
                if episode > 100:  # Only after some training
                    agent.save(f'checkpoints/best_model_{model_name}.pth')
                    tqdm.write(f"💾 New best model saved! Avg score: {best_avg_score:.2f}")
            else:
                patience_counter += 1
            
            # Adaptive epsilon boost for stuck agents
            if episode > 50 and len(set(scores[-10:])) == 1:  # Same score 10 times
                tqdm.write(f"🔄 Agent stuck! Boosting epsilon from {agent.epsilon:.3f} to 0.3")
                agent.epsilon = max(0.3, agent.epsilon)
            
            # Calculate episode time and ETA
            episode_time = time.time() - episode_start_time
            elapsed_total = time.time() - training_start_time
            avg_episode_time = elapsed_total / (episode + 1)
            eta_seconds = avg_episode_time * (episodes - episode - 1)
            eta_minutes = eta_seconds / 60
            
            # Update main progress bar with rich information
            episode_pbar.set_postfix({
                'Score': f'{total_reward:.0f}',
                'Avg': f'{avg_score:.1f}',
                'Best': f'{best_avg_score:.1f}',
                'ε': f'{agent.epsilon:.3f}',
                'Loss': f'{avg_loss:.4f}',
                'Steps': steps,
                'Time': f'{episode_time:.1f}s',
                'ETA': f'{eta_minutes:.1f}m',
                'Buffer': len(agent.replay_buffer),
                'Patience': f'{patience_counter}/{patience}'
            })
            
            # Periodic detailed logging (less frequent to avoid spam)
            if episode % 10 == 0 or save_frames_this_episode:
                tqdm.write(
                    f"📊 Episode {episode+1}: Score={total_reward:.0f}, "
                    f"Avg={avg_score:.1f}, Best={best_avg_score:.1f}, "
                    f"Epsilon={agent.epsilon:.3f}, X_pos={info.get('x_pos', 0)}, "
                    f"Buffer={len(agent.replay_buffer)}"
                )
            
            # Save gameplay GIF
            if save_frames_this_episode and frames:
                tqdm.write(f"💾 Saving GIF for episode {episode+1}...")
                save_frames_as_gif(frames, episode+1, saving_folder=SAVING_FOLDER)
            
            # Save training plots every 50 episodes
            if episode % 50 == 0 and episode > 0:
                tqdm.write(f"📈 Saving training plots at episode {episode+1}...")
                #save_training_plots(scores, losses, epsilon_history, episode_lengths, episode)
            
            # Save model checkpoints
            if episode % save_every == 0 and episode > 0:
                agent.save(f'checkpoints/episode_{episode+1}_{model_name}.pth')
                tqdm.write(f"💾 Model checkpoint saved at episode {episode+1}")
            
            # Early stopping check
            if patience_counter >= patience:
                tqdm.write(f"🛑 Early stopping: No improvement for {patience} episodes")
                tqdm.write(f"   Best average score: {best_avg_score:.2f}")
                break
            
            # Emergency epsilon reset if completely stuck
            if episode > 100 and agent.epsilon < 0.01 and avg_score < 500:
                tqdm.write(f"🚨 Emergency epsilon reset: Agent stuck with low score")
                agent.epsilon = 0.2
    
    except KeyboardInterrupt:
        tqdm.write("\n⏹️  Training interrupted by user")
    
    finally:
        # Close progress bar
        episode_pbar.close()
        
        # Training completed
        total_time = time.time() - training_start_time
        tqdm.write(f"\n🏁 Training completed!")
        tqdm.write(f"   • Episodes: {episode + 1}")
        tqdm.write(f"   • Total time: {total_time/60:.1f} minutes")
        tqdm.write(f"   • Avg time per episode: {total_time/(episode+1):.1f} seconds")
        tqdm.write(f"   • Best average score: {best_avg_score:.2f}")
        tqdm.write(f"   • Final epsilon: {agent.epsilon:.4f}")
        tqdm.write(f"   • Buffer size: {len(agent.replay_buffer)}")
        
        # Save final model and plots
        agent.save(f'checkpoints/final_model_{model_name}.pth')
        #save_training_plots(scores, losses, epsilon_history, episode_lengths, episode, final=True)
        
        env.close()
    
    return agent, scores, {
        'losses': losses,
        'epsilon_history': epsilon_history,
        'episode_lengths': episode_lengths,
        'best_avg_score': best_avg_score,
        'total_time_minutes': total_time/60 if 'total_time' in locals() else 0
    }

if __name__ == "__main__":

    date = datetime.datetime.now()
    now = f"{date.day}d_{date.hour}h_{date.minute}m"

    SAVING_FOLDER = "gameplay_gifs/" + now
    STATE_SHAPE = (84,84)
    EPISODES = 1000
    RENDER_EVERY = 500
    FRAMES_SKIP = 2
    BUFFER_SIZE = 30000
    BATCH_SIZE = 64
    REPLAY_FREQUENCY = 4
    FRAME_STACK = 4
    EPSILON_DECAY = 0.995
    SAVE_EVERY = 100
    LEARNING_RATE = 3e-4
    CUSTOM_ENV = True

    agent, scores, metrics = train_mario(
        model_name="ResNETv1",
        episodes=EPISODES,
        learning_rate=LEARNING_RATE,
        frame_stack= FRAME_STACK,
        frame_skip=FRAMES_SKIP,        # Better reactivity
        epsilon_decay=0.995, # Slower decay per episode
        batch_size=BATCH_SIZE,       # Larger batches for stability
        render_every= RENDER_EVERY,
        buffer_size=BUFFER_SIZE,   # More diverse experiences
        replay_frequency=REPLAY_FREQUENCY,   # Train every 4 steps
        save_every=SAVE_EVERY,
        custom_env= CUSTOM_ENV,
        state_shape = STATE_SHAPE,
    )
    
    print(f"🎯 Training completed with best score: {metrics['best_avg_score']:.2f}")