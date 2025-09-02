import os
import numpy as np
import datetime
from tqdm import tqdm
from collections import deque
from environnement import MarioEnvironmentRL
from custom_environnement import CUSTOMMarioEnvironmentRL, MarioV2Environment
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
        custom_env = True,
        moveset = "balanced",
        max_steps = 2000,
        advanced_pbar = True,
        test_every = 1,
        test_episodes = 3,  # Number of episodes to test each time
        test_save_gifs = True,  # Enable GIF saving for testing by default
        verbose = True,
        *args,**kwargs):  # Larger buffer
    """
    Train Mario using RL-optimized environment with improved training strategy and TQDM progress bars
    
    Args:
        model_name: Type of RL model to use
        episodes: Number of training episodes
        save_every: Save model every N episodes
        frame_stack: Number of frames to stack for temporal info
        frame_skip: Number of frames to skip (action repeat)
        learning_rate: Learning rate for optimizer
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate per episode
        batch_size: Training batch size
        max_steps: Maximum steps per episode
        replay_frequency: Train every N steps
        target_update_frequency: Update target network every N episodes
        buffer_size: Replay buffer size
        test_every: Test the model every M episodes (0 to disable)
        test_episodes: Number of episodes to run during testing
        test_save_gifs: Whether to save GIFs during periodic testing
    """
    if custom_env:
       # Create RL-optimized environment
        env = MarioV2Environment(
        resize_shape=state_shape,
        frame_stack=frame_stack,
        frame_skip=frame_skip,
        reward_shaping=True,
        action_set= moveset,
        verbose=verbose
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
    test_history = []  # Store test results over time
    
    # Create save directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('gameplay_gifs', exist_ok=True)
    if test_every > 0:
        os.makedirs('test_results', exist_ok=True)
    
    print(f"\n🎮 Starting training with:")
    print(f"   • Frame Stack: {frame_stack}")
    print(f"   • Frame Skip: {frame_skip}")
    print(f"   • State Shape: {stacked_state_shape}")
    print(f"   • Action Space: {n_actions}")
    print(f"   • Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
    print(f"   • Buffer Size: {buffer_size}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Replay Frequency: Every {replay_frequency} steps")
    print(f"   • Using Env: {type(env).__name__}")
    print(f"   • Using Move set: {moveset}")
    if test_every > 0:
        print(f"   • Testing: Every {test_every} episodes ({test_episodes} test episodes)")
    print()
    
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
    
    def run_test_evaluation(episode_num, agent):
        """Run test evaluation and return results"""
        try:
            # Save current model temporarily for testing
            temp_model_path = f'checkpoints/temp_test_model_{model_name}_{episode}.pth'
            agent.save(temp_model_path)
            
            # Determine if we should save GIFs this test cycle
            should_save_test_gifs = test_save_gifs
            
            # Run test with no exploration
            test_results = test_mario_integrated(
                model_path=temp_model_path,
                model_name=model_name,
                episodes=test_episodes,
                render_all=False,
                save_gifs=should_save_test_gifs,
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                state_shape=state_shape,
                custom_env=custom_env,
                moveset=moveset,
                advanced_display=False,  # Keep it quiet during training
                max_steps_per_episode=2000, 
                episode_num=episode_num
            )
            
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            return test_results
            
        except Exception as e:
            tqdm.write(f"⚠️ Test evaluation failed: {e}")
            return None
    
    try:
        for episode in episode_pbar:
            episode_start_time = time.time()
            
            # Reset environment
            stacked_state, raw_state = env.reset()
            total_reward = 0
            steps = 0
            episode_losses = []
            
            
            # Create inner progress bar for steps within episode (optional, for long episodes)
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
                
                # No frame saving during training
                
                # Update state
                stacked_state = next_stacked_state
                total_reward += reward
                steps += 1
                

                debug_info = env.get_debug_info()
                zone_tracker = debug_info['zone_tracker']
                game_state = debug_info['game_state']

                # Update step progress bar
                step_pbar.update(1)

                if advanced_pbar:
                    step_pbar.set_postfix({
                    'Score': f'{total_reward:.0f}',
                    'X_pos': f"{info.get('x_pos', 0)}",
                    'Y_pos': f"{info.get('y_pos', 0)}",
                    'Current_reward': reward,
                    'Steps': steps,
                    'Zone': zone_tracker['current_zone'][:8] if zone_tracker['current_zone'] else 'None',  # Truncated
                    'Stuck': zone_tracker['stuck_counter'],
                    'Max_X': game_state['max_x_pos']
                })
                else:
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
            
            # Prepare postfix data
            postfix_data = {
                'Score': f'{total_reward:.0f}',
                'Avg': f'{avg_score:.1f}',
                'Best': f'{best_avg_score:.1f}',
                'ε': f'{agent.epsilon:.3f}',
                'Loss': f'{avg_loss:.4f}',
                'Steps': steps,
                'Time': f'{episode_time:.1f}s',
                'ETA': f'{eta_minutes:.1f}m',
                'Buffer': len(agent.replay_buffer),
                'Patience': f'{patience_counter}/{patience}',
                'Best_X': f'{game_state["max_x_pos"]}'
            }
            
            # Run periodic testing
            if test_every > 0 and (episode + 1) % test_every == 0 and episode > 0:
                tqdm.write(f"🧪 Running test evaluation at episode {episode+1}...")
                
                # Temporarily close the main progress bar to avoid interference
                episode_pbar.close()
                
                # Run test evaluation
                test_results = run_test_evaluation(episode + 1, agent, {
                    'frame_stack': frame_stack,
                    'frame_skip': frame_skip,
                    'state_shape': state_shape,
                    'custom_env': custom_env,
                    'moveset': moveset
                })
                
                if test_results:
                    # Store test results
                    test_entry = {
                        'episode': episode + 1,
                        'training_avg_score': avg_score,
                        'test_results': test_results['statistics']
                    }
                    test_history.append(test_entry)
                    
                    # Log test results
                    test_stats = test_results['statistics']
                    tqdm.write(f"🏆 Test Results (Episode {episode+1}):")
                    tqdm.write(f"   • Test Avg Score: {test_stats['average_score']:.1f}")
                    tqdm.write(f"   • Completion Rate: {test_stats['completion_rate']:.1f}%")
                    tqdm.write(f"   • Best Test Score: {test_stats['best_score']:.0f}")
                    tqdm.write(f"   • Avg X Position: {test_stats['average_x_position']:.0f}")
                    
                    # Add test info to progress bar postfix
                    postfix_data.update({
                        'TestAvg': f"{test_stats['average_score']:.1f}",
                        'Complete%': f"{test_stats['completion_rate']:.0f}%"
                    })
                    
                # Recreate the progress bar
                episode_pbar = tqdm(
                    range(episode + 1, episodes),
                    initial=0,
                    desc="🍄 Training Mario",
                    unit="episode", 
                    colour="green",
                    dynamic_ncols=True,
                    leave=True
                )
            
            # Update main progress bar with rich information
            episode_pbar.set_postfix(postfix_data)
            save_frames_this_episode = True
            # Periodic detailed logging (less frequent to avoid spam)
            if episode % 10 == 0 or save_frames_this_episode:
                log_msg = (
                    f"📊 Episode {episode+1}: Score={total_reward:.0f}, "
                    f"Avg={avg_score:.1f}, Best={best_avg_score:.1f}, "
                    f"Epsilon={agent.epsilon:.3f}, X_pos={info.get('x_pos', 0)}, "
                    f"Buffer={len(agent.replay_buffer)}"
                )
                # Add test info if available
                if test_history and test_history[-1]['episode'] == episode + 1:
                    last_test = test_history[-1]['test_results']
                    log_msg += f", TestAvg={last_test['average_score']:.1f}"
                
                tqdm.write(log_msg)
            
            # GIF saving now happens during testing phases

            
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
        
        # Final test evaluation if testing is enabled
        if test_every > 0:
            tqdm.write(f"\n🧪 Running final test evaluation...")
            final_test_results = run_test_evaluation(episode + 1, agent, {
                'frame_stack': frame_stack,
                'frame_skip': frame_skip,
                'state_shape': state_shape,
                'custom_env': custom_env,
                'moveset': moveset
            })
            
            if final_test_results:
                test_entry = {
                    'episode': episode + 1,
                    'training_avg_score': np.mean(moving_avg) if moving_avg else 0,
                    'test_results': final_test_results['statistics'],
                    'final_test': True
                }
                test_history.append(test_entry)
        
        # Training completed
        total_time = time.time() - training_start_time
        tqdm.write(f"\n🏁 Training completed!")
        tqdm.write(f"   • Episodes: {episode + 1}")
        tqdm.write(f"   • Total time: {total_time/60:.1f} minutes")
        tqdm.write(f"   • Avg time per episode: {total_time/(episode+1):.1f} seconds")
        tqdm.write(f"   • Best average score: {best_avg_score:.2f}")
        tqdm.write(f"   • Final epsilon: {agent.epsilon:.4f}")
        tqdm.write(f"   • Buffer size: {len(agent.replay_buffer)}")
        
        # Print test summary if available
        if test_history:
            tqdm.write(f"\n🧪 Test Summary:")
            tqdm.write(f"   • Total tests run: {len(test_history)}")
            
            # Get best test performance
            best_test = max(test_history, key=lambda x: x['test_results']['completion_rate'])
            tqdm.write(f"   • Best test completion rate: {best_test['test_results']['completion_rate']:.1f}% (Episode {best_test['episode']})")
            
            if len(test_history) > 1:
                final_test = test_history[-1]['test_results']
                first_test = test_history[0]['test_results'] 
                improvement = final_test['completion_rate'] - first_test['completion_rate']
                tqdm.write(f"   • Improvement: {improvement:+.1f}% completion rate")
        
        # Save final model and plots
        agent.save(f'checkpoints/final_model_{model_name}.pth')
        #save_training_plots(scores, losses, epsilon_history, episode_lengths, episode, final=True)
        
        env.close()
    
    return agent, scores, {
        'losses': losses,
        'epsilon_history': epsilon_history,
        'episode_lengths': episode_lengths,
        'best_avg_score': best_avg_score,
        'total_time_minutes': total_time/60 if 'total_time' in locals() else 0,
        'test_history': test_history  # Include test results in return
    }


def test_mario_integrated(
        model_path=None,
        model_name="DQN",
        episodes=10,
        render_all=True,
        save_gifs=True,
        frame_stack=4,
        frame_skip=2,
        state_shape=(64, 64),
        custom_env=True,
        moveset="balanced",
        advanced_display=True,
        epsilon_override=0.0,  # Override epsilon for testing
        max_steps_per_episode=2000,
        episode_num=None,  # Training episode number for GIF naming,
        verbose = True,
        *args, **kwargs):
    """
    Integrated test function optimized for use during training
    Now includes frame capture and GIF saving functionality
    """
    
    # Create the same environment as training
    if custom_env:
        env = MarioV2Environment(
            resize_shape=state_shape,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            reward_shaping=True,
            action_set=moveset,
            verbose=verbose,
        )
    else:
        env = MarioEnvironmentRL(
            resize_shape=state_shape,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
        )
    
    # Create agent with same architecture
    n_actions = env.action_space.n
    stacked_state_shape = (frame_stack, state_shape[0], state_shape[1])
    
    # Load the trained model
    agent = load_model(
        model_name=model_name,
        state_shape=stacked_state_shape,
        n_actions=n_actions,
        learning_rate=1e-4,  # Not used in testing
        epsilon_start=epsilon_override,
        epsilon_end=epsilon_override,
        epsilon_decay=1.0,  # No decay during testing
        batch_size=32,  # Not used in testing
        buffer_size=100  # Minimal buffer for testing
    )
    
    # Load model weights
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    else:
        raise ValueError(f"Model path {model_path} does not exist")
    
    # Set agent to testing mode
    agent.epsilon = epsilon_override
    
    # Test metrics
    test_scores = []
    test_episode_lengths = []
    completion_rates = []
    max_x_positions = []
    level_completions = 0
    
    # Create directories for test GIFs
    if save_gifs:
        os.makedirs('test_gifs', exist_ok=True)
    
    # Run test episodes
    for episode in range(episodes):
        # Reset environment
        stacked_state, raw_state = env.reset()
        total_reward = 0
        steps = 0
        
        # Frame capture setup for GIF saving
        frames = []
        should_capture_frames = save_gifs and (render_all or episode == 0)  # Save first episode or all
        
        if should_capture_frames:
            print(f"📹 Recording test episode {episode+1} for GIF...")
            if raw_state is not None and len(raw_state.shape) == 3 and raw_state.shape[2] == 3:
                frames.append(raw_state.copy())
        
        # Episode loop
        while steps < max_steps_per_episode:
            # Select action (no exploration)
            action = agent.act(stacked_state, training=False)
            
            # Step environment
            next_stacked_state, reward, done, info, raw_next_state = env.step(action)
            
            # Capture frames for GIF (every 3 steps to keep GIF manageable)
            if should_capture_frames and steps % 3 == 0:
                if raw_next_state is not None and len(raw_next_state.shape) == 3 and raw_next_state.shape[2] == 3:
                    frames.append(raw_next_state.copy())
            
            # Update state and metrics
            stacked_state = next_stacked_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Save GIF for this test episode
        if should_capture_frames and frames:
            # Create descriptive filename
            level_completed = info.get('flag_get', False) or info.get('x_pos', 0) > 3000
            completion_status = "completed" if level_completed else "failed"
            x_pos = info.get('x_pos', 0)
            
            if episode_num is not None:  # Called during training
                gif_filename = f'{SAVING_FOLDER}/training_ep{episode_num}_test{episode+1}_{completion_status}_score{total_reward:.0f}_x{x_pos}.gif'
            else:  # Standalone test
                gif_filename = f'{SAVING_FOLDER}/test_ep{episode+1}_{completion_status}_score{total_reward:.0f}_x{x_pos}.gif'
            
            print(f"💾 Saving test GIF: {gif_filename}")
            
            # Save GIF using your existing function (assuming it exists)
            try:
                save_frames_as_gif(frames = frames, 
                                   episode = episode+1, 
                                   gif_filename= gif_filename,
                                   saving_folder=SAVING_FOLDER)
            except NameError:
                import imageio
                imageio.mimsave(gif_filename, frames, fps=10)
        
        # Collect metrics
        test_scores.append(total_reward)
        test_episode_lengths.append(steps)
        
        # Check if level was completed
        level_completed = info.get('flag_get', False) or info.get('x_pos', 0) > 3000
        if level_completed:
            level_completions += 1
        
        completion_rates.append(1.0 if level_completed else 0.0)
        max_x_positions.append(info.get('x_pos', 0))
    
    # Compile results
    results = {
        'scores': test_scores,
        'episode_lengths': test_episode_lengths,
        'max_x_positions': max_x_positions,
        'completion_rates': completion_rates,
        'statistics': {
            'episodes_tested': len(test_scores),
            'levels_completed': level_completions,
            'completion_rate': np.mean(completion_rates) * 100 if completion_rates else 0,
            'average_score': np.mean(test_scores) if test_scores else 0,
            'best_score': np.max(test_scores) if test_scores else 0,
            'worst_score': np.min(test_scores) if test_scores else 0,
            'score_std': np.std(test_scores) if test_scores else 0,
            'average_episode_length': np.mean(test_episode_lengths) if test_episode_lengths else 0,
            'average_x_position': np.mean(max_x_positions) if max_x_positions else 0,
            'best_x_position': np.max(max_x_positions) if max_x_positions else 0,
        }
    }
    
    env.close()
    return results


if __name__ == "__main__":
    date = datetime.datetime.now()
    now = f"{date.day}d_{date.hour}h_{date.minute}m"

    SAVING_FOLDER = "gameplay_gifs/" + now
    STATE_SHAPE = (84,84)
    EPISODES = 100
    MAX_STEPS = 5000
    TEST_EVERY = 1
    FRAMES_SKIP = 1
    BUFFER_SIZE = 20000
    BATCH_SIZE = 128
    REPLAY_FREQUENCY = 1
    FRAME_STACK = 5
    EPSILON_DECAY = 0.995
    SAVE_EVERY = 100
    LEARNING_RATE = 2e-3
    CUSTOM_ENV = True
    MOVESET = "balanced"
    MAX_STEPS = 5000

    agent, scores, metrics = train_mario(
        model_name="ResNETv1",
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        frame_stack= FRAME_STACK,
        frame_skip=FRAMES_SKIP,        # Better reactivity
        epsilon_decay=0.995, # Slower decay per episode
        batch_size=BATCH_SIZE,       # Larger batches for stability
        buffer_size=BUFFER_SIZE,   # More diverse experiences
        replay_frequency=REPLAY_FREQUENCY,   # Train every 4 steps
        save_every=SAVE_EVERY,
        custom_env= CUSTOM_ENV,
        state_shape = STATE_SHAPE,
        moveset = MOVESET,
        test_every = TEST_EVERY
    )
    print(f"🎯 Training completed with best score: {metrics['best_avg_score']:.2f}")