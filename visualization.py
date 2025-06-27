import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_frames_as_gif(frames, 
            episode, 
            fps=30,
            saving_folder = "gameplay_gifs/"
            ):
    """Save frames as a GIF for visualization"""
    os.makedirs(saving_folder, exist_ok=True)
    
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
            f'{saving_folder}/episode_{episode}.gif',
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000//fps,
            loop=0
        )
        print(f"Saved gameplay GIF: {saving_folder}/episode_{episode}.gif")



def progress_logger(
        episode,
        episodes,
        scores,
        total_reward,
        avg_score,
        agent,
        steps,
        info,
        save_every,
        ):

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
    