import os
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not available, using PIL for image processing")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting disabled")


def save_frames_as_gif(frames,
            episode,
            gif_filename,
            fps=30,
            saving_folder = "gameplay_gifs/",
            frame_skip=1,
            max_frames=None,
            ):
    """Save frames as a GIF for visualization

    Args:
        frames: List of frames to save
        episode: Episode number
        gif_filename: Output filename
        fps: Frames per second for the GIF
        saving_folder: Directory to save the GIF
        frame_skip: Skip every N frames (1 = no skipping, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to include (None = no limit)
    """
    os.makedirs(saving_folder, exist_ok=True)

    if not frames:
        print(f"No frames to save for episode {episode}")
        return

    # Apply frame skipping and max frames limit
    selected_frames = frames[::frame_skip]
    if max_frames and len(selected_frames) > max_frames:
        # Take evenly distributed frames if we exceed max_frames
        step = len(selected_frames) // max_frames
        selected_frames = selected_frames[::step][:max_frames]

    print(f"Saving GIF with {len(selected_frames)} frames (from {len(frames)} total frames)")

    # Convert frames to PIL Images
    pil_frames = []
    for i, frame in enumerate(selected_frames):
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Ensure frame is in correct format (RGB)
        if len(frame.shape) == 2:  # If grayscale
            if HAS_CV2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = np.stack([frame] * 3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # If RGBA
            if HAS_CV2:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                frame = frame[:, :, :3]  # Drop alpha channel

        # Resize frame for smaller file size
        if HAS_CV2:
            resized = cv2.resize(frame, (256, 240))
        else:
            pil_frame = Image.fromarray(frame)
            resized = np.array(pil_frame.resize((256, 240)))
        pil_frames.append(Image.fromarray(resized))
    
    # Save as GIF
    if pil_frames:
        pil_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000//fps,
            loop=0
        )
        print(f"Saved gameplay GIF: {gif_filename}")


def save_episode_as_gif(frames,
                       episode,
                       fps=15,
                       saving_folder="test_gifs/",
                       quality='medium'):
    """Save an entire episode as a GIF with optimized settings

    Args:
        frames: List of frames from the entire episode
        episode: Episode number for filename
        fps: Frames per second for the GIF
        saving_folder: Directory to save the GIF
        quality: 'low', 'medium', or 'high' - affects frame skip and resolution
    """
    quality_settings = {
        'low': {'frame_skip': 4, 'max_frames': 200, 'size': (128, 120)},
        'medium': {'frame_skip': 2, 'max_frames': 400, 'size': (192, 180)},
        'high': {'frame_skip': 1, 'max_frames': 800, 'size': (256, 240)}
    }

    settings = quality_settings.get(quality, quality_settings['medium'])

    gif_filename = os.path.join(saving_folder, f"episode_{episode}_full.gif")

    # Process frames directly here instead of calling save_frames_as_gif
    os.makedirs(saving_folder, exist_ok=True)

    if not frames:
        print(f"No frames to save for episode {episode}")
        return

    # Apply frame selection
    selected_frames = frames[::settings['frame_skip']]
    if settings['max_frames'] and len(selected_frames) > settings['max_frames']:
        step = len(selected_frames) // settings['max_frames']
        selected_frames = selected_frames[::step][:settings['max_frames']]

    print(f"Creating full episode GIF with {len(selected_frames)} frames (from {len(frames)} total)")

    # Process frames with custom resolution
    pil_frames = []
    for frame in selected_frames:
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Ensure RGB format
        if len(frame.shape) == 2:
            if HAS_CV2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = np.stack([frame] * 3, axis=-1)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            if HAS_CV2:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                frame = frame[:, :, :3]

        # Resize to custom resolution
        if HAS_CV2:
            resized = cv2.resize(frame, settings['size'])
        else:
            pil_frame = Image.fromarray(frame)
            resized = np.array(pil_frame.resize(settings['size']))
        pil_frames.append(Image.fromarray(resized))

    # Save as GIF with optimized settings
    if pil_frames:
        pil_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000//fps,
            loop=0,
            optimize=True  # Enable optimization for smaller file size
        )
        print(f"Full episode GIF saved: {gif_filename} ({quality} quality)")


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
    