"""
Visualization tools for DreamerV3-Sokoban.
Includes trajectory visualization, planning analysis, and debugging tools.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import imageio
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import io


def visualize_trajectory(
    real_obs: Optional[np.ndarray] = None,
    imagined_trajectory: Optional[Dict[str, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    fps: int = 4,
    show_info: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize real vs imagined trajectories.
    
    Args:
        real_obs: Real observation
        imagined_trajectory: Dictionary with imagined observations, rewards, etc.
        save_path: Path to save animation
        fps: Frames per second
        show_info: Whether to show additional information
        
    Returns:
        Array of frames if save_path is None
    """
    frames = []
    
    if imagined_trajectory is not None:
        imagined_obs = imagined_trajectory["observations"].cpu().numpy()
        imagined_rewards = imagined_trajectory.get("rewards", None)
        imagined_terminals = imagined_trajectory.get("terminals", None)
        
        # Handle batch dimension
        if imagined_obs.ndim == 5:  # (B, T, C, H, W)
            imagined_obs = imagined_obs[0]  # Take first batch
            if imagined_rewards is not None:
                imagined_rewards = imagined_rewards[0].cpu().numpy()
            if imagined_terminals is not None:
                imagined_terminals = imagined_terminals[0].cpu().numpy()
        
        # Convert to HWC format
        imagined_obs = imagined_obs.transpose(0, 2, 3, 1)
        
        # Normalize to [0, 255] if needed
        if imagined_obs.max() <= 1.0:
            imagined_obs = (imagined_obs * 255).astype(np.uint8)
        
        for t in range(imagined_obs.shape[0]):
            frame = create_comparison_frame(
                real=real_obs if t == 0 else None,
                imagined=imagined_obs[t],
                step=t,
                reward=imagined_rewards[t] if imagined_rewards is not None else None,
                terminal=imagined_terminals[t] if imagined_terminals is not None else None,
                show_info=show_info,
            )
            frames.append(frame)
    
    if save_path:
        # Save as GIF
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Saved trajectory visualization to {save_path}")
        return None
    else:
        return np.array(frames)


def create_comparison_frame(
    real: Optional[np.ndarray],
    imagined: np.ndarray,
    step: int,
    reward: Optional[float] = None,
    terminal: Optional[float] = None,
    show_info: bool = True,
) -> np.ndarray:
    """Create a single frame comparing real and imagined observations."""
    # Create figure
    if real is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Real observation
        axes[0].imshow(real)
        axes[0].set_title("Real Observation")
        axes[0].axis("off")
        
        # Imagined observation
        axes[1].imshow(imagined)
        axes[1].set_title(f"Imagined (t={step})")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(imagined)
        ax.set_title(f"Imagined (t={step})")
        ax.axis("off")
    
    # Add info text
    if show_info and (reward is not None or terminal is not None):
        info_text = []
        if reward is not None:
            info_text.append(f"Reward: {reward:.3f}")
        if terminal is not None:
            info_text.append(f"Terminal: {terminal:.3f}")
        
        plt.figtext(0.5, 0.02, " | ".join(info_text), ha="center", fontsize=10)
    
    # Convert to array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return frame


def visualize_world_model_predictions(
    agent,
    observations: np.ndarray,
    actions: np.ndarray,
    save_path: str,
) -> None:
    """
    Visualize world model predictions vs ground truth.
    
    Args:
        agent: DreamerV3 agent
        observations: Sequence of observations (T, H, W, C)
        actions: Sequence of actions (T,)
        save_path: Path to save visualization
    """
    device = next(agent.world_model.parameters()).device
    
    # Convert to tensors
    obs_tensor = torch.from_numpy(observations).float().to(device)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
    
    # Convert actions to one-hot
    action_tensor = torch.zeros(1, len(actions), agent.action_size, device=device)
    for t, action in enumerate(actions):
        action_tensor[0, t, action] = 1.0
    
    # Get world model predictions
    with torch.no_grad():
        outputs = agent.world_model(obs_tensor, action_tensor)
    
    # Extract predictions
    reconstructions = outputs["reconstructions"].cpu().numpy()[0]  # (T, C, H, W)
    rewards = outputs["rewards"].cpu().numpy()[0]  # (T,)
    
    # Create visualization
    num_steps = min(10, len(observations))
    fig, axes = plt.subplots(3, num_steps, figsize=(2 * num_steps, 6))
    
    if num_steps == 1:
        axes = axes.reshape(-1, 1)
    
    for t in range(num_steps):
        # Original observation
        axes[0, t].imshow(observations[t])
        axes[0, t].set_title(f"t={t}")
        axes[0, t].axis("off")
        
        # Reconstruction
        recon = reconstructions[t].transpose(1, 2, 0)
        if recon.max() > 1.0:
            recon = recon / 255.0
        axes[1, t].imshow(recon)
        axes[1, t].set_title(f"Recon")
        axes[1, t].axis("off")
        
        # Difference
        diff = np.abs(observations[t] / 255.0 - recon)
        axes[2, t].imshow(diff, cmap="hot")
        axes[2, t].set_title(f"Error")
        axes[2, t].axis("off")
    
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
    axes[2, 0].set_ylabel("Difference", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved world model predictions to {save_path}")


def visualize_attention_maps(
    agent,
    observation: np.ndarray,
    save_path: str,
) -> None:
    """
    Visualize attention maps from the world model encoder.
    
    Args:
        agent: DreamerV3 agent
        observation: Single observation (H, W, C)
        save_path: Path to save visualization
    """
    device = next(agent.world_model.parameters()).device
    
    # Convert to tensor
    obs_tensor = torch.from_numpy(observation).float().to(device)
    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    # Hook to capture intermediate activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations["encoder_features"] = output
    
    # Register hook on encoder
    hook = agent.world_model.encoder.conv_layers[-1].register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = agent.world_model.encoder(obs_tensor)
    
    # Remove hook
    hook.remove()
    
    # Get activations
    features = activations["encoder_features"][0].cpu().numpy()  # (C, H, W)
    
    # Create visualization
    num_channels = min(16, features.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_channels):
        feature_map = features[i]
        
        # Normalize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Resize to original size
        feature_map = cv2.resize(feature_map, (observation.shape[1], observation.shape[0]))
        
        # Overlay on original image
        axes[i].imshow(observation)
        axes[i].imshow(feature_map, alpha=0.5, cmap="jet")
        axes[i].set_title(f"Channel {i}")
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(num_channels, 16):
        axes[i].axis("off")
    
    plt.suptitle("Encoder Feature Maps", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved attention maps to {save_path}")


def visualize_value_landscape(
    agent,
    env,
    grid_size: int = 20,
    save_path: str = None,
) -> None:
    """
    Visualize value function landscape for different game states.
    
    Args:
        agent: DreamerV3 agent
        env: Sokoban environment
        grid_size: Resolution of value grid
        save_path: Path to save visualization
    """
    device = next(agent.world_model.parameters()).device
    
    # Create grid of positions (simplified for 2D visualization)
    values = np.zeros((grid_size, grid_size))
    
    # Sample different game states
    for i in range(grid_size):
        for j in range(grid_size):
            # Reset environment
            obs, _ = env.reset()
            
            # Simulate some random moves to get variety
            num_moves = int((i * grid_size + j) % (env.max_steps // 2))
            for _ in range(num_moves):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    break
            
            # Get value estimate
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().to(device)
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Encode observation
                obs_embed = agent.world_model.encoder(obs_tensor)
                
                # Get initial state
                state = agent.world_model.rssm.initial_state(1, device)
                dummy_action = torch.zeros(1, agent.action_size, device=device)
                state, _ = agent.world_model.rssm.observe(obs_embed, dummy_action, state)
                
                # Get features and value
                features = agent.world_model.rssm.get_features(state)
                value = agent.critic(features).item()
                
                values[i, j] = value
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(values, cmap="RdBu_r", center=0, annot=False, fmt=".2f")
    plt.title("Value Function Landscape", fontsize=16)
    plt.xlabel("State Variation (arbitrary)")
    plt.ylabel("State Variation (arbitrary)")
    
    # Add colorbar label
    cbar = plt.gca().collections[0].colorbar
    cbar.set_label("Estimated Value", rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved value landscape to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_training_summary_plot(
    metrics_log: str,
    save_path: str,
) -> None:
    """
    Create comprehensive training summary plots.
    
    Args:
        metrics_log: Path to metrics log file
        save_path: Path to save plot
    """
    import json
    import pandas as pd
    
    # Load metrics
    with open(metrics_log, "r") as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Episode rewards
    if "episode_reward" in df:
        axes[0, 0].plot(df.index, df["episode_reward"], alpha=0.5)
        axes[0, 0].plot(df.index, df["episode_reward"].rolling(100).mean(), linewidth=2)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Success rate
    if "success_rate" in df:
        axes[0, 1].plot(df.index, df["success_rate"].rolling(100).mean())
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Success Rate")
        axes[0, 1].set_title("Success Rate (Moving Average)")
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Model losses
    loss_cols = [col for col in df.columns if "loss" in col]
    if loss_cols:
        for col in loss_cols[:5]:  # Limit to 5 losses
            axes[0, 2].plot(df.index, df[col], label=col.replace("_", " ").title())
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("Loss")
        axes[0, 2].set_title("Training Losses")
        axes[0, 2].legend()
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Episode length
    if "episode_length" in df:
        axes[1, 0].plot(df.index, df["episode_length"], alpha=0.5)
        axes[1, 0].plot(df.index, df["episode_length"].rolling(100).mean(), linewidth=2)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Length")
        axes[1, 0].set_title("Episode Length")
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Value estimates
    if "critic/values" in df:
        axes[1, 1].plot(df.index, df["critic/values"])
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Average Value")
        axes[1, 1].set_title("Value Function Estimates")
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Curriculum progression
    if "curriculum_stage" in df:
        axes[1, 2].plot(df.index, df["curriculum_stage"])
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("Stage")
        axes[1, 2].set_title("Curriculum Progression")
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle("DreamerV3-Sokoban Training Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved training summary to {save_path}")


def create_sokoban_state_visualization(
    observation: np.ndarray,
    action: Optional[int] = None,
    reward: Optional[float] = None,
    value: Optional[float] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create annotated Sokoban state visualization.
    
    Args:
        observation: Game observation
        action: Action taken (0-4)
        reward: Reward received
        value: Value estimate
        save_path: Optional path to save
        
    Returns:
        Annotated image array
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Display observation
    ax.imshow(observation)
    
    # Add action arrow if provided
    if action is not None:
        action_names = ["Up", "Down", "Left", "Right", "Push"]
        ax.text(0.5, 0.95, f"Action: {action_names[action]}", 
                transform=ax.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add reward and value info
    info_text = []
    if reward is not None:
        info_text.append(f"Reward: {reward:.2f}")
    if value is not None:
        info_text.append(f"Value: {value:.2f}")
    
    if info_text:
        ax.text(0.5, 0.05, " | ".join(info_text),
                transform=ax.transAxes, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax.axis("off")
    
    # Convert to array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    if save_path:
        Image.fromarray(img_array).save(save_path)
    
    return img_array


if __name__ == "__main__":
    print("Visualization module loaded. Use the provided functions to visualize trajectories and training progress.")
