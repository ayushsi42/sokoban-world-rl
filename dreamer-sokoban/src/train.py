"""
Training script for DreamerV3-Sokoban with curriculum learning.
Includes logging, checkpointing, and evaluation.
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra import compose, initialize_config_dir

from dreamer.agent import DreamerV3Agent
from environments.sokoban_wrapper import create_sokoban_env, CurriculumSokobanWrapper
from analysis.evaluation import evaluate_agent
from analysis.visualization import visualize_trajectory


class Trainer:
    """Main training class for DreamerV3-Sokoban."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize trainer.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Create environment
        self.env = self._create_environment()
        
        # Get observation and action spaces
        obs_shape = self.env.observation_space.shape
        action_size = self.env.action_space.n
        
        # Create agent
        self.agent = DreamerV3Agent(
            observation_shape=obs_shape,
            action_size=action_size,
            config=config.agent,
            device=self.device,
        )
        
        # Initialize logging
        self._init_logging()
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.global_step = 0
        self.episode_count = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def _create_environment(self) -> gym.Env:
        """Create training environment."""
        env_config = self.config.environment
        
        if env_config.use_curriculum:
            env = create_sokoban_env(
                env_type="curriculum",
                initial_stage=env_config.initial_stage,
                observation_size=tuple(env_config.observation_size),
                reward_shaping=env_config.reward_shaping,
            )
        else:
            env = create_sokoban_env(
                env_type="standard",
                dim_room=tuple(env_config.dim_room),
                num_boxes=env_config.num_boxes,
                max_steps=env_config.max_steps,
                observation_size=tuple(env_config.observation_size),
                reward_shaping=env_config.reward_shaping,
            )
        
        return env
    
    def _init_logging(self):
        """Initialize logging (WandB or TensorBoard)."""
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.project,
                name=self.config.run_name,
                config=OmegaConf.to_container(self.config, resolve=True),
                save_code=True,
            )
        
        # Create TensorBoard writer
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(self.config.logging.log_dir) / self.config.run_name
        self.writer = SummaryWriter(log_dir)
        
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.total_steps} steps...")
        
        # Training loop
        episode_data = self._reset_episode()
        obs, info = self.env.reset()
        state = None
        
        pbar = tqdm(total=self.config.training.total_steps, desc="Training")
        
        while self.global_step < self.config.training.total_steps:
            # Select action
            action, state = self.agent.act(obs, state, training=True)
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_data["observations"].append(obs)
            episode_data["actions"].append(
                np.eye(self.agent.action_size)[action]  # One-hot encode
            )
            episode_data["rewards"].append(reward)
            episode_data["terminals"].append(terminated)
            episode_data["total_reward"] += reward
            
            # Update observation
            obs = next_obs
            self.global_step += 1
            pbar.update(1)
            
            # Episode ended
            if done:
                # Add episode to replay buffer
                self._finalize_episode(episode_data)
                self.agent.replay_buffer.add_episode(episode_data)
                
                # Log episode statistics
                self._log_episode(episode_data, info)
                
                # Reset episode
                episode_data = self._reset_episode()
                obs, info = self.env.reset()
                state = None
                self.episode_count += 1
            
            # Train agent
            if (self.global_step >= self.config.training.start_training_after and
                self.global_step % self.config.training.train_every == 0):
                
                for _ in range(self.config.training.train_steps):
                    metrics = self.agent.train()
                    
                    if metrics:
                        self._log_metrics(metrics, prefix="train")
            
            # Evaluate
            if self.global_step % self.config.evaluation.eval_every == 0:
                self._evaluate()
            
            # Save checkpoint
            if self.global_step % self.config.checkpoint.save_every == 0:
                self._save_checkpoint()
            
            # Visualize trajectory
            if (self.config.visualization.enabled and
                self.global_step % self.config.visualization.visualize_every == 0):
                self._visualize_imagination()
        
        pbar.close()
        print("Training completed!")
        
        # Final evaluation
        self._evaluate()
        self._save_checkpoint(final=True)
        
        # Close logging
        self.writer.close()
        if self.config.logging.use_wandb:
            wandb.finish()
    
    def _reset_episode(self) -> Dict[str, List]:
        """Reset episode data storage."""
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
            "total_reward": 0.0,
        }
    
    def _finalize_episode(self, episode_data: Dict[str, List]):
        """Convert episode data to numpy arrays."""
        for key in ["observations", "actions", "rewards", "terminals"]:
            episode_data[key] = np.array(episode_data[key])
    
    def _log_episode(self, episode_data: Dict, info: Dict):
        """Log episode statistics."""
        episode_reward = episode_data["total_reward"]
        episode_length = len(episode_data["rewards"])
        success = info.get("solved", False)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Log to TensorBoard
        self.writer.add_scalar("episode/reward", episode_reward, self.episode_count)
        self.writer.add_scalar("episode/length", episode_length, self.episode_count)
        self.writer.add_scalar("episode/success", float(success), self.episode_count)
        
        # Log curriculum stage if applicable
        if "curriculum_stage" in info:
            self.writer.add_scalar(
                "episode/curriculum_stage",
                info["curriculum_stage"],
                self.episode_count
            )
        
        # Log to WandB
        if self.config.logging.use_wandb:
            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": episode_length,
                "episode/success": float(success),
                "episode/count": self.episode_count,
                "global_step": self.global_step,
            })
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log training metrics."""
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(full_key, value, self.global_step)
        
        if self.config.logging.use_wandb:
            wandb_metrics = {
                f"{prefix}/{k}" if prefix else k: v
                for k, v in metrics.items()
            }
            wandb_metrics["global_step"] = self.global_step
            wandb.log(wandb_metrics)
    
    def _evaluate(self):
        """Evaluate agent performance."""
        print("\nEvaluating agent...")
        
        # Create evaluation environment
        eval_env = self._create_environment()
        
        # Run evaluation
        eval_results = evaluate_agent(
            agent=self.agent,
            env=eval_env,
            num_episodes=self.config.evaluation.num_episodes,
            render=self.config.evaluation.render,
            device=self.device,
        )
        
        # Log evaluation results
        self._log_metrics(eval_results, prefix="eval")
        
        # Update success rate tracking
        self.success_rates.append(eval_results["success_rate"])
        
        print(f"Evaluation results: {eval_results}")
        
        eval_env.close()
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_name = "final_checkpoint.pt" if final else f"checkpoint_{self.global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "success_rates": self.success_rates,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        
        state_path = self.checkpoint_dir / f"training_state_{self.global_step}.json"
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Keep only recent checkpoints
        if not final and self.config.checkpoint.keep_last_n > 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[1]))
        
        if len(checkpoints) > self.config.checkpoint.keep_last_n:
            for checkpoint in checkpoints[:-self.config.checkpoint.keep_last_n]:
                checkpoint.unlink()
                
                # Also remove associated training state
                state_file = checkpoint.parent / f"training_state_{checkpoint.stem.split('_')[1]}.json"
                if state_file.exists():
                    state_file.unlink()
    
    def _visualize_imagination(self):
        """Visualize agent's imagined trajectories."""
        # Get current observation
        obs = self.env.render()
        
        # Generate imagined trajectory
        with torch.no_grad():
            # Convert observation
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Create random action sequence for visualization
            action_sequence = torch.zeros(
                1,
                self.config.visualization.imagination_length,
                self.agent.action_size,
                device=self.device
            )
            
            for t in range(self.config.visualization.imagination_length):
                # Sample random action
                action_idx = np.random.randint(0, self.agent.action_size)
                action_sequence[0, t, action_idx] = 1.0
            
            # Imagine trajectory
            trajectory = self.agent.world_model.imagine_trajectory(
                obs_tensor,
                action_sequence
            )
        
        # Visualize and save
        viz_path = self.checkpoint_dir / "visualizations" / f"imagination_{self.global_step}.gif"
        viz_path.parent.mkdir(exist_ok=True)
        
        visualize_trajectory(
            real_obs=obs,
            imagined_trajectory=trajectory,
            save_path=str(viz_path),
        )
        
        # Log to WandB
        if self.config.logging.use_wandb:
            wandb.log({
                "imagination": wandb.Video(str(viz_path), fps=4, format="gif")
            })


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create trainer
    trainer = Trainer(cfg)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    # For standalone execution without Hydra
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Sokoban")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to configuration file")
    parser.add_argument("--env", type=str, default="Sokoban-v0",
                        help="Environment name")
    parser.add_argument("--steps", type=int, default=1000000,
                        help="Total training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if running with Hydra
    if "_HYDRA_COMPLETE" not in os.environ:
        # Load config manually
        config = OmegaConf.load(args.config)
        
        # Override with command line arguments
        config.environment.env_name = args.env
        config.training.total_steps = args.steps
        config.seed = args.seed
        config.device = args.device
        
        # Create trainer and run
        trainer = Trainer(config)
        trainer.train()
    else:
        # Run with Hydra
        main()
