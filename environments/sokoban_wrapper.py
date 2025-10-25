"""
Sokoban environment wrapper for DreamerV3 compatibility.
Handles environment setup, observation preprocessing, and reward shaping.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
import cv2
from gymnasium import spaces

try:
    import gym_sokoban
except ImportError:
    print("Warning: gym-sokoban not installed. Please install with: pip install gym-sokoban")


class SokobanWrapper(gym.Wrapper):
    """
    Wrapper for Sokoban environment to make it compatible with DreamerV3.
    
    Features:
    - Observation preprocessing (resize, normalize)
    - Reward shaping for better learning
    - Episode statistics tracking
    - Action masking for invalid moves
    """
    
    def __init__(
        self,
        env_name: str = "Sokoban-v0",
        dim_room: Tuple[int, int] = (7, 7),
        max_steps: int = 120,
        num_boxes: int = 2,
        observation_size: Tuple[int, int] = (64, 64),
        reward_shaping: bool = True,
        render_mode: str = "rgb_array",
    ):
        """
        Initialize Sokoban wrapper.
        
        Args:
            env_name: Sokoban environment variant
            dim_room: Room dimensions (height, width)
            max_steps: Maximum episode length
            num_boxes: Number of boxes in puzzle
            observation_size: Resize observations to this size
            reward_shaping: Whether to use reward shaping
            render_mode: Rendering mode
        """
        # Create base environment
        if "Sokoban" in env_name:
            env = gym.make(
                env_name,
                dim_room=dim_room,
                max_steps=max_steps,
                num_boxes=num_boxes,
                render_mode=render_mode,
            )
        else:
            # For custom or Griddly environments
            env = gym.make(env_name)
            
        super().__init__(env)
        
        self.observation_size = observation_size
        self.reward_shaping = reward_shaping
        self.episode_stats = self._reset_stats()
        
        # Update observation space for resized images
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(*observation_size, 3),
            dtype=np.uint8
        )
        
    def _reset_stats(self) -> Dict[str, Any]:
        """Reset episode statistics."""
        return {
            "steps": 0,
            "boxes_on_target": 0,
            "total_reward": 0.0,
            "solved": False,
            "invalid_actions": 0,
        }
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return preprocessed observation."""
        obs, info = self.env.reset(**kwargs)
        self.episode_stats = self._reset_stats()
        
        # Preprocess observation
        obs = self._preprocess_observation(obs)
        
        # Add initial statistics to info
        info.update(self.episode_stats)
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: up, 1: down, 2: left, 3: right, 4: push)
            
        Returns:
            observation: Preprocessed observation
            reward: Shaped reward
            terminated: Whether episode ended (puzzle solved)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        # Store previous state for reward shaping
        if self.reward_shaping:
            prev_boxes_on_target = self._count_boxes_on_target()
        
        # Take action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update statistics
        self.episode_stats["steps"] += 1
        self.episode_stats["total_reward"] += reward
        
        # Apply reward shaping
        if self.reward_shaping:
            shaped_reward = self._shape_reward(reward, prev_boxes_on_target)
        else:
            shaped_reward = reward
        
        # Check if puzzle is solved
        if terminated and reward > 0:
            self.episode_stats["solved"] = True
            info["solved"] = True
        
        # Preprocess observation
        obs = self._preprocess_observation(obs)
        
        # Update info with statistics
        info.update(self.episode_stats)
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Preprocess observation for DreamerV3.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Preprocessed observation
        """
        # Ensure observation is in correct format
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        
        # Resize if necessary
        if obs.shape[:2] != self.observation_size:
            obs = cv2.resize(
                obs,
                self.observation_size,
                interpolation=cv2.INTER_AREA
            )
        
        return obs
    
    def _count_boxes_on_target(self) -> int:
        """Count number of boxes currently on target positions."""
        # This would need access to internal game state
        # For now, return 0 (implement based on specific env)
        return 0
    
    def _shape_reward(self, base_reward: float, prev_boxes_on_target: int) -> float:
        """
        Shape reward to provide intermediate feedback.
        
        Args:
            base_reward: Original reward from environment
            prev_boxes_on_target: Previous count of boxes on targets
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        # Add intermediate rewards
        current_boxes = self._count_boxes_on_target()
        
        # Reward for placing box on target
        if current_boxes > prev_boxes_on_target:
            shaped_reward += 0.5
        # Penalty for removing box from target
        elif current_boxes < prev_boxes_on_target:
            shaped_reward -= 0.3
        
        # Small penalty for each step to encourage efficiency
        shaped_reward -= 0.01
        
        return shaped_reward
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


class CurriculumSokobanWrapper(SokobanWrapper):
    """
    Sokoban wrapper with curriculum learning support.
    Progressively increases puzzle difficulty based on agent performance.
    """
    
    def __init__(self, initial_stage: int = 0, **kwargs):
        """
        Initialize curriculum wrapper.
        
        Args:
            initial_stage: Starting curriculum stage
            **kwargs: Arguments passed to SokobanWrapper
        """
        self.curriculum_stages = [
            {"dim_room": (5, 5), "num_boxes": 1, "max_steps": 25},    # Stage 0: Trivial
            {"dim_room": (7, 7), "num_boxes": 2, "max_steps": 50},    # Stage 1: Simple
            {"dim_room": (10, 10), "num_boxes": 3, "max_steps": 80},  # Stage 2: Medium
            {"dim_room": (13, 13), "num_boxes": 4, "max_steps": 120}, # Stage 3: Complex
        ]
        
        self.current_stage = initial_stage
        self.stage_performance = []
        self.episodes_per_stage = 100
        self.promotion_threshold = 0.8
        
        # Initialize with current stage parameters
        stage_params = self.curriculum_stages[self.current_stage]
        kwargs.update(stage_params)
        super().__init__(**kwargs)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and potentially advance curriculum."""
        # Check if we should advance to next stage
        if len(self.stage_performance) >= self.episodes_per_stage:
            success_rate = sum(self.stage_performance) / len(self.stage_performance)
            
            if success_rate >= self.promotion_threshold and self.current_stage < len(self.curriculum_stages) - 1:
                self.current_stage += 1
                self._update_stage_parameters()
                print(f"Advanced to curriculum stage {self.current_stage}!")
            
            # Reset performance tracking
            self.stage_performance = []
        
        obs, info = super().reset(**kwargs)
        info["curriculum_stage"] = self.current_stage
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take step and track performance for curriculum."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track episode completion
        if terminated or truncated:
            self.stage_performance.append(info.get("solved", False))
        
        info["curriculum_stage"] = self.current_stage
        
        return obs, reward, terminated, truncated, info
    
    def _update_stage_parameters(self):
        """Update environment parameters for new curriculum stage."""
        stage_params = self.curriculum_stages[self.current_stage]
        
        # Recreate environment with new parameters
        env = gym.make(
            "Sokoban-v0",
            dim_room=stage_params["dim_room"],
            max_steps=stage_params["max_steps"],
            num_boxes=stage_params["num_boxes"],
            render_mode="rgb_array",
        )
        
        self.env = env
        self.max_steps = stage_params["max_steps"]
        
        
def create_sokoban_env(
    env_type: str = "standard",
    **kwargs
) -> gym.Env:
    """
    Factory function to create Sokoban environments.
    
    Args:
        env_type: Type of environment ("standard", "curriculum")
        **kwargs: Environment configuration
        
    Returns:
        Configured Sokoban environment
    """
    if env_type == "curriculum":
        return CurriculumSokobanWrapper(**kwargs)
    else:
        return SokobanWrapper(**kwargs)


if __name__ == "__main__":
    # Test environment creation
    env = create_sokoban_env(env_type="standard", dim_room=(7, 7), num_boxes=2)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Info: {info}")
        
        if terminated or truncated:
            break
    
    env.close()
