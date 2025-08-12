"""
Evaluation framework for DreamerV3-Sokoban.
Includes metrics computation and performance analysis.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 100,
    render: bool = False,
    device: torch.device = torch.device("cpu"),
    save_trajectories: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate agent performance across multiple episodes.
    
    Args:
        agent: DreamerV3 agent to evaluate
        env: Evaluation environment
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        device: Device for computation
        save_trajectories: Whether to save episode trajectories
        verbose: Whether to print progress
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = defaultdict(list)
    trajectories = []
    
    # Set agent to evaluation mode
    agent.world_model.eval()
    agent.actor.eval()
    agent.critic.eval()
    
    iterator = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
    
    for episode_idx in iterator:
        episode_metrics, trajectory = run_episode(
            agent=agent,
            env=env,
            render=render,
            device=device,
        )
        
        # Collect metrics
        for key, value in episode_metrics.items():
            metrics[key].append(value)
        
        if save_trajectories:
            trajectories.append(trajectory)
    
    # Compute aggregate statistics
    results = {}
    for key, values in metrics.items():
        values = np.array(values)
        results[f"{key}_mean"] = float(np.mean(values))
        results[f"{key}_std"] = float(np.std(values))
        results[f"{key}_min"] = float(np.min(values))
        results[f"{key}_max"] = float(np.max(values))
    
    # Add success rate
    results["success_rate"] = float(np.mean(metrics["solved"]))
    
    # Add efficiency metrics
    successful_episodes = [i for i, s in enumerate(metrics["solved"]) if s]
    if successful_episodes:
        successful_lengths = [metrics["length"][i] for i in successful_episodes]
        results["avg_successful_length"] = float(np.mean(successful_lengths))
    else:
        results["avg_successful_length"] = float(env.max_steps)
    
    # Set agent back to training mode
    agent.world_model.train()
    agent.actor.train()
    agent.critic.train()
    
    return results


def run_episode(
    agent,
    env,
    render: bool = False,
    device: torch.device = torch.device("cpu"),
    max_steps: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, List]]:
    """
    Run a single evaluation episode.
    
    Args:
        agent: DreamerV3 agent
        env: Environment
        render: Whether to render
        device: Device for computation
        max_steps: Maximum steps (uses env default if None)
        
    Returns:
        episode_metrics: Dictionary of episode metrics
        trajectory: Episode trajectory data
    """
    obs, info = env.reset()
    state = None
    
    trajectory = {
        "observations": [obs],
        "actions": [],
        "rewards": [],
        "infos": [info],
    }
    
    total_reward = 0.0
    steps = 0
    done = False
    
    max_steps = max_steps or env.max_steps
    
    while not done and steps < max_steps:
        # Select action (no exploration in evaluation)
        with torch.no_grad():
            action, state = agent.act(obs, state, training=False)
        
        # Take environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update trajectory
        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["infos"].append(info)
        
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
    
    # Episode metrics
    episode_metrics = {
        "reward": total_reward,
        "length": steps,
        "solved": float(info.get("solved", False)),
        "boxes_on_target": info.get("boxes_on_target", 0),
    }
    
    # Add curriculum stage if applicable
    if "curriculum_stage" in info:
        episode_metrics["curriculum_stage"] = info["curriculum_stage"]
    
    return episode_metrics, trajectory


def evaluate_planning_quality(
    agent,
    env,
    num_episodes: int = 10,
    planning_horizon: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluate the quality of agent's planning/imagination.
    
    Args:
        agent: DreamerV3 agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        planning_horizon: Horizon for planning evaluation
        device: Device for computation
        
    Returns:
        Dictionary of planning quality metrics
    """
    planning_metrics = defaultdict(list)
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        
        # Generate real trajectory
        real_observations = [obs]
        real_rewards = []
        actions = []
        
        for _ in range(planning_horizon):
            action = env.action_space.sample()
            actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            real_observations.append(obs)
            real_rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Reset and imagine same trajectory
        obs, _ = env.reset()
        
        with torch.no_grad():
            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).float().to(device)
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            
            action_sequence = torch.zeros(1, len(actions), agent.action_size, device=device)
            for t, action in enumerate(actions):
                action_sequence[0, t, action] = 1.0
            
            # Imagine trajectory
            imagined = agent.world_model.imagine_trajectory(obs_tensor, action_sequence)
            
            # Extract predictions
            imagined_observations = imagined["observations"].cpu().numpy()
            imagined_rewards = imagined["rewards"].cpu().numpy()
        
        # Compute metrics
        # Observation prediction error
        obs_errors = []
        for t in range(min(len(real_observations) - 1, imagined_observations.shape[1] - 1)):
            real_obs = real_observations[t + 1]
            pred_obs = imagined_observations[0, t + 1].transpose(1, 2, 0)
            
            # Resize if needed
            if real_obs.shape != pred_obs.shape:
                import cv2
                pred_obs = cv2.resize(pred_obs, (real_obs.shape[1], real_obs.shape[0]))
            
            error = np.mean(np.abs(real_obs - pred_obs))
            obs_errors.append(error)
        
        planning_metrics["observation_error"].extend(obs_errors)
        
        # Reward prediction error
        reward_errors = []
        for t in range(min(len(real_rewards), imagined_rewards.shape[1])):
            error = abs(real_rewards[t] - imagined_rewards[0, t])
            reward_errors.append(error)
        
        planning_metrics["reward_error"].extend(reward_errors)
    
    # Aggregate metrics
    results = {
        "avg_observation_error": float(np.mean(planning_metrics["observation_error"])),
        "avg_reward_error": float(np.mean(planning_metrics["reward_error"])),
        "max_observation_error": float(np.max(planning_metrics["observation_error"])),
        "max_reward_error": float(np.max(planning_metrics["reward_error"])),
    }
    
    return results


def evaluate_curriculum_progression(
    training_logs: str,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze curriculum learning progression from training logs.
    
    Args:
        training_logs: Path to training logs
        save_path: Optional path to save analysis plots
        
    Returns:
        Dictionary of curriculum analysis results
    """
    # Load training data
    with open(training_logs, "r") as f:
        data = json.load(f)
    
    # Extract curriculum-related metrics
    stages = []
    success_rates = []
    episode_lengths = []
    
    for episode in data.get("episodes", []):
        if "curriculum_stage" in episode:
            stages.append(episode["curriculum_stage"])
            success_rates.append(episode.get("solved", False))
            episode_lengths.append(episode["length"])
    
    if not stages:
        return {"error": "No curriculum data found"}
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        "stage": stages,
        "success": success_rates,
        "length": episode_lengths,
    })
    
    # Compute statistics per stage
    stage_stats = df.groupby("stage").agg({
        "success": ["mean", "count"],
        "length": ["mean", "std"],
    })
    
    results = {
        "stage_progression": stages,
        "stage_statistics": stage_stats.to_dict(),
        "total_stages_reached": int(max(stages)) + 1,
        "final_stage_success_rate": float(
            df[df["stage"] == max(stages)]["success"].mean()
        ),
    }
    
    # Create visualization if requested
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Stage progression over time
        axes[0, 0].plot(stages)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Curriculum Stage")
        axes[0, 0].set_title("Stage Progression")
        
        # Success rate by stage
        stage_success = df.groupby("stage")["success"].mean()
        axes[0, 1].bar(stage_success.index, stage_success.values)
        axes[0, 1].set_xlabel("Stage")
        axes[0, 1].set_ylabel("Success Rate")
        axes[0, 1].set_title("Success Rate by Stage")
        
        # Episode length by stage
        axes[1, 0].boxplot(
            [df[df["stage"] == s]["length"].values for s in sorted(df["stage"].unique())]
        )
        axes[1, 0].set_xlabel("Stage")
        axes[1, 0].set_ylabel("Episode Length")
        axes[1, 0].set_title("Episode Length Distribution by Stage")
        
        # Rolling success rate
        window = 50
        rolling_success = pd.Series(success_rates).rolling(window).mean()
        axes[1, 1].plot(rolling_success)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel(f"Success Rate (rolling {window})")
        axes[1, 1].set_title("Learning Progress")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    return results


def evaluate_sample_efficiency(
    agent_logs: List[str],
    baseline_logs: Optional[List[str]] = None,
    metric: str = "success_rate",
) -> Dict[str, Any]:
    """
    Compare sample efficiency across different agents.
    
    Args:
        agent_logs: List of paths to agent training logs
        baseline_logs: Optional list of baseline agent logs
        metric: Metric to compare
        
    Returns:
        Sample efficiency comparison results
    """
    results = {}
    
    # Load agent data
    agent_data = []
    for log_path in agent_logs:
        with open(log_path, "r") as f:
            data = json.load(f)
        agent_data.append(data)
    
    # Load baseline data if provided
    baseline_data = []
    if baseline_logs:
        for log_path in baseline_logs:
            with open(log_path, "r") as f:
                data = json.load(f)
            baseline_data.append(data)
    
    # Compute steps to reach performance thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        # Agent performance
        agent_steps = []
        for data in agent_data:
            steps = find_threshold_crossing(data, metric, threshold)
            agent_steps.append(steps)
        
        results[f"agent_steps_to_{threshold}"] = {
            "mean": float(np.mean([s for s in agent_steps if s is not None])),
            "std": float(np.std([s for s in agent_steps if s is not None])),
            "values": agent_steps,
        }
        
        # Baseline performance
        if baseline_data:
            baseline_steps = []
            for data in baseline_data:
                steps = find_threshold_crossing(data, metric, threshold)
                baseline_steps.append(steps)
            
            results[f"baseline_steps_to_{threshold}"] = {
                "mean": float(np.mean([s for s in baseline_steps if s is not None])),
                "std": float(np.std([s for s in baseline_steps if s is not None])),
                "values": baseline_steps,
            }
    
    return results


def find_threshold_crossing(
    data: Dict[str, Any],
    metric: str,
    threshold: float,
) -> Optional[int]:
    """Find first step where metric crosses threshold."""
    values = data.get(metric, [])
    steps = data.get("steps", list(range(len(values))))
    
    for i, value in enumerate(values):
        if value >= threshold:
            return steps[i]
    
    return None


def create_evaluation_report(
    results: Dict[str, Any],
    save_path: str,
) -> None:
    """
    Create comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save report
    """
    report = []
    report.append("# DreamerV3-Sokoban Evaluation Report\n")
    report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Performance metrics
    report.append("\n## Performance Metrics\n")
    for key, value in results.items():
        if isinstance(value, float):
            report.append(f"- **{key}**: {value:.4f}\n")
        else:
            report.append(f"- **{key}**: {value}\n")
    
    # Save report
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.writelines(report)
    
    print(f"Evaluation report saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded. Use evaluate_agent() to evaluate a trained agent.")
