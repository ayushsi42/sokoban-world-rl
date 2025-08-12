#!/usr/bin/env python3
"""
Main script to run DreamerV3-Sokoban experiments.
Supports training, evaluation, and visualization.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
import torch

from train import Trainer
from dreamer.agent import DreamerV3Agent
from environments.sokoban_wrapper import create_sokoban_env
from analysis.evaluation import (
    evaluate_agent,
    evaluate_planning_quality,
    evaluate_curriculum_progression,
    create_evaluation_report,
)
from analysis.visualization import (
    visualize_trajectory,
    create_training_summary_plot,
    visualize_value_landscape,
)


def train_agent(args):
    """Train DreamerV3 agent."""
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with experiment config if provided
    if args.experiment:
        exp_config = OmegaConf.load("configs/experiment.yaml")[args.experiment]
        config = OmegaConf.merge(config, exp_config)
    
    # Override with command line arguments
    if args.steps:
        config.training.total_steps = args.steps
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device
    
    # Update run name
    config.run_name = args.run_name or f"dreamer_sokoban_{args.experiment or 'default'}"
    
    # Create trainer and run
    trainer = Trainer(config)
    trainer.train()


def evaluate_trained_agent(args):
    """Evaluate a trained agent."""
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    
    # Create environment
    env = create_sokoban_env(
        env_type="standard" if not config["environment"].get("use_curriculum", False) else "curriculum",
        **config["environment"]
    )
    
    # Create agent
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    agent = DreamerV3Agent(
        observation_shape=obs_shape,
        action_size=action_size,
        config=config["agent"],
        device=args.device,
    )
    
    # Load weights
    agent.load(args.checkpoint)
    
    # Run evaluation
    print("Running standard evaluation...")
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        render=args.render,
        device=torch.device(args.device),
    )
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Planning quality evaluation
    if args.eval_planning:
        print("\nEvaluating planning quality...")
        planning_results = evaluate_planning_quality(
            agent=agent,
            env=env,
            num_episodes=10,
            device=torch.device(args.device),
        )
        
        print("\nPlanning Quality:")
        for key, value in planning_results.items():
            print(f"{key}: {value:.4f}")
    
    # Save report
    if args.save_report:
        report_path = Path(args.checkpoint).parent / "evaluation_report.md"
        create_evaluation_report(results, str(report_path))
    
    env.close()


def visualize_agent(args):
    """Visualize agent behavior."""
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    
    # Create environment
    env = create_sokoban_env(
        env_type="standard",
        **config["environment"]
    )
    
    # Create agent
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    agent = DreamerV3Agent(
        observation_shape=obs_shape,
        action_size=action_size,
        config=config["agent"],
        device=args.device,
    )
    
    # Load weights
    agent.load(args.checkpoint)
    
    # Generate trajectory
    obs, _ = env.reset()
    
    # Create action sequence
    if args.random_actions:
        actions = [env.action_space.sample() for _ in range(args.trajectory_length)]
    else:
        # Use agent's policy
        actions = []
        state = None
        for _ in range(args.trajectory_length):
            action, state = agent.act(obs, state, training=False)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    # Reset and get initial observation
    obs, _ = env.reset()
    
    # Convert actions to tensor
    action_sequence = torch.zeros(1, len(actions), agent.action_size, device=args.device)
    for t, action in enumerate(actions):
        action_sequence[0, t, action] = 1.0
    
    # Imagine trajectory
    obs_tensor = torch.from_numpy(obs).float().to(args.device)
    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        trajectory = agent.world_model.imagine_trajectory(obs_tensor, action_sequence)
    
    # Visualize
    save_path = Path(args.checkpoint).parent / f"trajectory_{args.trajectory_name}.gif"
    visualize_trajectory(
        real_obs=obs,
        imagined_trajectory=trajectory,
        save_path=str(save_path),
        fps=args.fps,
    )
    
    # Value landscape
    if args.visualize_values:
        value_path = Path(args.checkpoint).parent / "value_landscape.png"
        visualize_value_landscape(agent, env, save_path=str(value_path))
    
    env.close()


def analyze_training(args):
    """Analyze training logs."""
    log_dir = Path(args.log_dir)
    
    # Find training state files
    state_files = list(log_dir.glob("training_state_*.json"))
    if not state_files:
        print(f"No training state files found in {log_dir}")
        return
    
    # Use latest state file
    latest_state = sorted(state_files, key=lambda x: int(x.stem.split("_")[-1]))[-1]
    
    # Create training summary
    summary_path = log_dir / "training_summary.png"
    create_training_summary_plot(str(latest_state), str(summary_path))
    
    # Curriculum progression analysis
    if args.analyze_curriculum:
        curriculum_path = log_dir / "curriculum_analysis.png"
        curriculum_results = evaluate_curriculum_progression(
            str(latest_state),
            save_path=str(curriculum_path),
        )
        
        print("\nCurriculum Analysis:")
        print(f"Total stages reached: {curriculum_results.get('total_stages_reached', 'N/A')}")
        print(f"Final stage success rate: {curriculum_results.get('final_stage_success_rate', 'N/A'):.2%}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3-Sokoban Experiment Runner")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train agent")
    train_parser.add_argument("--config", type=str, default="configs/train.yaml",
                              help="Configuration file")
    train_parser.add_argument("--experiment", type=str, choices=[
        "small_puzzle", "medium_puzzle", "large_puzzle",
        "no_curriculum", "fast_curriculum", "discrete_optimized",
        "planning_focused", "baseline"
    ], help="Predefined experiment configuration")
    train_parser.add_argument("--run-name", type=str, help="Run name for logging")
    train_parser.add_argument("--steps", type=int, help="Total training steps")
    train_parser.add_argument("--seed", type=int, help="Random seed")
    train_parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                              help="Device to use")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--num-episodes", type=int, default=100,
                             help="Number of evaluation episodes")
    eval_parser.add_argument("--render", action="store_true",
                             help="Render episodes")
    eval_parser.add_argument("--eval-planning", action="store_true",
                             help="Evaluate planning quality")
    eval_parser.add_argument("--save-report", action="store_true",
                             help="Save evaluation report")
    eval_parser.add_argument("--device", type=str, default="cuda",
                             help="Device to use")
    
    # Visualization parser
    viz_parser = subparsers.add_parser("visualize", help="Visualize agent behavior")
    viz_parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    viz_parser.add_argument("--trajectory-length", type=int, default=15,
                            help="Length of trajectory to visualize")
    viz_parser.add_argument("--trajectory-name", type=str, default="imagined",
                            help="Name for saved trajectory")
    viz_parser.add_argument("--random-actions", action="store_true",
                            help="Use random actions instead of policy")
    viz_parser.add_argument("--visualize-values", action="store_true",
                            help="Visualize value landscape")
    viz_parser.add_argument("--fps", type=int, default=4,
                            help="FPS for trajectory animation")
    viz_parser.add_argument("--device", type=str, default="cuda",
                            help="Device to use")
    
    # Analysis parser
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training logs")
    analyze_parser.add_argument("log_dir", type=str, help="Log directory")
    analyze_parser.add_argument("--analyze-curriculum", action="store_true",
                                help="Analyze curriculum progression")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_agent(args)
    elif args.command == "evaluate":
        evaluate_trained_agent(args)
    elif args.command == "visualize":
        visualize_agent(args)
    elif args.command == "analyze":
        analyze_training(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
