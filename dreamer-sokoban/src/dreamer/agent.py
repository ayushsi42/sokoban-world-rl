"""
DreamerV3 agent implementation for Sokoban.
Includes actor-critic networks and planning algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import deque
import copy

from .world_model import HierarchicalWorldModel


class Actor(nn.Module):
    """Actor network for DreamerV3."""
    
    def __init__(
        self,
        feature_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = True,
        action_std: float = 0.1,
    ):
        super().__init__()
        
        self.action_size = action_size
        self.action_std = action_std
        
        layers = []
        in_size = feature_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(getattr(nn, activation.upper())())
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> dist.Distribution:
        """Forward pass returning action distribution."""
        logits = self.network(features)
        
        # For discrete actions, use categorical distribution
        return dist.Categorical(logits=logits)
    
    def act(self, features: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Select action from features."""
        action_dist = self(features)
        
        if sample:
            action = action_dist.sample()
        else:
            action = action_dist.probs.argmax(dim=-1)
        
        # Convert to one-hot
        action_one_hot = F.one_hot(action, num_classes=self.action_size).float()
        
        return action_one_hot
    
    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions."""
        action_dist = self(features)
        
        # Convert one-hot to indices if needed
        if actions.dim() > 1 and actions.shape[-1] == self.action_size:
            actions = actions.argmax(dim=-1)
        
        return action_dist.log_prob(actions)


class Critic(nn.Module):
    """Critic network for DreamerV3."""
    
    def __init__(
        self,
        feature_size: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu",
        layer_norm: bool = True,
        output_shape: Tuple[int] = (1,),
    ):
        super().__init__()
        
        layers = []
        in_size = feature_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(getattr(nn, activation.upper())())
            in_size = hidden_size
        
        layers.append(nn.Linear(in_size, np.prod(output_shape)))
        
        self.network = nn.Sequential(*layers)
        self.output_shape = output_shape
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimates."""
        value = self.network(features)
        
        if len(self.output_shape) > 1:
            value = value.view(features.shape[0], *self.output_shape)
        
        return value


class DreamerV3Agent:
    """
    DreamerV3 agent for Sokoban puzzle solving.
    Includes world model, actor-critic, and planning components.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        action_size: int = 5,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize DreamerV3 agent.
        
        Args:
            observation_shape: Shape of observations (C, H, W)
            action_size: Number of discrete actions
            config: Configuration dictionary
            device: Device to use
        """
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.device = torch.device(device)
        
        # Default configuration
        default_config = {
            "world_model": {
                "encoder_depths": [32, 64, 128, 256],
                "decoder_depths": [256, 128, 64, 32],
                "rssm": {
                    "deterministic_size": 512,
                    "stochastic_size": 32,
                    "categories": 32,
                },
            },
            "actor": {
                "hidden_sizes": [256, 256],
                "layer_norm": True,
            },
            "critic": {
                "hidden_sizes": [256, 256],
                "layer_norm": True,
            },
            "training": {
                "batch_size": 16,
                "sequence_length": 64,
                "imagination_horizon": 15,
                "discount": 0.99,
                "lambda_": 0.95,
                "model_lr": 1e-4,
                "actor_lr": 3e-4,
                "critic_lr": 3e-4,
                "grad_clip": 100.0,
                "free_nats": 3.0,
                "kl_weight": 1.0,
                "reward_weight": 1.0,
                "terminal_weight": 1.0,
                "reconstruction_weight": 1.0,
            },
            "planning": {
                "use_value_guidance": True,
                "num_candidates": 5,
                "temperature": 1.0,
            },
        }
        
        # Merge with provided config
        self.config = self._merge_configs(default_config, config or {})
        
        # Initialize components
        self._build_networks()
        self._build_optimizers()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            observation_shape=observation_shape,
            action_size=action_size,
            sequence_length=self.config["training"]["sequence_length"],
            device=self.device,
        )
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        merged = copy.deepcopy(default)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _build_networks(self):
        """Build neural networks."""
        # World model
        self.world_model = HierarchicalWorldModel(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            encoder_depths=self.config["world_model"]["encoder_depths"],
            decoder_depths=self.config["world_model"]["decoder_depths"],
            rssm_config=self.config["world_model"]["rssm"],
        ).to(self.device)
        
        # Get feature size from world model
        with torch.no_grad():
            dummy_state = self.world_model.rssm.initial_state(1, self.device)
            feature_size = self.world_model.rssm.get_features(dummy_state).shape[-1]
        
        # Actor
        self.actor = Actor(
            feature_size=feature_size,
            action_size=self.action_size,
            **self.config["actor"],
        ).to(self.device)
        
        # Critic
        self.critic = Critic(
            feature_size=feature_size,
            **self.config["critic"],
        ).to(self.device)
        
        # Target critic for stable training
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        
    def _build_optimizers(self):
        """Build optimizers for each component."""
        # World model optimizer
        self.model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.config["training"]["model_lr"],
        )
        
        # Actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config["training"]["actor_lr"],
        )
        
        # Critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config["training"]["critic_lr"],
        )
    
    def act(
        self,
        observation: np.ndarray,
        state: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = True,
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            state: Current RSSM state
            training: Whether in training mode
            
        Returns:
            action: Selected action
            state: Updated RSSM state
        """
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(observation).float().to(self.device)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (B, C, H, W)
        
        # Initialize state if needed
        if state is None:
            state = self.world_model.rssm.initial_state(1, self.device)
        
        with torch.no_grad():
            # Encode observation
            obs_embed = self.world_model.encoder(obs_tensor)
            
            # Update state with observation
            dummy_action = torch.zeros(1, self.action_size, device=self.device)
            state, _ = self.world_model.rssm.observe(
                obs_embed, dummy_action, state, sample=False
            )
            
            # Get features
            features = self.world_model.rssm.get_features(state)
            
            # Select action
            if self.config["planning"]["use_value_guidance"]:
                action_one_hot = self._value_guided_action(features, state)
            else:
                action_one_hot = self.actor.act(features, sample=training)
            
            # Convert to integer
            action = action_one_hot.argmax(dim=-1).item()
        
        return action, state
    
    def _value_guided_action(
        self,
        features: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Select action using value-guided planning."""
        horizon = self.config["training"]["imagination_horizon"]
        num_candidates = self.config["planning"]["num_candidates"]
        
        # Sample candidate action sequences
        candidate_actions = []
        candidate_values = []
        
        for _ in range(num_candidates):
            actions = []
            values = []
            current_state = copy.deepcopy(state)
            
            for t in range(horizon):
                # Get features for current state
                current_features = self.world_model.rssm.get_features(current_state)
                
                # Sample action
                action = self.actor.act(current_features, sample=True)
                actions.append(action)
                
                # Imagine next state
                current_state, _ = self.world_model.rssm.imagine(
                    action, current_state, sample=True
                )
                
                # Estimate value
                next_features = self.world_model.rssm.get_features(current_state)
                value = self.critic(next_features).squeeze()
                values.append(value)
            
            candidate_actions.append(torch.stack(actions))
            candidate_values.append(torch.stack(values).sum())
        
        # Select best candidate based on total value
        best_idx = torch.stack(candidate_values).argmax()
        
        # Return first action of best sequence
        return candidate_actions[best_idx][0]
    
    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Train agent on batch of data.
        
        Args:
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < batch_size or batch_size:
            return {}
        
        batch_size = batch_size or self.config["training"]["batch_size"]
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        
        # Train world model
        model_metrics = self._train_world_model(batch)
        
        # Train actor-critic on imagined trajectories
        actor_critic_metrics = self._train_actor_critic(batch)
        
        # Update target network
        if self.training_step % 100 == 0:
            self._update_target_critic()
        
        self.training_step += 1
        
        # Combine metrics
        metrics = {**model_metrics, **actor_critic_metrics}
        metrics["training_step"] = self.training_step
        
        return metrics
    
    def _train_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train world model on batch."""
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        
        # Forward pass through world model
        outputs = self.world_model(
            observations, actions, initial_state=None, imagine=False
        )
        
        # Reconstruction loss
        recon_loss = F.mse_loss(
            outputs["reconstructions"],
            observations,
            reduction="mean"
        )
        
        # Reward prediction loss
        reward_loss = F.mse_loss(
            outputs["rewards"],
            rewards,
            reduction="mean"
        )
        
        # Terminal prediction loss
        terminal_loss = F.binary_cross_entropy(
            outputs["terminals"],
            terminals.float(),
            reduction="mean"
        )
        
        # KL divergence loss
        kl_loss = self.world_model.rssm.kl_divergence(
            outputs["posterior_dists"],
            outputs["prior_dists"]
        ).mean()
        
        # Apply free nats
        kl_loss = torch.max(
            kl_loss,
            torch.tensor(self.config["training"]["free_nats"], device=self.device)
        )
        
        # Total loss
        total_loss = (
            self.config["training"]["reconstruction_weight"] * recon_loss +
            self.config["training"]["reward_weight"] * reward_loss +
            self.config["training"]["terminal_weight"] * terminal_loss +
            self.config["training"]["kl_weight"] * kl_loss
        )
        
        # Optimize
        self.model_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.world_model.parameters(),
            self.config["training"]["grad_clip"]
        )
        
        self.model_optimizer.step()
        
        return {
            "model/total_loss": total_loss.item(),
            "model/recon_loss": recon_loss.item(),
            "model/reward_loss": reward_loss.item(),
            "model/terminal_loss": terminal_loss.item(),
            "model/kl_loss": kl_loss.item(),
        }
    
    def _train_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train actor and critic on imagined trajectories."""
        observations = batch["observations"]
        actions = batch["actions"]
        
        batch_size, seq_len = observations.shape[:2]
        horizon = self.config["training"]["imagination_horizon"]
        
        with torch.no_grad():
            # Get initial states by encoding observations
            initial_obs = observations[:, 0]
            obs_embed = self.world_model.encoder(initial_obs)
            
            # Initialize state
            state = self.world_model.rssm.initial_state(batch_size, self.device)
            
            # Update with observation
            dummy_action = torch.zeros(batch_size, self.action_size, device=self.device)
            state, _ = self.world_model.rssm.observe(
                obs_embed, dummy_action, state, sample=True
            )
        
        # Imagine trajectories
        imagined_features = []
        imagined_rewards = []
        imagined_terminals = []
        imagined_actions = []
        
        for t in range(horizon):
            # Get features
            features = self.world_model.rssm.get_features(state)
            imagined_features.append(features)
            
            # Sample action from actor
            action = self.actor.act(features, sample=True)
            imagined_actions.append(action)
            
            # Imagine next state
            with torch.no_grad():
                state, _ = self.world_model.rssm.imagine(action, state, sample=True)
                
                # Predict reward and terminal
                next_features = self.world_model.rssm.get_features(state)
                reward = self.world_model.reward_predictor(next_features).squeeze(-1)
                terminal = torch.sigmoid(
                    self.world_model.terminal_predictor(next_features).squeeze(-1)
                )
                
                imagined_rewards.append(reward)
                imagined_terminals.append(terminal)
        
        # Stack imagined data
        imagined_features = torch.stack(imagined_features, dim=1)
        imagined_actions = torch.stack(imagined_actions, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_terminals = torch.stack(imagined_terminals, dim=1)
        
        # Compute returns using TD-lambda
        with torch.no_grad():
            values = self.critic(imagined_features)
            returns = self._compute_returns(
                imagined_rewards,
                values,
                imagined_terminals,
                self.config["training"]["discount"],
                self.config["training"]["lambda_"]
            )
        
        # Actor loss (maximize returns)
        actor_loss = -returns.mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config["training"]["grad_clip"]
        )
        
        self.actor_optimizer.step()
        
        # Critic loss (minimize value prediction error)
        values = self.critic(imagined_features.detach())
        critic_loss = F.mse_loss(values, returns.detach())
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config["training"]["grad_clip"]
        )
        
        self.critic_optimizer.step()
        
        return {
            "actor/loss": actor_loss.item(),
            "critic/loss": critic_loss.item(),
            "actor/returns": returns.mean().item(),
            "critic/values": values.mean().item(),
        }
    
    def _compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminals: torch.Tensor,
        discount: float,
        lambda_: float,
    ) -> torch.Tensor:
        """Compute TD-lambda returns."""
        batch_size, horizon = rewards.shape
        
        # Last value for bootstrapping
        with torch.no_grad():
            last_value = self.target_critic(
                self.world_model.rssm.get_features(
                    self.world_model.rssm.initial_state(batch_size, self.device)
                )
            ).squeeze(-1)
        
        returns = torch.zeros_like(rewards)
        
        # Backward pass to compute returns
        returns[:, -1] = rewards[:, -1] + discount * (1 - terminals[:, -1]) * last_value
        
        for t in reversed(range(horizon - 1)):
            returns[:, t] = (
                rewards[:, t] +
                discount * (1 - terminals[:, t]) * (
                    (1 - lambda_) * values[:, t + 1].squeeze(-1) +
                    lambda_ * returns[:, t + 1]
                )
            )
        
        return returns
    
    def _update_target_critic(self):
        """Update target critic network."""
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "config": self.config,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.config = checkpoint["config"]


class ReplayBuffer:
    """Replay buffer for storing and sampling experience."""
    
    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, int, int],
        action_size: int,
        sequence_length: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.device = device
        
        self.episodes = deque(maxlen=capacity)
        
    def add_episode(self, episode: Dict[str, np.ndarray]):
        """Add complete episode to buffer."""
        self.episodes.append(episode)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of sequences."""
        # Sample episodes
        episode_indices = np.random.randint(0, len(self.episodes), batch_size)
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_terminals = []
        
        for idx in episode_indices:
            episode = self.episodes[idx]
            
            # Sample sequence start
            max_start = len(episode["observations"]) - self.sequence_length
            start = np.random.randint(0, max(1, max_start))
            
            # Extract sequence
            obs_seq = episode["observations"][start:start + self.sequence_length]
            action_seq = episode["actions"][start:start + self.sequence_length]
            reward_seq = episode["rewards"][start:start + self.sequence_length]
            terminal_seq = episode["terminals"][start:start + self.sequence_length]
            
            # Pad if necessary
            if len(obs_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(obs_seq)
                obs_seq = np.concatenate([
                    obs_seq,
                    np.zeros((pad_len, *self.observation_shape), dtype=obs_seq.dtype)
                ])
                action_seq = np.concatenate([
                    action_seq,
                    np.zeros((pad_len, self.action_size), dtype=action_seq.dtype)
                ])
                reward_seq = np.concatenate([
                    reward_seq,
                    np.zeros(pad_len, dtype=reward_seq.dtype)
                ])
                terminal_seq = np.concatenate([
                    terminal_seq,
                    np.zeros(pad_len, dtype=terminal_seq.dtype)
                ])
            
            batch_obs.append(obs_seq)
            batch_actions.append(action_seq)
            batch_rewards.append(reward_seq)
            batch_terminals.append(terminal_seq)
        
        # Convert to tensors
        batch = {
            "observations": torch.from_numpy(np.array(batch_obs)).float().to(self.device),
            "actions": torch.from_numpy(np.array(batch_actions)).float().to(self.device),
            "rewards": torch.from_numpy(np.array(batch_rewards)).float().to(self.device),
            "terminals": torch.from_numpy(np.array(batch_terminals)).float().to(self.device),
        }
        
        # Ensure correct shape for observations (B, T, C, H, W)
        if batch["observations"].shape[2] != self.observation_shape[0]:
            batch["observations"] = batch["observations"].permute(0, 1, 4, 2, 3)
        
        return batch
    
    def __len__(self):
        """Get number of episodes in buffer."""
        return len(self.episodes)
