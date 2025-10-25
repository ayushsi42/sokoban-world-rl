"""
Discrete world model implementation for DreamerV3-Sokoban.
Includes CategoricalRSSM and hierarchical state representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional, List
import numpy as np
from einops import rearrange, reduce


class CategoricalRSSM(nn.Module):
    """
    Categorical Recurrent State Space Model for discrete environments.
    Uses discrete latent states instead of continuous ones.
    """
    
    def __init__(
        self,
        obs_embed_size: int = 512,
        action_size: int = 5,  # Sokoban has 5 actions
        deterministic_size: int = 512,
        stochastic_size: int = 32,
        categories: int = 32,
        hidden_size: int = 512,
        activation: str = "relu",
        layer_norm: bool = True,
    ):
        """
        Initialize CategoricalRSSM.
        
        Args:
            obs_embed_size: Size of observation embeddings
            action_size: Number of actions
            deterministic_size: Size of deterministic state
            stochastic_size: Number of categorical variables
            categories: Number of categories per variable
            hidden_size: Hidden layer size
            activation: Activation function
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size
        self.categories = categories
        self.action_size = action_size
        
        # Activation function
        self.act = getattr(F, activation)
        
        # Prior network: p(z_t | h_t-1, a_t-1)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_size + action_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_size * categories),
        )
        
        # Posterior network: q(z_t | h_t-1, a_t-1, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_size + action_size + obs_embed_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, stochastic_size * categories),
        )
        
        # Deterministic state update: GRU cell
        self.gru_cell = nn.GRUCell(
            input_size=stochastic_size * categories + action_size,
            hidden_size=deterministic_size,
        )
        
        # State feature extraction
        self.state_to_feat = nn.Sequential(
            nn.Linear(deterministic_size + stochastic_size * categories, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        
    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Get initial state for sequence."""
        return {
            "deterministic": torch.zeros(batch_size, self.deterministic_size, device=device),
            "stochastic": torch.zeros(batch_size, self.stochastic_size, self.categories, device=device),
            "stochastic_logits": torch.zeros(batch_size, self.stochastic_size * self.categories, device=device),
        }
    
    def observe(
        self,
        obs_embed: torch.Tensor,
        action: torch.Tensor,
        state: Dict[str, torch.Tensor],
        sample: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Observe step: update state given observation.
        
        Args:
            obs_embed: Embedded observation
            action: Previous action (one-hot)
            state: Previous state
            sample: Whether to sample from posterior
            
        Returns:
            posterior_state: Updated state
            posterior_dist: Posterior distribution info
        """
        # Get previous states
        prev_deterministic = state["deterministic"]
        prev_stochastic = state["stochastic"].reshape(prev_deterministic.shape[0], -1)
        
        # Update deterministic state
        deterministic = self.gru_cell(
            torch.cat([prev_stochastic, action], dim=-1),
            prev_deterministic,
        )
        
        # Compute posterior
        posterior_input = torch.cat([deterministic, action, obs_embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = rearrange(
            posterior_logits,
            "b (n c) -> b n c",
            n=self.stochastic_size,
            c=self.categories,
        )
        
        # Sample or use mode
        if sample:
            stochastic = self._sample_categorical(posterior_logits)
        else:
            stochastic = F.one_hot(
                posterior_logits.argmax(dim=-1),
                num_classes=self.categories,
            ).float()
        
        posterior_state = {
            "deterministic": deterministic,
            "stochastic": stochastic,
            "stochastic_logits": rearrange(posterior_logits, "b n c -> b (n c)"),
        }
        
        posterior_dist = {
            "logits": posterior_logits,
            "probs": F.softmax(posterior_logits, dim=-1),
        }
        
        return posterior_state, posterior_dist
    
    def imagine(
        self,
        action: torch.Tensor,
        state: Dict[str, torch.Tensor],
        sample: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Imagine step: predict next state without observation.
        
        Args:
            action: Action to take (one-hot)
            state: Current state
            sample: Whether to sample from prior
            
        Returns:
            prior_state: Predicted next state
            prior_dist: Prior distribution info
        """
        # Get current states
        deterministic = state["deterministic"]
        stochastic = state["stochastic"].reshape(deterministic.shape[0], -1)
        
        # Update deterministic state
        next_deterministic = self.gru_cell(
            torch.cat([stochastic, action], dim=-1),
            deterministic,
        )
        
        # Compute prior
        prior_input = torch.cat([next_deterministic, action], dim=-1)
        prior_logits = self.prior_net(prior_input)
        prior_logits = rearrange(
            prior_logits,
            "b (n c) -> b n c",
            n=self.stochastic_size,
            c=self.categories,
        )
        
        # Sample or use mode
        if sample:
            next_stochastic = self._sample_categorical(prior_logits)
        else:
            next_stochastic = F.one_hot(
                prior_logits.argmax(dim=-1),
                num_classes=self.categories,
            ).float()
        
        prior_state = {
            "deterministic": next_deterministic,
            "stochastic": next_stochastic,
            "stochastic_logits": rearrange(prior_logits, "b n c -> b (n c)"),
        }
        
        prior_dist = {
            "logits": prior_logits,
            "probs": F.softmax(prior_logits, dim=-1),
        }
        
        return prior_state, prior_dist
    
    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from state for downstream networks."""
        deterministic = state["deterministic"]
        stochastic = state["stochastic"].reshape(deterministic.shape[0], -1)
        
        return self.state_to_feat(torch.cat([deterministic, stochastic], dim=-1))
    
    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with Gumbel-Softmax."""
        # Use Gumbel-Softmax for differentiable sampling
        return F.gumbel_softmax(logits, tau=0.5, hard=True)
    
    def kl_divergence(
        self,
        posterior_dist: Dict[str, torch.Tensor],
        prior_dist: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        posterior_probs = posterior_dist["probs"]
        prior_probs = prior_dist["probs"]
        
        # KL divergence for categorical distributions
        kl = posterior_probs * (posterior_probs.log() - prior_probs.log())
        kl = kl.sum(dim=-1).mean(dim=-1)  # Sum over categories, mean over variables
        
        return kl


class HierarchicalWorldModel(nn.Module):
    """
    Hierarchical world model with multiple levels of abstraction.
    Captures both local (cell-level) and global (room-level) dynamics.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        action_size: int = 5,
        encoder_depths: List[int] = [32, 64, 128, 256],
        decoder_depths: List[int] = [256, 128, 64, 32],
        rssm_config: Optional[Dict] = None,
    ):
        """
        Initialize hierarchical world model.
        
        Args:
            observation_shape: Shape of observations (H, W, C)
            action_size: Number of actions
            encoder_depths: Channel depths for encoder layers
            decoder_depths: Channel depths for decoder layers
            rssm_config: Configuration for RSSM
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.action_size = action_size
        
        # Default RSSM config
        if rssm_config is None:
            rssm_config = {
                "deterministic_size": 512,
                "stochastic_size": 32,
                "categories": 32,
            }
        
        # Image encoder
        self.encoder = ConvEncoder(
            input_shape=observation_shape,
            depths=encoder_depths,
        )
        
        # Get encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            encoder_output = self.encoder(dummy_input)
            self.obs_embed_size = encoder_output.shape[-1]
        
        # RSSM
        self.rssm = CategoricalRSSM(
            obs_embed_size=self.obs_embed_size,
            action_size=action_size,
            **rssm_config,
        )
        
        # Get feature size from RSSM
        with torch.no_grad():
            dummy_state = self.rssm.initial_state(1, torch.device("cpu"))
            self.feature_size = self.rssm.get_features(dummy_state).shape[-1]
        
        # Image decoder
        self.decoder = ConvDecoder(
            feature_size=self.feature_size,
            output_shape=observation_shape,
            depths=decoder_depths,
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Terminal predictor
        self.terminal_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
        imagine: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.
        
        Args:
            observations: Batch of observations (B, T, C, H, W)
            actions: Batch of actions (B, T, A)
            initial_state: Initial RSSM state
            imagine: Whether to use imagination (no observations)
            
        Returns:
            Dictionary containing predictions and states
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        
        # Initialize state if not provided
        if initial_state is None:
            state = self.rssm.initial_state(batch_size, device)
        else:
            state = initial_state
        
        # Storage for outputs
        states = []
        reconstructions = []
        rewards = []
        terminals = []
        posterior_dists = []
        prior_dists = []
        
        for t in range(seq_len):
            # Get observation and action at time t
            obs_t = observations[:, t]
            action_t = actions[:, t]
            
            if not imagine:
                # Encode observation
                obs_embed = self.encoder(obs_t)
                
                # Update state with observation (posterior)
                state, posterior_dist = self.rssm.observe(
                    obs_embed, action_t, state, sample=True
                )
                posterior_dists.append(posterior_dist)
                
                # Also compute prior for KL divergence
                _, prior_dist = self.rssm.imagine(
                    action_t, state, sample=False
                )
                prior_dists.append(prior_dist)
            else:
                # Imagine next state without observation (prior)
                state, prior_dist = self.rssm.imagine(
                    action_t, state, sample=True
                )
                prior_dists.append(prior_dist)
            
            # Store state
            states.append(state)
            
            # Get features from state
            features = self.rssm.get_features(state)
            
            # Decode to reconstruction
            recon = self.decoder(features)
            reconstructions.append(recon)
            
            # Predict reward and terminal
            reward = self.reward_predictor(features)
            terminal = self.terminal_predictor(features)
            
            rewards.append(reward)
            terminals.append(terminal)
        
        # Stack outputs
        outputs = {
            "reconstructions": torch.stack(reconstructions, dim=1),
            "rewards": torch.stack(rewards, dim=1).squeeze(-1),
            "terminals": torch.sigmoid(torch.stack(terminals, dim=1).squeeze(-1)),
            "states": self._stack_states(states),
        }
        
        if not imagine:
            outputs["posterior_dists"] = self._stack_dists(posterior_dists)
            outputs["prior_dists"] = self._stack_dists(prior_dists)
        else:
            outputs["prior_dists"] = self._stack_dists(prior_dists)
        
        return outputs
    
    def imagine_trajectory(
        self,
        initial_obs: torch.Tensor,
        action_sequence: torch.Tensor,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine a trajectory given initial observation and action sequence.
        
        Args:
            initial_obs: Initial observation (B, C, H, W)
            action_sequence: Sequence of actions (B, T, A)
            initial_state: Initial RSSM state
            
        Returns:
            Imagined trajectory
        """
        batch_size = initial_obs.shape[0]
        device = initial_obs.device
        
        # Initialize or use provided state
        if initial_state is None:
            # Encode initial observation
            obs_embed = self.encoder(initial_obs)
            
            # Create initial state
            state = self.rssm.initial_state(batch_size, device)
            
            # Update with observation
            dummy_action = torch.zeros(batch_size, self.action_size, device=device)
            state, _ = self.rssm.observe(obs_embed, dummy_action, state)
        else:
            state = initial_state
        
        # Imagine trajectory
        trajectory = {
            "observations": [initial_obs],
            "rewards": [],
            "terminals": [],
            "states": [state],
        }
        
        for t in range(action_sequence.shape[1]):
            action_t = action_sequence[:, t]
            
            # Imagine next state
            state, _ = self.rssm.imagine(action_t, state)
            
            # Get features
            features = self.rssm.get_features(state)
            
            # Decode observation
            obs = self.decoder(features)
            
            # Predict reward and terminal
            reward = self.reward_predictor(features).squeeze(-1)
            terminal = torch.sigmoid(self.terminal_predictor(features).squeeze(-1))
            
            trajectory["observations"].append(obs)
            trajectory["rewards"].append(reward)
            trajectory["terminals"].append(terminal)
            trajectory["states"].append(state)
        
        # Stack outputs
        trajectory["observations"] = torch.stack(trajectory["observations"], dim=1)
        trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=1)
        trajectory["terminals"] = torch.stack(trajectory["terminals"], dim=1)
        trajectory["states"] = self._stack_states(trajectory["states"])
        
        return trajectory
    
    def _stack_states(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack list of state dicts into single dict of stacked tensors."""
        stacked = {}
        for key in states[0].keys():
            stacked[key] = torch.stack([s[key] for s in states], dim=1)
        return stacked
    
    def _stack_dists(self, dists: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack list of distribution dicts."""
        stacked = {}
        for key in dists[0].keys():
            stacked[key] = torch.stack([d[key] for d in dists], dim=1)
        return stacked


class ConvEncoder(nn.Module):
    """Convolutional encoder for image observations."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        depths: List[int] = [32, 64, 128, 256],
        kernel_size: int = 3,
    ):
        super().__init__()
        
        channels, height, width = input_shape
        
        layers = []
        in_channels = channels
        
        for depth in depths:
            layers.extend([
                nn.Conv2d(in_channels, depth, kernel_size, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(depth),
            ])
            in_channels = depth
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            conv_output = self.conv_layers(dummy_input)
            self.output_size = conv_output.view(1, -1).shape[1]
        
        # Final linear layer
        self.fc = nn.Linear(self.output_size, depths[-1])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding."""
        # Ensure correct shape (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten and apply final linear
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x


class ConvDecoder(nn.Module):
    """Convolutional decoder for image reconstruction."""
    
    def __init__(
        self,
        feature_size: int,
        output_shape: Tuple[int, int, int],
        depths: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.output_shape = output_shape
        channels, height, width = output_shape
        
        # Initial reshape dimensions
        self.initial_height = height // (2 ** len(depths))
        self.initial_width = width // (2 ** len(depths))
        self.initial_channels = depths[0]
        
        # Initial linear layer
        self.fc = nn.Linear(
            feature_size,
            self.initial_channels * self.initial_height * self.initial_width
        )
        
        # Deconvolutional layers
        layers = []
        in_channels = self.initial_channels
        
        for i, depth in enumerate(depths[1:]):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels, depth, kernel_size,
                    stride=2, padding=1, output_padding=1
                ),
                nn.ReLU(),
                nn.BatchNorm2d(depth),
            ])
            in_channels = depth
        
        # Final layer to match output channels
        layers.append(
            nn.ConvTranspose2d(
                in_channels, channels, kernel_size,
                stride=2, padding=1, output_padding=1
            )
        )
        
        self.deconv_layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features to image."""
        # Linear transformation
        x = self.fc(x)
        
        # Reshape to initial conv shape
        x = x.view(
            x.shape[0],
            self.initial_channels,
            self.initial_height,
            self.initial_width
        )
        
        # Apply deconvolutional layers
        x = self.deconv_layers(x)
        
        # Ensure output is correct size
        if x.shape[2:] != self.output_shape[1:]:
            x = F.interpolate(
                x,
                size=self.output_shape[1:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply sigmoid to get pixel values in [0, 1]
        x = torch.sigmoid(x)
        
        # Convert to uint8 range
        x = x * 255.0
        
        return x
