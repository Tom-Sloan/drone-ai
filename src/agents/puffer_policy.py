"""
Neural Network Policy for PPO - Complete Implementation
Actor-Critic architecture with continuous action space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic network for PPO

    Architecture:
      Input (14) -> Encoder [128, 128] -> Actor (mean, log_std)
                                        -> Critic (value)

    Actor outputs: Gaussian policy (mean and log_std for each action)
    Critic outputs: State value estimate
    """

    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 4,
        hidden_sizes: list = [128, 128],
        activation: str = 'tanh',
        init_log_std: float = 0.0,
    ):
        """
        Build Actor-Critic network

        Args:
            obs_dim: Observation dimension (14 for drone)
            action_dim: Action dimension (4 for drone: throttle, roll, pitch, yaw)
            hidden_sizes: Hidden layer sizes for encoder
            activation: Activation function ('tanh', 'relu', 'elu')
            init_log_std: Initial log standard deviation for actions
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Select activation function
        if activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'elu':
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Shared encoder network
        encoder_layers = []
        prev_size = obs_dim

        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(self.activation())
            prev_size = hidden_size

        self.encoder = nn.Sequential(*encoder_layers)

        # Actor head (policy): outputs action mean
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)

        # Actor log_std: learnable parameter (shared across all observations)
        self.actor_logstd = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # Critic head (value function): outputs scalar value
        self.critic = nn.Linear(hidden_sizes[-1], 1)

        # Initialize weights
        self._init_weights()

        print(f"ActorCriticPolicy initialized")
        print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
        print(f"  Architecture: {obs_dim} -> {' -> '.join(map(str, hidden_sizes))} -> Actor/Critic")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        """Initialize network weights"""
        # Orthogonal initialization for better gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for actor mean (more stable at start)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

        # Smaller initialization for critic (more stable value estimates)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network

        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]

        Returns:
            action_mean: Mean of action distribution [batch_size, action_dim]
            action_logstd: Log std of action distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        # Handle single observation (add batch dimension)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Shared encoder
        features = self.encoder(obs)

        # Actor: mean of Gaussian policy
        mean = torch.tanh(self.actor_mean(features))  # Bound to [-1, 1]

        # Actor: log standard deviation (broadcasted to batch size)
        logstd = self.actor_logstd.expand_as(mean)

        # Critic: state value estimate
        value = self.critic(features)

        return mean, logstd, value

    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample action from policy

        Args:
            obs: Observation array [obs_dim] or [batch_size, obs_dim]
            deterministic: If True, return mean action. If False, sample from distribution.

        Returns:
            action: Action array [action_dim] or [batch_size, action_dim]
            log_prob: Log probability of action (None if deterministic)
            value: Value estimate
        """
        with torch.no_grad():
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs)

            # Forward pass
            mean, logstd, value = self.forward(obs_tensor)

            if deterministic:
                # Return mean action (no sampling)
                action = mean
                log_prob = None
            else:
                # Sample from Gaussian distribution
                std = torch.exp(logstd)
                dist = Normal(mean, std)
                action = dist.sample()

                # Compute log probability
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

            # Convert to numpy
            action_np = action.squeeze().cpu().numpy()

            return action_np, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy
        Used during PPO training to compute log probs and entropy

        Args:
            obs: Observation tensor [batch_size, obs_dim]
            actions: Action tensor [batch_size, action_dim]

        Returns:
            log_probs: Log probabilities [batch_size, 1]
            entropy: Entropy of distribution [batch_size, 1]
            values: Value estimates [batch_size, 1]
        """
        # Forward pass
        mean, logstd, values = self.forward(obs)

        # Create distribution
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        # Compute log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for observation

        Args:
            obs: Observation tensor [batch_size, obs_dim]

        Returns:
            value: Value estimate [batch_size, 1]
        """
        _, _, value = self.forward(obs)
        return value


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ActorCriticPolicy Test")
    print("=" * 70)

    # Create policy
    policy = ActorCriticPolicy(
        obs_dim=14,
        action_dim=4,
        hidden_sizes=[128, 128],
        activation='tanh'
    )

    print("\n1. Testing single observation forward pass...")
    obs = torch.randn(14)
    mean, logstd, value = policy(obs)
    print(f"  Input shape: {obs.shape}")
    print(f"  Action mean shape: {mean.shape}")
    print(f"  Action logstd shape: {logstd.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Action mean: {mean.squeeze().detach().numpy()}")
    print(f"  Value: {value.item():.3f}")

    print("\n2. Testing batch forward pass...")
    obs_batch = torch.randn(32, 14)
    mean, logstd, value = policy(obs_batch)
    print(f"  Input shape: {obs_batch.shape}")
    print(f"  Action mean shape: {mean.shape}")
    print(f"  Value shape: {value.shape}")

    print("\n3. Testing get_action (deterministic)...")
    obs_np = np.random.randn(14)
    action, log_prob, value = policy.get_action(obs_np, deterministic=True)
    print(f"  Observation: {obs_np[:4]}...")
    print(f"  Action: {action}")
    print(f"  Value: {value.item():.3f}")

    print("\n4. Testing get_action (stochastic)...")
    action, log_prob, value = policy.get_action(obs_np, deterministic=False)
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob.item():.3f}")
    print(f"  Value: {value.item():.3f}")

    print("\n5. Testing evaluate_actions...")
    obs_batch = torch.randn(32, 14)
    actions_batch = torch.randn(32, 4)
    log_probs, entropy, values = policy.evaluate_actions(obs_batch, actions_batch)
    print(f"  Batch size: 32")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Entropy shape: {entropy.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Mean log prob: {log_probs.mean().item():.3f}")
    print(f"  Mean entropy: {entropy.mean().item():.3f}")
    print(f"  Mean value: {values.mean().item():.3f}")

    print("\n6. Testing action bounds (should be in [-1, 1])...")
    num_samples = 1000
    actions = []
    for _ in range(num_samples):
        obs_np = np.random.randn(14)
        action, _, _ = policy.get_action(obs_np, deterministic=True)
        actions.append(action)

    actions = np.array(actions)
    print(f"  Sampled {num_samples} actions")
    print(f"  Action min: {actions.min(axis=0)}")
    print(f"  Action max: {actions.max(axis=0)}")
    print(f"  Action mean: {actions.mean(axis=0)}")
    print(f"  All actions in [-1, 1]: {np.all((actions >= -1) & (actions <= 1))}")

    print("\n7. Testing gradient flow...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Dummy training step
    obs = torch.randn(32, 14)
    actions = torch.randn(32, 4)
    returns = torch.randn(32, 1)

    # Forward pass
    log_probs, entropy, values = policy.evaluate_actions(obs, actions)

    # Compute loss (simplified)
    value_loss = F.mse_loss(values, returns)
    policy_loss = -(log_probs * returns).mean()
    entropy_loss = -entropy.mean()

    total_loss = value_loss + policy_loss + 0.01 * entropy_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"  Value loss: {value_loss.item():.3f}")
    print(f"  Policy loss: {policy_loss.item():.3f}")
    print(f"  Entropy loss: {entropy_loss.item():.3f}")
    print(f"  Total loss: {total_loss.item():.3f}")
    print(f"  Gradients computed successfully!")

    print("\nTest complete!")
