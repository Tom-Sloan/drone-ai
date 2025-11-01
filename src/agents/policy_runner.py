"""
Policy Runner for Inference - Complete Implementation
Loads and runs trained policies for deployment
"""

import torch
import numpy as np
from typing import Optional
from pathlib import Path

from .puffer_policy import ActorCriticPolicy
from ..rl.state import StateNormalizer


class PolicyRunner:
    """
    Wrapper for running trained policy

    Usage:
        runner = PolicyRunner("path/to/checkpoint.pt")
        action = runner.get_action(state, deterministic=True)
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Load trained policy from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file (.pt)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Load checkpoint
        print(f"Loading policy from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config
        self.config = checkpoint['config']
        network_config = self.config['network']

        # Create policy network
        self.policy = ActorCriticPolicy(
            obs_dim=14,  # Drone observation dimension
            action_dim=4,  # Drone action dimension
            hidden_sizes=network_config['hidden_sizes'],
            activation=network_config['activation']
        )

        # Load weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()  # Set to evaluation mode
        self.policy.to(device)

        # Create state normalizer
        self.normalizer = StateNormalizer()

        # Training metadata
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.num_updates = checkpoint.get('num_updates', 0)

        print(f"Policy loaded successfully!")
        print(f"  Training timesteps: {self.total_timesteps:,}")
        print(f"  Training updates: {self.num_updates:,}")
        print(f"  Device: {device}")

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get action from policy

        Args:
            state: State array (14D for drone)
                   Can be raw state or already normalized
            deterministic: If True, return mean action.
                          If False, sample from distribution.

        Returns:
            action: Action array [throttle, roll, pitch, yaw] in [-1, 1]
        """
        # Normalize state if it's not already normalized
        # (Check if values are mostly in [-1, 1] range)
        if np.abs(state).max() > 2.0:
            state = self.normalizer.normalize(state)

        # Get action from policy
        with torch.no_grad():
            action, _, _ = self.policy.get_action(state, deterministic=deterministic)

        return action

    def get_action_and_value(self, state: np.ndarray, deterministic: bool = True):
        """
        Get action and value estimate from policy

        Args:
            state: State array (14D for drone)
            deterministic: If True, return mean action

        Returns:
            action: Action array in [-1, 1]
            value: Value estimate (scalar)
        """
        # Normalize state if needed
        if np.abs(state).max() > 2.0:
            state = self.normalizer.normalize(state)

        # Get action and value
        with torch.no_grad():
            action, _, value = self.policy.get_action(state, deterministic=deterministic)

        return action, value.item()

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """
        Alias for get_action (compatible with stable-baselines3 interface)

        Args:
            observation: Observation array
            deterministic: If True, return deterministic action

        Returns:
            action: Action array
            state: None (for compatibility)
        """
        action = self.get_action(observation, deterministic=deterministic)
        return action, None


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PolicyRunner Test")
    print("=" * 70)

    # This test requires a trained checkpoint
    checkpoint_path = "data/checkpoints/final_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train a policy first using scripts/train.py")
        print("\nCreating a dummy checkpoint for testing...")

        # Create dummy checkpoint for testing
        from .puffer_policy import ActorCriticPolicy
        import yaml

        # Load config
        config_path = Path(__file__).parent.parent.parent / "configs/training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create policy
        policy = ActorCriticPolicy(obs_dim=14, action_dim=4)

        # Save dummy checkpoint
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'total_timesteps': 0,
            'num_updates': 0,
            'config': config
        }, checkpoint_path)

        print(f"Dummy checkpoint created at: {checkpoint_path}")

    # Load policy
    runner = PolicyRunner(checkpoint_path)

    print("\n1. Testing with random state...")
    state = np.random.randn(14)
    action = runner.get_action(state, deterministic=True)
    print(f"  State (first 4): {state[:4]}")
    print(f"  Action: {action}")
    print(f"  Action bounds OK: {np.all((action >= -1) & (action <= 1))}")

    print("\n2. Testing with stochastic action...")
    action = runner.get_action(state, deterministic=False)
    print(f"  Action: {action}")

    print("\n3. Testing get_action_and_value...")
    action, value = runner.get_action_and_value(state)
    print(f"  Action: {action}")
    print(f"  Value: {value:.3f}")

    print("\n4. Testing predict (SB3 interface)...")
    action, _ = runner.predict(state, deterministic=True)
    print(f"  Action: {action}")

    print("\n5. Testing with batch of states...")
    num_states = 100
    states = np.random.randn(num_states, 14)
    actions = np.array([runner.get_action(s) for s in states])
    print(f"  Processed {num_states} states")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions min: {actions.min(axis=0)}")
    print(f"  Actions max: {actions.max(axis=0)}")
    print(f"  All actions in bounds: {np.all((actions >= -1) & (actions <= 1))}")

    print("\nTest complete!")
