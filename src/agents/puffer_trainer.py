"""
PPO Trainer - Complete Implementation
Training loop with GAE, PPO clipping, and logging
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
import time
from pathlib import Path
from collections import deque

from .puffer_policy import ActorCriticPolicy
from ..utils.logger import TrainingLogger


class RolloutBuffer:
    """Buffer for storing rollout data during training"""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Initialize buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done):
        """Add transition to buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True

    def get(self):
        """Get all data from buffer"""
        size = self.buffer_size if self.full else self.ptr
        return {
            'observations': self.observations[:size],
            'actions': self.actions[:size],
            'rewards': self.rewards[:size],
            'values': self.values[:size],
            'log_probs': self.log_probs[:size],
            'dones': self.dones[:size]
        }

    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.full = False


class PPOTrainer:
    """
    PPO trainer with GAE and multiple parallel environments

    Features:
    - Generalized Advantage Estimation (GAE)
    - PPO clipping for stable updates
    - Multiple parallel environments
    - Logging and checkpointing
    - Learning rate scheduling
    """

    def __init__(self, envs: List, config: Dict):
        """
        Initialize PPO trainer

        Args:
            envs: List of Gymnasium environments
            config: Training configuration dict
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.config = config

        # Extract key parameters
        ppo_config = config['ppo']
        self.learning_rate = ppo_config['learning_rate']
        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['gae_lambda']
        self.clip_range = ppo_config['clip_range']
        self.value_coef = ppo_config['value_coef']
        self.entropy_coef = ppo_config['ent_coef']
        self.max_grad_norm = ppo_config['max_grad_norm']
        self.n_epochs = ppo_config['n_epochs']
        self.batch_size = ppo_config['batch_size']

        # Training schedule
        schedule_config = config['training']['schedule']
        self.rollout_length = schedule_config['rollout_length']
        self.eval_freq = schedule_config['eval_freq']
        self.save_freq = schedule_config['save_freq']

        # Get environment dimensions
        env = envs[0]
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Create policy
        network_config = config['network']
        self.policy = ActorCriticPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=network_config['hidden_sizes'],
            activation=network_config['activation']
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate
        )

        # Create rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_length * self.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim
        )

        # Initialize logger
        log_dir = Path(config['logging']['log_dir'])
        self.logger = TrainingLogger(
            log_dir=str(log_dir),
            use_tensorboard=config['logging']['use_tensorboard'],
            use_wandb=config['logging']['use_wandb']
        )

        # Training state
        self.total_timesteps = 0
        self.num_updates = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # Checkpoint directory
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"PPOTrainer initialized")
        print(f"  Num environments: {self.num_envs}")
        print(f"  Rollout length: {self.rollout_length}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")

    def train(self, total_timesteps: int, resume_from: Optional[str] = None):
        """
        Main training loop

        Args:
            total_timesteps: Total number of environment steps to train for
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        print(f"\nStarting training for {total_timesteps:,} timesteps...")
        start_time = time.time()

        # Reset all environments
        observations = [env.reset()[0] for env in self.envs]
        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs

        num_iterations = total_timesteps // (self.rollout_length * self.num_envs)

        for iteration in range(num_iterations):
            # Collect rollouts
            rollout_start = time.time()

            for step in range(self.rollout_length):
                # Get actions from policy
                actions = []
                values_list = []
                log_probs_list = []

                for obs in observations:
                    action, log_prob, value = self.policy.get_action(obs, deterministic=False)
                    actions.append(action)
                    values_list.append(value.item())
                    log_probs_list.append(log_prob.item())

                # Step environments
                next_observations = []
                for env_idx, (env, action) in enumerate(zip(self.envs, actions)):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Add to buffer
                    self.buffer.add(
                        observations[env_idx],
                        action,
                        reward,
                        values_list[env_idx],
                        log_probs_list[env_idx],
                        done
                    )

                    # Track episode stats
                    episode_rewards[env_idx] += reward
                    episode_lengths[env_idx] += 1

                    # Handle episode end
                    if done:
                        self.episode_rewards.append(episode_rewards[env_idx])
                        self.episode_lengths.append(episode_lengths[env_idx])
                        episode_rewards[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        obs, _ = env.reset()

                    next_observations.append(obs)

                observations = next_observations
                self.total_timesteps += self.num_envs

            rollout_time = time.time() - rollout_start

            # Update policy
            update_start = time.time()
            train_stats = self._update_policy()
            update_time = time.time() - update_start

            self.num_updates += 1

            # Logging
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)

                self.logger.log_scalars({
                    'train/episode_reward': mean_reward,
                    'train/episode_length': mean_length,
                    'train/policy_loss': train_stats['policy_loss'],
                    'train/value_loss': train_stats['value_loss'],
                    'train/entropy': train_stats['entropy'],
                    'train/clip_fraction': train_stats['clip_fraction'],
                    'train/learning_rate': self.learning_rate,
                    'time/rollout': rollout_time,
                    'time/update': update_time,
                }, self.total_timesteps)

            # Print progress
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed
                print(f"Iter {iteration}/{num_iterations} | "
                      f"Steps: {self.total_timesteps:,} | "
                      f"Reward: {mean_reward:.2f} | "
                      f"FPS: {fps:.0f}")

            # Evaluation
            if self.total_timesteps % self.eval_freq == 0:
                eval_stats = self._evaluate()
                self.logger.log_scalars(eval_stats, self.total_timesteps)

            # Save checkpoint
            if self.total_timesteps % self.save_freq == 0:
                self.save_checkpoint(f"checkpoint_{self.total_timesteps}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")

    def _update_policy(self) -> Dict:
        """
        Perform PPO policy update

        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        data = self.buffer.get()
        buffer_size = len(data['observations'])

        # Convert to tensors
        obs = torch.FloatTensor(data['observations'])
        actions = torch.FloatTensor(data['actions'])
        old_log_probs = torch.FloatTensor(data['log_probs'])
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_batches = 0

        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(buffer_size)

            # Mini-batch updates
            for start in range(0, buffer_size, self.batch_size):
                end = min(start + self.batch_size, buffer_size)
                batch_indices = indices[start:end]

                # Get batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions under current policy
                log_probs, entropy, values_pred = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )

                # Compute ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(log_probs.squeeze() - batch_old_log_probs)

                # Compute clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                # Clip fraction (how often we're clipping)
                clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                total_clip_fraction += clip_fraction

                num_batches += 1

        # Clear buffer
        self.buffer.clear()

        # Return statistics
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy': total_entropy / num_batches,
            'clip_fraction': total_clip_fraction / num_batches
        }

    def _compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: Array of rewards [T]
            values: Array of value estimates [T]
            dones: Array of done flags [T]

        Returns:
            advantages: GAE advantages [T]
            returns: Value targets [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0
        next_value = 0  # Assumes episode ends with value 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # Bootstrap value (0 if episode ends)
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate current policy

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(num_episodes):
            obs, _ = self.envs[0].reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done and episode_length < 1000:
                action, _, _ = self.policy.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.envs[0].step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        return {
            'eval/mean_reward': np.mean(eval_rewards),
            'eval/std_reward': np.std(eval_rewards),
            'eval/mean_length': np.mean(eval_lengths)
        }

    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        filepath = self.checkpoint_dir / filename
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'config': self.config
        }, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        print(f"Checkpoint loaded from: {filepath}")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("PPOTrainer requires environment setup - see scripts/train.py for usage")
