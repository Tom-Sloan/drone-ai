"""
Gymnasium Environment for Drone RL - Complete Implementation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

from .drone_sim import DroneSimulator
from .msp_interface import SimDroneInterface, RealDroneInterface
from .state import DroneState, StateNormalizer
from .rewards import calculate_reward, RewardConfig, check_crash


class DroneHoverEnv(gym.Env):
    """
    Gymnasium environment for drone hover stabilization

    Observation: [roll, pitch, yaw_rate, vx, vy, vz, x, y, z, battery, prev_actions...] (14D)
    Action: [throttle, roll, pitch, yaw] continuous in [-1, 1] (4D)
    Reward: Hover stability + energy efficiency
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        config_path: Optional[str] = None,
        sim_config_path: Optional[str] = None,
        use_sim: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize environment

        Args:
            config_path: Path to env_config.yaml (optional)
            sim_config_path: Path to sim_config.yaml (optional)
            use_sim: If True, use simulation. If False, use real drone
            render_mode: 'human' for GUI, 'rgb_array' for numpy arrays
        """
        super().__init__()

        # Load configurations
        self.env_config = self._load_config(config_path, 'env_config.yaml')
        self.sim_config = self._load_config(sim_config_path, 'sim_config.yaml')

        # Extract task configuration
        self.task_config = self.env_config['task']
        self.task_type = self.task_config['type']
        self.target_position = np.array(self.task_config['target_position'])

        # Episode limits
        self.max_steps = self.env_config['simulation']['max_episode_steps']
        self.bounds = self.env_config['simulation']['boundaries']

        # Success criteria
        self.success_criteria = self.env_config['success_criteria']

        # Render mode
        self.render_mode = render_mode

        # Observation space: 14D continuous normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # Action space: 4D continuous (throttle, roll, pitch, yaw)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Initialize backend (simulation or real drone)
        self.use_sim = use_sim
        if use_sim:
            # Create simulator with GUI if render_mode is 'human'
            gui_enabled = (render_mode == 'human')
            simulator = DroneSimulator(config=self.sim_config, gui=gui_enabled)
            self.backend = SimDroneInterface(simulator)
        else:
            # TODO: Initialize real drone interface
            # Requires serial connection, MSP parser, and control loop
            raise NotImplementedError("Real drone interface not yet implemented")

        # Setup normalizer and reward config
        self.normalizer = StateNormalizer()
        self.reward_config = RewardConfig(
            position_weight=self.env_config['rewards']['position_weight'],
            velocity_weight=self.env_config['rewards']['velocity_weight'],
            attitude_weight=self.env_config['rewards']['attitude_weight'],
            energy_weight=self.env_config['rewards']['energy_weight']
        )

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.success_count = 0

        print(f"DroneHoverEnv initialized")
        print(f"  Task: {self.task_type}")
        print(f"  Target position: {self.target_position}")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Backend: {'Simulation' if use_sim else 'Real Drone'}")

    def _load_config(self, config_path: Optional[str], default_name: str) -> Dict:
        """Load configuration file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / default_name

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state

        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., target_position override)

        Returns:
            observation: Normalized 14D state vector
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Reset backend (simulator or real drone)
        state = self.backend.reset()

        # Reset episode counters
        self.current_step = 0
        self.episode_reward = 0.0

        # Override target position if provided in options
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'])

        # Normalize observation
        obs = self.normalizer.normalize(state.to_array())

        # Info dictionary
        info = {
            'state': state,
            'target_position': self.target_position.copy(),
            'episode_step': self.current_step
        }

        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return new state

        Args:
            action: [throttle, roll, pitch, yaw] in [-1, 1]

        Returns:
            observation: Normalized 14D state vector
            reward: Scalar reward value
            terminated: True if episode ends (crash or success)
            truncated: True if max steps reached
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Execute action on backend
        next_state = self.backend.step(action)

        # Calculate reward
        reward_info = calculate_reward(
            state=next_state,
            action=action,
            target_position=self.target_position,
            config=self.reward_config
        )

        reward = reward_info['total_reward']

        # Check termination conditions
        is_crash = check_crash(next_state, self.bounds)
        is_success = self._check_success(next_state, reward_info)

        terminated = is_crash or is_success
        truncated = self.current_step >= self.max_steps

        # Update episode state
        self.current_step += 1
        self.episode_reward += reward

        if is_success:
            self.success_count += 1

        # Normalize observation
        obs = self.normalizer.normalize(next_state.to_array())

        # Info dictionary
        info = {
            'state': next_state,
            'target_position': self.target_position.copy(),
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'is_crash': is_crash,
            'is_success': is_success,
            'position_error': reward_info['position_error'],
            'velocity_magnitude': reward_info['velocity_magnitude'],
            'attitude_error': reward_info['attitude_error'],
            'reward_components': {
                'position': reward_info['position_reward'],
                'velocity': reward_info['velocity_reward'],
                'attitude': reward_info['attitude_reward'],
                'energy': reward_info['energy_penalty']
            }
        }

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _check_success(self, state: DroneState, reward_info: Dict) -> bool:
        """
        Check if drone has successfully completed the task

        Args:
            state: Current drone state
            reward_info: Reward calculation info

        Returns:
            True if success criteria met
        """
        # Position within tolerance
        position_ok = reward_info['position_error'] < self.success_criteria['position_tolerance']

        # Velocity within tolerance
        velocity_ok = reward_info['velocity_magnitude'] < self.success_criteria['velocity_tolerance']

        # Attitude within tolerance
        attitude_ok = reward_info['attitude_error'] < np.deg2rad(
            self.success_criteria['attitude_tolerance']
        )

        # Must maintain for minimum duration
        # (Note: This is simplified - full implementation would track consecutive success steps)
        success = position_ok and velocity_ok and attitude_ok

        return success

    def render(self):
        """
        Render environment

        Returns:
            RGB image if render_mode='rgb_array', None otherwise
        """
        if self.render_mode == 'rgb_array' and self.use_sim:
            # Get camera image from simulator
            return self.backend.simulator.get_camera_image()
        elif self.render_mode == 'human':
            # GUI is already rendering (PyBullet GUI mode)
            pass
        else:
            return None

    def close(self):
        """Clean up environment"""
        if hasattr(self, 'backend'):
            self.backend.close()
        print("DroneHoverEnv closed")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DroneHoverEnv Test")
    print("=" * 70)

    # Create environment with rendering
    env = DroneHoverEnv(render_mode='human')

    print("\nTesting with random policy...")

    # Run a few episodes
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")

        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 500:
            # Random action (with bias toward hover)
            # Hover throttle is around 0.3 for our drone
            action = np.random.uniform(-0.2, 0.2, 4)
            action[0] += 0.3  # Add hover throttle bias

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1
            done = terminated or truncated

            # Print progress every 50 steps
            if step % 50 == 0:
                state = info['state']
                print(f"  Step {step}:")
                print(f"    Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
                print(f"    Position error: {info['position_error']:.3f}m")
                print(f"    Reward: {reward:.3f}")

        print(f"\nEpisode finished:")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Crash: {info['is_crash']}")
        print(f"  Success: {info['is_success']}")

    env.close()
    print("\nTest complete!")
