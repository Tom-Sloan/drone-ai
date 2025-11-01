"""
Reward Functions for Drone RL
Defines reward shaping for hover stabilization and other tasks
"""

import numpy as np
from typing import Dict, Optional
from .state import DroneState


class RewardConfig:
    """Configuration for reward function weights"""

    def __init__(
        self,
        position_weight: float = 10.0,
        velocity_weight: float = 5.0,
        attitude_weight: float = 3.0,
        energy_weight: float = -0.001,
        crash_penalty: float = -100.0,
        success_bonus: float = 10.0,
        position_tolerance: float = 0.1,  # meters
        velocity_tolerance: float = 0.2,  # m/s
        attitude_tolerance: float = 5.0,  # degrees
    ):
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.energy_weight = energy_weight
        self.crash_penalty = crash_penalty
        self.success_bonus = success_bonus
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.attitude_tolerance = attitude_tolerance


def hover_reward(
    state: DroneState,
    action: np.ndarray,
    target_position: np.ndarray = np.array([0.0, 0.0, 0.5]),
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute reward for hover stabilization task

    The drone should:
    1. Maintain position at target_position
    2. Have zero velocity
    3. Stay level (roll and pitch near zero)
    4. Minimize energy consumption

    Args:
        state: Current drone state
        action: Action taken [throttle, roll, pitch, yaw] in [-1, 1]
        target_position: Target hover position [x, y, z] in meters
        config: Reward configuration

    Returns:
        Scalar reward value
    """
    if config is None:
        config = RewardConfig()

    # Position error from target
    pos_error = np.linalg.norm(state.position - target_position)
    position_reward = np.exp(-5.0 * pos_error)  # Exponential decay

    # Velocity (should be near zero for hover)
    vel_magnitude = np.linalg.norm(state.velocity)
    velocity_reward = np.exp(-2.0 * vel_magnitude)

    # Attitude stability (roll and pitch should be small)
    roll_error = np.abs(state.attitude[0])  # degrees
    pitch_error = np.abs(state.attitude[1])  # degrees
    attitude_error = np.deg2rad(roll_error + pitch_error)
    attitude_reward = np.exp(-2.0 * attitude_error)

    # Energy penalty (encourage efficient control)
    # Penalize large deviations from hover throttle (~0.3 normalized)
    throttle_deviation = np.abs(action[0] - 0.3)
    control_effort = np.sum(np.abs(action[1:]))  # roll, pitch, yaw effort
    energy_penalty = throttle_deviation + 0.5 * control_effort

    # Combine weighted rewards
    total_reward = (
        config.position_weight * position_reward
        + config.velocity_weight * velocity_reward
        + config.attitude_weight * attitude_reward
        + config.energy_weight * energy_penalty
    )

    return float(total_reward)


def check_success(
    state: DroneState,
    target_position: np.ndarray = np.array([0.0, 0.0, 0.5]),
    config: Optional[RewardConfig] = None,
) -> bool:
    """
    Check if drone is successfully hovering at target

    Args:
        state: Current drone state
        target_position: Target hover position
        config: Reward configuration

    Returns:
        True if hovering within tolerance
    """
    if config is None:
        config = RewardConfig()

    # Check position tolerance
    pos_error = np.linalg.norm(state.position - target_position)
    if pos_error > config.position_tolerance:
        return False

    # Check velocity tolerance
    vel_magnitude = np.linalg.norm(state.velocity)
    if vel_magnitude > config.velocity_tolerance:
        return False

    # Check attitude tolerance
    roll_error = np.abs(state.attitude[0])
    pitch_error = np.abs(state.attitude[1])
    if roll_error > config.attitude_tolerance or pitch_error > config.attitude_tolerance:
        return False

    return True


def check_crash(state: DroneState, bounds: Optional[Dict[str, float]] = None) -> bool:
    """
    Check if drone has crashed or gone out of bounds

    Args:
        state: Current drone state
        bounds: Dictionary with boundary limits

    Returns:
        True if crashed
    """
    if bounds is None:
        bounds = {
            'x_min': -2.0, 'x_max': 2.0,
            'y_min': -2.0, 'y_max': 2.0,
            'z_min': 0.05, 'z_max': 2.0,  # Minimum height to avoid ground collision
            'roll_max': 60.0,  # degrees
            'pitch_max': 60.0,  # degrees
        }

    # Check height (z-axis)
    if state.position[2] < bounds['z_min']:
        return True  # Ground collision

    if state.position[2] > bounds['z_max']:
        return True  # Flew too high

    # Check horizontal boundaries
    if not (bounds['x_min'] <= state.position[0] <= bounds['x_max']):
        return True

    if not (bounds['y_min'] <= state.position[1] <= bounds['y_max']):
        return True

    # Check extreme attitudes
    if np.abs(state.attitude[0]) > bounds['roll_max']:
        return True  # Extreme roll

    if np.abs(state.attitude[1]) > bounds['pitch_max']:
        return True  # Extreme pitch

    # Check battery voltage (safety cutoff)
    if state.battery_voltage < 3.0:
        return True  # Battery too low

    return False


def calculate_reward(
    state: DroneState,
    action: np.ndarray,
    next_state: DroneState,
    target_position: np.ndarray = np.array([0.0, 0.0, 0.5]),
    config: Optional[RewardConfig] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive reward with breakdown

    Args:
        state: Current state
        action: Action taken
        next_state: Resulting state
        target_position: Target hover position
        config: Reward configuration

    Returns:
        Dictionary with reward components and total
    """
    if config is None:
        config = RewardConfig()

    # Base hover reward
    base_reward = hover_reward(next_state, action, target_position, config)

    # Check for success
    is_success = check_success(next_state, target_position, config)
    success_bonus = config.success_bonus if is_success else 0.0

    # Check for crash
    is_crash = check_crash(next_state)
    crash_penalty = config.crash_penalty if is_crash else 0.0

    # Total reward
    total_reward = base_reward + success_bonus + crash_penalty

    return {
        'base_reward': base_reward,
        'success_bonus': success_bonus,
        'crash_penalty': crash_penalty,
        'total_reward': total_reward,
        'is_success': is_success,
        'is_crash': is_crash,
    }


def waypoint_reward(
    state: DroneState,
    action: np.ndarray,
    waypoints: np.ndarray,
    current_waypoint_idx: int,
    config: Optional[RewardConfig] = None,
) -> Dict[str, float]:
    """
    Reward function for waypoint navigation task

    Args:
        state: Current drone state
        action: Action taken
        waypoints: Array of waypoints shape (N, 3)
        current_waypoint_idx: Index of current target waypoint
        config: Reward configuration

    Returns:
        Dictionary with reward components
    """
    if config is None:
        config = RewardConfig()

    if current_waypoint_idx >= len(waypoints):
        # All waypoints reached
        return {
            'reward': config.success_bonus,
            'waypoint_reached': True,
            'all_waypoints_reached': True,
        }

    target_waypoint = waypoints[current_waypoint_idx]

    # Distance to current waypoint
    distance_to_waypoint = np.linalg.norm(state.position - target_waypoint)

    # Reward for approaching waypoint
    approach_reward = np.exp(-3.0 * distance_to_waypoint)

    # Check if waypoint reached
    waypoint_reached = distance_to_waypoint < config.position_tolerance

    # Velocity alignment reward (moving towards waypoint)
    if np.linalg.norm(state.velocity) > 0.01:
        direction_to_waypoint = (target_waypoint - state.position) / (distance_to_waypoint + 1e-6)
        velocity_direction = state.velocity / (np.linalg.norm(state.velocity) + 1e-6)
        alignment = np.dot(direction_to_waypoint, velocity_direction)
        alignment_reward = max(0.0, alignment)  # Reward only forward progress
    else:
        alignment_reward = 0.0

    # Stability penalty (avoid aggressive maneuvers)
    roll_penalty = np.abs(state.attitude[0]) / 45.0
    pitch_penalty = np.abs(state.attitude[1]) / 45.0
    stability_penalty = (roll_penalty + pitch_penalty) * 0.5

    total_reward = (
        5.0 * approach_reward
        + 2.0 * alignment_reward
        - 1.0 * stability_penalty
    )

    if waypoint_reached:
        total_reward += config.success_bonus

    return {
        'reward': total_reward,
        'waypoint_reached': waypoint_reached,
        'all_waypoints_reached': False,
        'distance_to_waypoint': distance_to_waypoint,
    }


if __name__ == "__main__":
    # Test reward functions
    print("Testing reward functions...")

    # Create test states
    state = DroneState.zero_state()
    state.position = np.array([0.05, 0.03, 0.52])  # Close to target
    state.velocity = np.array([0.01, -0.02, 0.0])  # Small velocity
    state.attitude = np.array([2.0, -1.5, 0.0])  # Nearly level

    action = np.array([0.3, 0.0, 0.0, 0.0])  # Hover throttle, no control

    # Test hover reward
    reward = hover_reward(state, action)
    print(f"\nHover reward: {reward:.3f}")

    # Test success check
    is_success = check_success(state)
    print(f"Is hovering successfully: {is_success}")

    # Test crash check
    is_crash = check_crash(state)
    print(f"Has crashed: {is_crash}")

    # Test comprehensive reward
    next_state = DroneState.zero_state()
    next_state.position = np.array([0.02, 0.01, 0.50])
    next_state.velocity = np.array([0.0, 0.0, 0.0])
    next_state.attitude = np.array([0.5, 0.3, 0.0])

    reward_breakdown = calculate_reward(state, action, next_state)
    print(f"\nReward breakdown:")
    for key, value in reward_breakdown.items():
        print(f"  {key}: {value}")

    # Test crash scenario
    crash_state = DroneState.zero_state()
    crash_state.position = np.array([0.0, 0.0, 0.02])  # Too low
    crash_state.attitude = np.array([70.0, 0.0, 0.0])  # Extreme roll

    is_crash = check_crash(crash_state)
    print(f"\nCrash state detected: {is_crash}")

    # Test waypoint reward
    waypoints = np.array([
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.7],
        [1.0, 1.0, 0.5],
    ])

    waypoint_state = DroneState.zero_state()
    waypoint_state.position = np.array([0.3, 0.4, 0.48])
    waypoint_state.velocity = np.array([0.2, 0.1, 0.02])

    waypoint_result = waypoint_reward(waypoint_state, action, waypoints, 0)
    print(f"\nWaypoint navigation reward:")
    for key, value in waypoint_result.items():
        print(f"  {key}: {value}")
