"""
State Representation for Drone RL
Defines state structure, normalization, and denormalization
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DroneState:
    """
    Complete state representation for Air65 drone

    Attributes:
        position: [x, y, z] in meters
        velocity: [vx, vy, vz] in m/s
        attitude: [roll, pitch, yaw] in degrees
        angular_velocity: [roll_rate, pitch_rate, yaw_rate] in deg/s
        battery_voltage: Battery voltage in volts
        prev_action: Previous action [throttle, roll, pitch, yaw] in [-1, 1]
        timestamp: Time in seconds
    """
    position: np.ndarray  # [x, y, z] meters
    velocity: np.ndarray  # [vx, vy, vz] m/s
    attitude: np.ndarray  # [roll, pitch, yaw] degrees
    angular_velocity: np.ndarray  # [roll_rate, pitch_rate, yaw_rate] deg/s
    battery_voltage: float  # volts
    prev_action: np.ndarray  # [throttle, roll, pitch, yaw] in [-1, 1]
    timestamp: float  # seconds

    def to_array(self) -> np.ndarray:
        """
        Convert state to flat numpy array for neural network input

        Returns:
            State vector of shape (14,):
            [roll, pitch, yaw_rate, vx, vy, vz, x, y, z, battery,
             prev_throttle, prev_roll, prev_pitch, prev_yaw]
        """
        return np.array([
            self.attitude[0],  # roll
            self.attitude[1],  # pitch
            self.angular_velocity[2],  # yaw_rate
            self.velocity[0],  # vx
            self.velocity[1],  # vy
            self.velocity[2],  # vz
            self.position[0],  # x
            self.position[1],  # y
            self.position[2],  # z
            self.battery_voltage,  # battery
            self.prev_action[0],  # prev_throttle
            self.prev_action[1],  # prev_roll
            self.prev_action[2],  # prev_pitch
            self.prev_action[3],  # prev_yaw
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, state_array: np.ndarray, timestamp: float = 0.0) -> 'DroneState':
        """
        Create DroneState from flat array

        Args:
            state_array: State vector of shape (14,)
            timestamp: Time in seconds

        Returns:
            DroneState instance
        """
        return cls(
            attitude=np.array([state_array[0], state_array[1], 0.0]),  # roll, pitch, yaw (yaw not in state)
            angular_velocity=np.array([0.0, 0.0, state_array[2]]),  # yaw_rate only
            velocity=state_array[3:6].copy(),  # vx, vy, vz
            position=state_array[6:9].copy(),  # x, y, z
            battery_voltage=float(state_array[9]),
            prev_action=state_array[10:14].copy(),  # prev actions
            timestamp=timestamp
        )

    @classmethod
    def zero_state(cls) -> 'DroneState':
        """Create a zero-initialized state"""
        return cls(
            position=np.zeros(3),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            battery_voltage=3.7,  # 1S nominal voltage
            prev_action=np.zeros(4),
            timestamp=0.0
        )


class StateNormalizer:
    """
    Normalizes and denormalizes drone states for neural network training

    Normalization helps with training stability by keeping all state values
    in a similar range (typically [-1, 1] or [0, 1]).
    """

    def __init__(self):
        """Initialize normalization parameters"""
        # Normalization bounds [min, max] for each state dimension
        self.bounds = {
            'roll': (-45.0, 45.0),  # degrees
            'pitch': (-45.0, 45.0),  # degrees
            'yaw_rate': (-180.0, 180.0),  # deg/s
            'vx': (-2.0, 2.0),  # m/s
            'vy': (-2.0, 2.0),  # m/s
            'vz': (-2.0, 2.0),  # m/s
            'x': (-2.0, 2.0),  # meters
            'y': (-2.0, 2.0),  # meters
            'z': (0.0, 2.0),  # meters (always positive)
            'battery': (3.0, 4.2),  # volts (1S LiPo)
            'action': (-1.0, 1.0),  # prev actions already normalized
        }

        # Pre-compute normalization factors for efficiency
        self.state_min = np.array([
            self.bounds['roll'][0],
            self.bounds['pitch'][0],
            self.bounds['yaw_rate'][0],
            self.bounds['vx'][0],
            self.bounds['vy'][0],
            self.bounds['vz'][0],
            self.bounds['x'][0],
            self.bounds['y'][0],
            self.bounds['z'][0],
            self.bounds['battery'][0],
            self.bounds['action'][0],
            self.bounds['action'][0],
            self.bounds['action'][0],
            self.bounds['action'][0],
        ], dtype=np.float32)

        self.state_max = np.array([
            self.bounds['roll'][1],
            self.bounds['pitch'][1],
            self.bounds['yaw_rate'][1],
            self.bounds['vx'][1],
            self.bounds['vy'][1],
            self.bounds['vz'][1],
            self.bounds['x'][1],
            self.bounds['y'][1],
            self.bounds['z'][1],
            self.bounds['battery'][1],
            self.bounds['action'][1],
            self.bounds['action'][1],
            self.bounds['action'][1],
            self.bounds['action'][1],
        ], dtype=np.float32)

        self.state_range = self.state_max - self.state_min

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state to [-1, 1] range

        Args:
            state: Raw state vector of shape (14,)

        Returns:
            Normalized state vector of shape (14,) in [-1, 1]
        """
        # Clamp to bounds
        state_clamped = np.clip(state, self.state_min, self.state_max)

        # Normalize to [0, 1]
        normalized = (state_clamped - self.state_min) / self.state_range

        # Scale to [-1, 1]
        return 2.0 * normalized - 1.0

    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalize state from [-1, 1] range back to original scale

        Args:
            normalized_state: Normalized state vector of shape (14,) in [-1, 1]

        Returns:
            Raw state vector of shape (14,)
        """
        # Scale from [-1, 1] to [0, 1]
        scaled = (normalized_state + 1.0) / 2.0

        # Denormalize to original range
        return scaled * self.state_range + self.state_min

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action (already in [-1, 1], so just clip)

        Args:
            action: Action vector [throttle, roll, pitch, yaw] in [-1, 1]

        Returns:
            Clipped action vector
        """
        return np.clip(action, -1.0, 1.0)

    def denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Denormalize action (identity function since actions are in [-1, 1])

        Args:
            normalized_action: Action in [-1, 1]

        Returns:
            Same action (actions are already normalized)
        """
        return np.clip(normalized_action, -1.0, 1.0)


def action_to_channels(action: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Convert normalized action [-1, 1] to MSP channel values [1000, 2000]

    Args:
        action: [throttle, roll, pitch, yaw] in [-1, 1]

    Returns:
        Tuple of (roll_channel, pitch_channel, yaw_channel, throttle_channel) in [1000, 2000]
    """
    # Map [-1, 1] to [1000, 2000]
    channels = ((action + 1.0) * 500.0 + 1000.0).astype(int)

    # Clip to valid range
    channels = np.clip(channels, 1000, 2000)

    # Return in MSP order: roll, pitch, yaw, throttle
    return (
        int(channels[1]),  # roll
        int(channels[2]),  # pitch
        int(channels[3]),  # yaw
        int(channels[0]),  # throttle
    )


def channels_to_action(roll: int, pitch: int, yaw: int, throttle: int) -> np.ndarray:
    """
    Convert MSP channel values [1000, 2000] to normalized action [-1, 1]

    Args:
        roll, pitch, yaw, throttle: Channel values in [1000, 2000]

    Returns:
        Action array [throttle, roll, pitch, yaw] in [-1, 1]
    """
    channels = np.array([throttle, roll, pitch, yaw], dtype=np.float32)

    # Map [1000, 2000] to [-1, 1]
    action = (channels - 1000.0) / 500.0 - 1.0

    return np.clip(action, -1.0, 1.0)


if __name__ == "__main__":
    # Test state representation
    print("Testing DroneState...")
    state = DroneState.zero_state()
    state.position = np.array([0.1, 0.2, 0.5])
    state.velocity = np.array([0.0, 0.0, 0.1])
    state.attitude = np.array([5.0, -3.0, 45.0])

    # Convert to array and back
    state_array = state.to_array()
    print(f"State array shape: {state_array.shape}")
    print(f"State array: {state_array}")

    reconstructed = DroneState.from_array(state_array)
    print(f"Reconstructed position: {reconstructed.position}")

    # Test normalization
    print("\nTesting StateNormalizer...")
    normalizer = StateNormalizer()
    normalized = normalizer.normalize(state_array)
    print(f"Normalized state: {normalized}")
    print(f"Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")

    denormalized = normalizer.denormalize(normalized)
    print(f"Denormalized matches original: {np.allclose(state_array, denormalized)}")

    # Test action conversion
    print("\nTesting action conversion...")
    action = np.array([0.5, -0.2, 0.3, 0.1])  # [throttle, roll, pitch, yaw]
    channels = action_to_channels(action)
    print(f"Action {action} -> Channels {channels}")

    recovered_action = channels_to_action(*channels)
    print(f"Channels {channels} -> Action {recovered_action}")
    print(f"Recovery error: {np.abs(action - recovered_action).max():.6f}")
