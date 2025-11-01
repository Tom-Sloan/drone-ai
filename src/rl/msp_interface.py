"""
MSP Interface Abstraction Layer
Provides unified interface for both simulation and real hardware
"""

import sys
import os
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import time
import threading

# Add parent directory to path to import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .state import DroneState, action_to_channels


class DroneInterface(ABC):
    """Abstract base class for drone control interface"""

    @abstractmethod
    def reset(self) -> DroneState:
        """
        Reset drone to initial state

        Returns:
            Initial drone state
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> DroneState:
        """
        Execute action and return new state

        Args:
            action: [throttle, roll, pitch, yaw] in [-1, 1]

        Returns:
            New drone state after action
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to drone"""
        pass


class SimDroneInterface(DroneInterface):
    """
    Interface for PyBullet simulator

    This wraps the DroneSimulator class and provides state extraction
    """

    def __init__(self, simulator):
        """
        Initialize simulator interface

        Args:
            simulator: DroneSimulator instance
        """
        self.simulator = simulator
        self._connected = False

    def reset(self) -> DroneState:
        """Reset simulator to initial state"""
        self._connected = True
        return self.simulator.reset()

    def step(self, action: np.ndarray) -> DroneState:
        """Execute action in simulator"""
        if not self._connected:
            raise RuntimeError("Simulator not connected. Call reset() first.")

        return self.simulator.step(action)

    def close(self):
        """Clean up simulator"""
        if hasattr(self.simulator, 'close'):
            self.simulator.close()
        self._connected = False

    def is_connected(self) -> bool:
        """Check if simulator is active"""
        return self._connected


class RealDroneInterface(DroneInterface):
    """
    Interface for real Air65 drone via MSP

    This wraps the existing MSPControlLoop and serial communication
    """

    def __init__(self, serial_conn, msp_parser, control_loop):
        """
        Initialize real drone interface

        Args:
            serial_conn: SerialConnection instance
            msp_parser: MSPParser instance
            control_loop: MSPControlLoop instance
        """
        self.serial_conn = serial_conn
        self.msp_parser = msp_parser
        self.control_loop = control_loop

        # Telemetry data (updated from serial callbacks)
        self.telemetry_lock = threading.Lock()
        self.telemetry_data = {
            'voltage': 3.7,
            'current': 0.0,
            'rssi': 0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'armed': False,
        }

        # State estimation
        self.last_state = DroneState.zero_state()
        self.last_update_time = time.time()

        # Position/velocity estimation (requires external tracking system)
        # For now, we'll use a simple integration approach
        # TODO: Integrate with motion capture system (Vicon, OptiTrack, etc.)
        self.estimated_position = np.array([0.0, 0.0, 0.5])
        self.estimated_velocity = np.zeros(3)

        # Setup telemetry callback
        self.serial_conn.set_data_callback(self._on_telemetry)

    def _on_telemetry(self, data: bytes):
        """
        Callback for incoming serial data

        Parses MSP messages and updates telemetry
        """
        messages = self.msp_parser.parse(data)

        with self.telemetry_lock:
            for cmd, payload in messages:
                # MSP_ANALOG (110) - Battery and RSSI
                if cmd == 110:
                    parsed = self.msp_parser.parse_analog(payload)
                    if parsed:
                        self.telemetry_data['voltage'] = parsed.get('voltage', 3.7)
                        self.telemetry_data['current'] = parsed.get('current', 0.0)
                        self.telemetry_data['rssi'] = parsed.get('rssi', 0)

                # MSP_ATTITUDE (108) - Orientation
                elif cmd == 108:
                    parsed = self.msp_parser.parse_attitude(payload)
                    if parsed:
                        self.telemetry_data['roll'] = parsed.get('roll', 0.0)
                        self.telemetry_data['pitch'] = parsed.get('pitch', 0.0)
                        self.telemetry_data['yaw'] = parsed.get('yaw', 0.0)

                # MSP_STATUS (101) - Armed state
                elif cmd == 101:
                    parsed = self.msp_parser.parse_status(payload)
                    if parsed:
                        self.telemetry_data['armed'] = parsed.get('armed', False)

    def _get_current_state(self) -> DroneState:
        """
        Construct DroneState from current telemetry

        Note: Position and velocity estimation requires external tracking
        (motion capture, visual odometry, etc.)
        """
        with self.telemetry_lock:
            current_time = time.time()
            dt = current_time - self.last_update_time

            # Get attitude from telemetry
            attitude = np.array([
                self.telemetry_data['roll'],
                self.telemetry_data['pitch'],
                self.telemetry_data['yaw']
            ])

            # Estimate angular velocity (simple finite difference)
            angular_velocity = (attitude - self.last_state.attitude) / (dt + 1e-6)

            # TODO: Integrate with motion capture or visual odometry
            # For now, position and velocity are estimated (not accurate!)
            position = self.estimated_position.copy()
            velocity = self.estimated_velocity.copy()

            state = DroneState(
                position=position,
                velocity=velocity,
                attitude=attitude,
                angular_velocity=angular_velocity,
                battery_voltage=self.telemetry_data['voltage'],
                prev_action=self.control_loop.get_channels()[:4] if hasattr(self.control_loop, 'get_channels') else np.zeros(4),
                timestamp=current_time
            )

            self.last_state = state
            self.last_update_time = current_time

            return state

    def reset(self) -> DroneState:
        """
        Reset drone to initial hover state

        For real hardware, this means:
        1. Disarm if armed
        2. Wait for drone to be placed at initial position
        3. Arm and takeoff to hover

        SAFETY: This should be done with extreme caution!
        """
        # Disarm for safety
        if self.telemetry_data['armed']:
            self.control_loop.disarm()
            time.sleep(1.0)

        # Reset state estimation
        self.estimated_position = np.array([0.0, 0.0, 0.5])
        self.estimated_velocity = np.zeros(3)
        self.last_update_time = time.time()

        # Wait for user to position drone
        print("Place drone at starting position and press Enter...")
        input()

        # Arm and gentle takeoff
        self.control_loop.arm()
        time.sleep(0.5)

        # Gradual throttle increase to hover
        for throttle in np.linspace(1000, 1400, 20):
            self.control_loop.set_channels(throttle=int(throttle))
            time.sleep(0.05)

        # Get initial state
        time.sleep(0.5)  # Allow telemetry to update
        return self._get_current_state()

    def step(self, action: np.ndarray) -> DroneState:
        """
        Execute action on real drone

        Args:
            action: [throttle, roll, pitch, yaw] in [-1, 1]

        Returns:
            New drone state
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to drone")

        # Convert action to MSP channels
        roll_ch, pitch_ch, yaw_ch, throttle_ch = action_to_channels(action)

        # Send to drone
        self.control_loop.set_channels(
            roll=roll_ch,
            pitch=pitch_ch,
            yaw=yaw_ch,
            throttle=throttle_ch
        )

        # Wait for one control cycle (20ms @ 50Hz)
        time.sleep(0.02)

        # Get updated state
        return self._get_current_state()

    def close(self):
        """Clean up real drone connection"""
        # Emergency stop
        self.control_loop.emergency_stop()

        # Disarm
        time.sleep(0.5)
        self.control_loop.disarm()

        # Stop control loop
        if hasattr(self.control_loop, 'stop'):
            self.control_loop.stop()

    def is_connected(self) -> bool:
        """Check if connected to real drone"""
        return (
            self.serial_conn is not None
            and hasattr(self.serial_conn, 'is_connected')
            and self.serial_conn.is_connected
        )

    def set_position_estimate(self, position: np.ndarray, velocity: np.ndarray):
        """
        Update position estimate from external tracking system

        Args:
            position: [x, y, z] in meters
            velocity: [vx, vy, vz] in m/s
        """
        self.estimated_position = position.copy()
        self.estimated_velocity = velocity.copy()


if __name__ == "__main__":
    print("MSP Interface Abstraction Layer")
    print("=" * 50)

    print("\nThis module provides:")
    print("1. DroneInterface - Abstract base class")
    print("2. SimDroneInterface - Wraps PyBullet simulator")
    print("3. RealDroneInterface - Wraps MSPControlLoop for real hardware")

    print("\nUsage in Gymnasium environment:")
    print("  if use_sim:")
    print("      simulator = DroneSimulator()")
    print("      backend = SimDroneInterface(simulator)")
    print("  else:")
    print("      backend = RealDroneInterface(serial_conn, msp_parser, control_loop)")

    print("\n  state = backend.reset()")
    print("  next_state = backend.step(action)")

    print("\nThis abstraction enables seamless sim-to-real transfer!")
