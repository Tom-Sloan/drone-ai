"""
PyBullet Drone Simulator - Complete Implementation
Simulates a quadrotor with realistic physics
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Optional
import yaml
from pathlib import Path
import time

from .state import DroneState


class DroneSimulator:
    """
    PyBullet-based quadrotor simulator with realistic dynamics

    Features:
    - Quad-X configuration
    - First-order motor dynamics
    - Battery discharge model
    - Domain randomization for sim-to-real transfer
    - Ground effect physics
    """

    def __init__(self, config: Optional[Dict] = None, gui: bool = False):
        """
        Initialize simulator

        Args:
            config: Simulation configuration (from sim_config.yaml)
            gui: Enable PyBullet GUI for visualization
        """
        self.gui = gui
        self.config = config or self._load_default_config()

        # Extract config parameters
        self.timestep = self.config['pybullet']['timestep']
        self.gravity = self.config['simulation']['gravity']

        # Drone physical parameters
        drone_params = self.config['drone']
        self.mass = drone_params['mass']
        self.arm_length = drone_params['wheelbase'] / 2.0  # Distance from center to motor

        # Motor parameters
        motor_params = self.config['motors']
        self.max_thrust = motor_params['max_thrust']
        self.motor_time_constant = motor_params['time_constant']
        self.thrust_to_torque = motor_params['thrust_to_torque_ratio']

        # Battery parameters
        battery_params = self.config['battery']
        self.battery_voltage = battery_params['voltage_max']
        self.battery_max = battery_params['voltage_max']
        self.battery_min = battery_params['voltage_min']
        self.battery_capacity = battery_params['capacity_mah']

        # Domain randomization
        self.randomization = self.config['simulation']['randomization']

        # Initialize PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.timestep)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create simple quadrotor from primitives
        self.drone_id = self._create_quadrotor()

        # Motor state (for first-order dynamics)
        self.motor_thrusts = np.zeros(4)
        self.motor_commands = np.zeros(4)

        # Control mixing matrix (Quad-X configuration)
        # Motor layout:  1(FL)  0(FR)
        #                2(BL)  3(BR)
        self.mixing_matrix = np.array([
            [ 1.0,  1.0,  1.0, -1.0],  # Motor 0 (FR): +thrust, +roll, +pitch, -yaw
            [ 1.0, -1.0,  1.0,  1.0],  # Motor 1 (FL): +thrust, -roll, +pitch, +yaw
            [ 1.0, -1.0, -1.0, -1.0],  # Motor 2 (BL): +thrust, -roll, -pitch, -yaw
            [ 1.0,  1.0, -1.0,  1.0],  # Motor 3 (BR): +thrust, +roll, -pitch, +yaw
        ])

        # Motor positions in body frame (X forward, Y left, Z up)
        self.motor_positions = np.array([
            [ self.arm_length,  self.arm_length, 0],  # FR
            [ self.arm_length, -self.arm_length, 0],  # FL
            [-self.arm_length, -self.arm_length, 0],  # BL
            [-self.arm_length,  self.arm_length, 0],  # BR
        ])

        # Motor spin directions (for yaw torque)
        self.motor_directions = np.array([1, -1, 1, -1])  # CW, CCW, CW, CCW

        # Simulation state
        self.timesteps = 0
        self.prev_action = np.zeros(4)

        # Randomization state
        self.current_mass = self.mass
        self.mass_multiplier = 1.0
        self.drag_multiplier = 1.0

        print(f"DroneSimulator initialized (GUI={'ON' if gui else 'OFF'})")
        print(f"  Mass: {self.mass:.3f} kg")
        print(f"  Max thrust per motor: {self.max_thrust:.3f} N")
        print(f"  Hover thrust per motor: {self.mass * abs(self.gravity) / 4:.3f} N")

    def _load_default_config(self) -> Dict:
        """Load default sim config"""
        config_path = Path(__file__).parent.parent.parent / "configs/sim_config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_quadrotor(self):
        """Create a simple quadrotor using PyBullet primitives"""
        # Create collision shape (sphere for simplicity)
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)

        # Create visual shape (sphere)
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[0.2, 0.2, 0.8, 1.0]
        )

        # Create multibody
        drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, 1.0],
            baseOrientation=[0, 0, 0, 1]
        )

        # Set dynamics properties
        p.changeDynamics(
            drone_id, -1,
            linearDamping=0.04,
            angularDamping=0.04,
            restitution=0.0,
            lateralFriction=0.5
        )

        return drone_id

    def reset(self, randomize: bool = True) -> DroneState:
        """
        Reset simulator to initial state

        Args:
            randomize: Apply domain randomization

        Returns:
            Initial drone state
        """
        # Reset pose
        initial_pos = [0, 0, 0.5]  # Start 0.5m above ground
        initial_orn = [0, 0, 0, 1]  # Quaternion (no rotation)

        # Add randomization to initial pose if enabled
        if randomize and self.randomization['enabled']:
            pose_rand = self.randomization['initial_pose']
            initial_pos[0] += np.random.uniform(-pose_rand['position'], pose_rand['position'])
            initial_pos[1] += np.random.uniform(-pose_rand['position'], pose_rand['position'])
            initial_pos[2] += np.random.uniform(0, pose_rand['position'])

            # Random initial orientation (small angles)
            euler_noise = np.random.uniform(-pose_rand['orientation'], pose_rand['orientation'], 3)
            initial_orn = p.getQuaternionFromEuler(euler_noise)

        p.resetBasePositionAndOrientation(self.drone_id, initial_pos, initial_orn)
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])

        # Reset motor state
        self.motor_thrusts = np.zeros(4)
        self.motor_commands = np.zeros(4)

        # Reset battery
        self.battery_voltage = self.battery_max

        # Reset timesteps
        self.timesteps = 0
        self.prev_action = np.zeros(4)

        # Apply domain randomization
        if randomize and self.randomization['enabled']:
            self._apply_randomization()
        else:
            self.current_mass = self.mass
            self.mass_multiplier = 1.0
            self.drag_multiplier = 1.0

        # Return initial state
        return self._get_state()

    def step(self, action: np.ndarray) -> DroneState:
        """
        Step simulation forward one timestep

        Args:
            action: [throttle, roll, pitch, yaw] in [-1, 1]
                    throttle: -1=min, 1=max thrust
                    roll: -1=left, 1=right
                    pitch: -1=backward, 1=forward
                    yaw: -1=CCW, 1=CW

        Returns:
            New drone state after physics step
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Convert action to motor commands
        motor_commands = self._action_to_motor_commands(action)

        # Apply first-order motor dynamics
        alpha = self.timestep / self.motor_time_constant
        alpha = min(alpha, 1.0)  # Prevent overshoot
        self.motor_thrusts += alpha * (motor_commands - self.motor_thrusts)

        # Add motor noise if randomization enabled
        if self.randomization['enabled']:
            noise_std = self.randomization['motor_noise']['thrust_std']
            noise = np.random.normal(0, noise_std * self.max_thrust, 4)
            motor_thrusts_noisy = np.clip(self.motor_thrusts + noise, 0, self.max_thrust)
        else:
            motor_thrusts_noisy = self.motor_thrusts

        # Apply forces and torques to drone
        self._apply_motor_forces(motor_thrusts_noisy)

        # Apply aerodynamic drag
        self._apply_drag()

        # Step physics simulation
        p.stepSimulation()

        # Update battery based on current draw
        self._update_battery(motor_thrusts_noisy)

        # Update counters
        self.timesteps += 1
        self.prev_action = action.copy()

        # Add small delay for GUI
        if self.gui:
            time.sleep(self.timestep)

        # Extract and return new state
        return self._get_state()

    def _action_to_motor_commands(self, action: np.ndarray) -> np.ndarray:
        """
        Convert action to individual motor thrust commands using mixing matrix

        Args:
            action: [throttle, roll, pitch, yaw] normalized to [-1, 1]

        Returns:
            Motor thrusts [m0, m1, m2, m3] in Newtons
        """
        # Normalize throttle from [-1, 1] to [0, 1]
        throttle_normalized = (action[0] + 1.0) / 2.0

        # Calculate hover thrust (thrust needed to counteract gravity)
        hover_thrust = self.current_mass * abs(self.gravity) / 4.0

        # Base thrust per motor
        thrust_base = throttle_normalized * self.max_thrust

        # Control authority (how much roll/pitch/yaw can affect thrust)
        roll_authority = 0.3 * self.max_thrust
        pitch_authority = 0.3 * self.max_thrust
        yaw_authority = 0.2 * self.max_thrust

        # Create control input vector
        control_input = np.array([
            thrust_base,
            action[1] * roll_authority,
            action[2] * pitch_authority,
            action[3] * yaw_authority
        ])

        # Apply mixing matrix: motor_thrusts = mixing_matrix @ control_input
        motor_thrusts = self.mixing_matrix @ control_input

        # Clip to valid range [0, max_thrust]
        motor_thrusts = np.clip(motor_thrusts, 0.0, self.max_thrust)

        return motor_thrusts

    def _apply_motor_forces(self, motor_thrusts: np.ndarray):
        """
        Apply thrust forces and yaw torques from motors to drone

        Args:
            motor_thrusts: Thrust for each motor [N]
        """
        for i, thrust in enumerate(motor_thrusts):
            # Motor position in body frame
            motor_pos = self.motor_positions[i]

            # Thrust force (upward in body frame)
            force = [0, 0, thrust]

            # Apply force at motor position
            p.applyExternalForce(
                self.drone_id,
                -1,  # Apply to base link
                force,
                motor_pos,
                p.LINK_FRAME
            )

            # Apply yaw torque (reaction torque from motor spin)
            # Torque = motor_direction * thrust_to_torque * thrust
            yaw_torque = self.motor_directions[i] * self.thrust_to_torque * thrust
            torque = [0, 0, yaw_torque]

            p.applyExternalTorque(
                self.drone_id,
                -1,
                torque,
                p.LINK_FRAME
            )

    def _apply_drag(self):
        """Apply aerodynamic drag forces"""
        # Get current velocity
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)

        # Linear drag: F_drag = -k * v^2 * sign(v)
        drag_coeff = self.config['aerodynamics']['drag_coefficient'] * self.drag_multiplier
        drag_force = -drag_coeff * lin_vel * np.abs(lin_vel)

        # Angular drag
        angular_drag = -0.01 * ang_vel * np.abs(ang_vel)

        # Apply drag
        p.applyExternalForce(
            self.drone_id, -1,
            drag_force.tolist(),
            [0, 0, 0],
            p.WORLD_FRAME
        )

        p.applyExternalTorque(
            self.drone_id, -1,
            angular_drag.tolist(),
            p.WORLD_FRAME
        )

    def _update_battery(self, motor_thrusts: np.ndarray):
        """
        Update battery voltage based on current draw

        Args:
            motor_thrusts: Current motor thrusts [N]
        """
        # Simple battery model: discharge rate proportional to thrust
        # Current draw (A) = k * total_thrust
        total_thrust = np.sum(motor_thrusts)
        current_draw = total_thrust * 2.0  # Rough approximation: 2A per Newton

        # Energy consumed this timestep (mAh)
        energy_consumed = current_draw * (self.timestep / 3600.0) * 1000.0

        # Update voltage (linear discharge model)
        discharge_fraction = energy_consumed / self.battery_capacity
        voltage_drop = discharge_fraction * (self.battery_max - self.battery_min)

        self.battery_voltage = max(
            self.battery_min,
            self.battery_voltage - voltage_drop
        )

    def _get_state(self) -> DroneState:
        """
        Extract current state from PyBullet

        Returns:
            Current drone state
        """
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        pos = np.array(pos)

        # Convert quaternion to Euler angles (in degrees)
        euler_rad = p.getEulerFromQuaternion(orn)
        euler_deg = np.degrees(euler_rad)
        roll, pitch, yaw = euler_deg

        # Get velocities
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)

        # Convert angular velocity to degrees/sec
        ang_vel_deg = np.degrees(ang_vel)

        # Create DroneState
        state = DroneState(
            position=pos,
            velocity=lin_vel,
            attitude=np.array([roll, pitch, yaw]),
            angular_velocity=ang_vel_deg,
            battery_voltage=self.battery_voltage,
            prev_action=self.prev_action.copy(),
            timestamp=self.timesteps * self.timestep
        )

        return state

    def _apply_randomization(self):
        """
        Apply domain randomization for sim-to-real transfer

        Randomizes:
        - Mass (±20%)
        - Drag coefficients
        - Motor characteristics
        """
        rand_config = self.randomization

        # Randomize mass
        mass_range = rand_config['mass_range']
        self.mass_multiplier = np.random.uniform(mass_range[0] / self.mass, mass_range[1] / self.mass)
        self.current_mass = self.mass * self.mass_multiplier

        # Update PyBullet mass
        p.changeDynamics(self.drone_id, -1, mass=self.current_mass)

        # Randomize drag
        drag_range = rand_config['drag_range']
        self.drag_multiplier = np.random.uniform(
            drag_range[0] / self.config['aerodynamics']['drag_coefficient'],
            drag_range[1] / self.config['aerodynamics']['drag_coefficient']
        )

    def get_camera_image(self, width=320, height=240):
        """
        Get camera image from drone's perspective

        Args:
            width: Image width
            height: Image height

        Returns:
            RGB image as numpy array
        """
        # Get drone pose
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)

        # Camera position (slightly in front and above drone)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        camera_offset = np.array([0.1, 0, 0.05])  # Forward and up
        camera_pos = pos + rot_matrix @ camera_offset

        # Target position (forward from drone)
        target_offset = np.array([1.0, 0, 0])
        target_pos = pos + rot_matrix @ target_offset

        # Up vector
        up_vector = rot_matrix @ np.array([0, 0, 1])

        # Setup camera
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=width/height, nearVal=0.01, farVal=100
        )

        # Get image
        img = p.getCameraImage(
            width, height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )

        rgb = np.array(img[2]).reshape(height, width, 4)[:, :, :3]
        return rgb

    def close(self):
        """Clean up PyBullet connection"""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            print("DroneSimulator closed")


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DroneSimulator Test")
    print("=" * 70)

    # Create simulator with GUI
    sim = DroneSimulator(gui=True)

    print("\nStarting simulation with random actions...")
    print("The drone should hover roughly in place with noisy control")

    # Reset
    state = sim.reset()
    print(f"\nInitial state:")
    print(f"  Position: {state.position}")
    print(f"  Attitude: {state.attitude}")

    # Run simulation with hover-ish action
    for i in range(1000):
        # Hover action with small random noise
        hover_throttle = 0.3  # Roughly hover thrust
        action = np.array([
            hover_throttle + np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),  # Small roll
            np.random.uniform(-0.1, 0.1),  # Small pitch
            np.random.uniform(-0.05, 0.05)  # Small yaw
        ])

        state = sim.step(action)

        # Print status every 100 steps
        if i % 100 == 0:
            print(f"\nStep {i}:")
            print(f"  Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
            print(f"  Velocity: [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}]")
            print(f"  Attitude: [{state.attitude[0]:.1f}°, {state.attitude[1]:.1f}°, {state.attitude[2]:.1f}°]")
            print(f"  Battery: {state.battery_voltage:.2f}V")

    print("\nTest complete! Closing simulator...")
    sim.close()
