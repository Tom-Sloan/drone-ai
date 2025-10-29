"""
MSP Control Loop for continuous drone control
Sends MSP_SET_RAW_RC commands at 50 Hz to maintain control
"""

import threading
import time
from typing import Optional
from msp_parser import MSPParser


class MSPControlLoop:
    """Manages continuous MSP control command transmission"""

    def __init__(self, serial_connection):
        """
        Initialize MSP control loop

        Args:
            serial_connection: SerialConnection instance for sending commands
        """
        self.serial_conn = serial_connection

        # Control state
        self.running = False
        self.control_thread: Optional[threading.Thread] = None

        # Channel values (1000-2000, center at 1500)
        # [roll, pitch, yaw, throttle, aux1, aux2, aux3, aux4]
        self.channels = [1500, 1500, 1500, 1000, 1000, 1000, 1000, 1000]
        self.channels_lock = threading.Lock()

        # Control loop settings
        self.frequency = 50  # Hz (20ms period)
        self.period = 1.0 / self.frequency

        # Statistics
        self.commands_sent = 0
        self.last_send_time = 0.0
        self.actual_frequency = 0.0

    def start(self):
        """Start the control loop"""
        if self.running:
            return

        self.running = True
        self.commands_sent = 0
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("MSP Control Loop started at {} Hz".format(self.frequency))

    def stop(self):
        """Stop the control loop"""
        if not self.running:
            return

        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        print("MSP Control Loop stopped. Commands sent: {}".format(self.commands_sent))

    def set_channels(self, roll=None, pitch=None, yaw=None, throttle=None,
                    aux1=None, aux2=None, aux3=None, aux4=None):
        """
        Update RC channel values

        Args:
            roll: Roll channel (1000-2000), None to keep current
            pitch: Pitch channel (1000-2000), None to keep current
            yaw: Yaw channel (1000-2000), None to keep current
            throttle: Throttle channel (1000-2000), None to keep current
            aux1-aux4: Auxiliary channels (1000-2000), None to keep current
        """
        with self.channels_lock:
            if roll is not None:
                self.channels[0] = self._clamp(roll)
            if pitch is not None:
                self.channels[1] = self._clamp(pitch)
            if yaw is not None:
                self.channels[2] = self._clamp(yaw)
            if throttle is not None:
                self.channels[3] = self._clamp(throttle)
            if aux1 is not None:
                self.channels[4] = self._clamp(aux1)
            if aux2 is not None:
                self.channels[5] = self._clamp(aux2)
            if aux3 is not None:
                self.channels[6] = self._clamp(aux3)
            if aux4 is not None:
                self.channels[7] = self._clamp(aux4)

    def get_channels(self) -> list:
        """Get current channel values"""
        with self.channels_lock:
            return self.channels.copy()

    def arm(self):
        """
        Arm the drone
        Arming typically requires:
        - Throttle low (1000)
        - Yaw right (>1500)
        - AUX1 high (>1500) for arm switch
        """
        with self.channels_lock:
            self.channels[3] = 1000  # Throttle low
            self.channels[2] = 1500  # Yaw center
            self.channels[4] = 2000  # AUX1 high (arm channel)
        print("Arming command sent (AUX1 = 2000)")

    def disarm(self):
        """
        Disarm the drone
        Set throttle low and arm channel low
        """
        with self.channels_lock:
            self.channels[3] = 1000  # Throttle low
            self.channels[4] = 1000  # AUX1 low (disarm)
        print("Disarming command sent (AUX1 = 1000)")

    def emergency_stop(self):
        """
        Emergency stop - disarm and reset all channels to safe values
        """
        with self.channels_lock:
            self.channels = [1500, 1500, 1500, 1000, 1000, 1000, 1000, 1000]
        print("EMERGENCY STOP - All channels reset")

    def _control_loop(self):
        """Main control loop running at target frequency"""
        last_time = time.time()
        frame_count = 0
        fps_start_time = time.time()

        while self.running:
            loop_start = time.time()

            # Get current channel values
            with self.channels_lock:
                channels = self.channels.copy()

            # Create and send MSP_SET_RAW_RC command
            command = MSPParser.create_set_raw_rc(channels)

            if self.serial_conn and self.serial_conn.is_connected:
                success = self.serial_conn.send_data(command)
                if success:
                    self.commands_sent += 1
                    self.last_send_time = time.time()

            # Calculate actual frequency every 50 frames
            frame_count += 1
            if frame_count >= 50:
                elapsed = time.time() - fps_start_time
                self.actual_frequency = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Sleep to maintain target frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.period - elapsed)
            time.sleep(sleep_time)

    @staticmethod
    def _clamp(value: float, min_val: int = 1000, max_val: int = 2000) -> int:
        """Clamp value to valid RC range"""
        return int(max(min_val, min(max_val, value)))

    def get_stats(self) -> dict:
        """Get control loop statistics"""
        return {
            'running': self.running,
            'commands_sent': self.commands_sent,
            'frequency': self.actual_frequency,
            'channels': self.get_channels()
        }
