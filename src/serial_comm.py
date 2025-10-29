"""
Serial Communication Module for BETAFPV Devices
Handles connection and data reading from serial ports
"""

import serial
import threading
import time
from typing import Callable, Optional


class SerialConnection:
    """Manages serial port connection for BETAFPV devices"""

    def __init__(self, port: str = "COM2", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected = False
        self.read_thread: Optional[threading.Thread] = None
        self.running = False
        self.data_callback: Optional[Callable] = None

    def connect(self) -> bool:
        """
        Establish connection to the serial port
        Returns True if successful, False otherwise
        """
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0
            )
            self.is_connected = True
            self.running = True

            # Start reading thread
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()

            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Close the serial port connection"""
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        self.is_connected = False

    def _read_loop(self):
        """Background thread for reading serial data"""
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    if self.data_callback and data:
                        self.data_callback(data)
            except serial.SerialException as e:
                print(f"Read error: {e}")
                self.running = False
                self.is_connected = False
                break
            except Exception as e:
                print(f"Unexpected error: {e}")

            time.sleep(0.01)  # 10ms polling

    def set_data_callback(self, callback: Callable):
        """Set callback function for received data"""
        self.data_callback = callback

    def send_data(self, data: bytes) -> bool:
        """
        Send data to the serial port
        Returns True if successful, False otherwise
        """
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.write(data)
                return True
            return False
        except serial.SerialException as e:
            print(f"Send error: {e}")
            return False
