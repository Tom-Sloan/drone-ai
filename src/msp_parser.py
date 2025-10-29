"""
MSP (MultiWii Serial Protocol) Parser
Implements MSP V1 and V2 protocol parsing
"""

import struct
from enum import IntEnum


class MSPCommands(IntEnum):
    """Common MSP command codes"""
    MSP_API_VERSION = 1
    MSP_FC_VARIANT = 2
    MSP_FC_VERSION = 3
    MSP_BOARD_INFO = 4
    MSP_BUILD_INFO = 5
    MSP_NAME = 10
    MSP_STATUS = 101
    MSP_RAW_IMU = 102
    MSP_MOTOR = 104
    MSP_RC = 105
    MSP_ATTITUDE = 108
    MSP_ALTITUDE = 109
    MSP_ANALOG = 110
    MSP_BATTERY_STATE = 130


class MSPParser:
    """Parse MSP protocol messages"""

    def __init__(self):
        self.buffer = bytearray()
        self.state = 'IDLE'
        self.message_direction = None
        self.message_length = 0
        self.message_cmd = 0
        self.message_payload = bytearray()
        self.message_checksum = 0

    def parse(self, data: bytes) -> list:
        """
        Parse incoming data and return list of complete messages
        Returns list of tuples: (command, payload)
        """
        messages = []
        self.buffer.extend(data)

        while len(self.buffer) > 0:
            msg = self._parse_byte(self.buffer.pop(0))
            if msg:
                messages.append(msg)

        return messages

    def _parse_byte(self, byte: int):
        """Parse a single byte in the MSP state machine"""

        if self.state == 'IDLE':
            if byte == ord('$'):
                self.state = 'HEADER_START'

        elif self.state == 'HEADER_START':
            if byte == ord('M'):
                self.state = 'HEADER_M'
            else:
                self.state = 'IDLE'

        elif self.state == 'HEADER_M':
            if byte == ord('>'):  # From FC
                self.message_direction = 'FROM_FC'
                self.state = 'HEADER_SIZE'
            elif byte == ord('<'):  # To FC
                self.message_direction = 'TO_FC'
                self.state = 'HEADER_SIZE'
            elif byte == ord('!'):  # Error
                self.message_direction = 'ERROR'
                self.state = 'HEADER_SIZE'
            else:
                self.state = 'IDLE'

        elif self.state == 'HEADER_SIZE':
            self.message_length = byte
            self.state = 'HEADER_CMD'

        elif self.state == 'HEADER_CMD':
            self.message_cmd = byte
            self.message_payload = bytearray()
            if self.message_length == 0:
                self.state = 'CHECKSUM'
            else:
                self.state = 'PAYLOAD'

        elif self.state == 'PAYLOAD':
            self.message_payload.append(byte)
            if len(self.message_payload) >= self.message_length:
                self.state = 'CHECKSUM'

        elif self.state == 'CHECKSUM':
            self.message_checksum = byte

            # Verify checksum
            checksum = self.message_length ^ self.message_cmd
            for b in self.message_payload:
                checksum ^= b

            self.state = 'IDLE'

            if checksum == self.message_checksum:
                return (self.message_cmd, bytes(self.message_payload))

        return None

    @staticmethod
    def create_request(command: int, payload: bytes = b'') -> bytes:
        """Create an MSP request message"""
        length = len(payload)
        checksum = length ^ command
        for b in payload:
            checksum ^= b

        message = bytearray(b'$M<')
        message.append(length)
        message.append(command)
        message.extend(payload)
        message.append(checksum)

        return bytes(message)

    @staticmethod
    def parse_analog(payload: bytes) -> dict:
        """Parse MSP_ANALOG message"""
        if len(payload) >= 7:
            vbat, mah_drawn, rssi, amperage = struct.unpack('<BHHH', payload[:7])
            return {
                'voltage': vbat / 10.0,  # Convert to volts
                'mah_drawn': mah_drawn,
                'rssi': rssi,
                'amperage': amperage / 100.0  # Convert to amps
            }
        return {}

    @staticmethod
    def parse_attitude(payload: bytes) -> dict:
        """Parse MSP_ATTITUDE message"""
        if len(payload) >= 6:
            roll, pitch, yaw = struct.unpack('<hhh', payload[:6])
            return {
                'roll': roll / 10.0,  # Convert to degrees
                'pitch': pitch / 10.0,
                'yaw': yaw
            }
        return {}

    @staticmethod
    def parse_status(payload: bytes) -> dict:
        """Parse MSP_STATUS message"""
        if len(payload) >= 11:
            cycle_time, i2c_error, sensor, flag = struct.unpack('<HHHI', payload[:11])
            return {
                'cycle_time': cycle_time,
                'i2c_error': i2c_error,
                'sensors': sensor,
                'flags': flag,
                'armed': bool(flag & 0x01)
            }
        return {}
