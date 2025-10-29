"""
MAVLink 1.0 Protocol Parser
Implements MAVLink parsing for BETAFPV devices
"""

import struct
from enum import IntEnum


class MAVLinkMessages(IntEnum):
    """MAVLink message IDs used by BETAFPV (from BETAFPV_Configurator)"""
    BAD_DATA = -1
    HEARTBEAT = 1
    FIRMWARE_INFO = 2
    SYS_STATUS = 3
    IMU = 4
    ATTITUDE = 5
    LOCAL_POSITION = 6
    RC_CHANNELS = 7  # CORRECTED: was 6
    COMMAND = 8
    COMMAND_ACK = 9
    STATUS_TEXT = 10
    MOTORS_TEST = 11  # CORRECTED: was 8
    ALTITUDE = 12  # CORRECTED: was 9
    RATE = 13  # CORRECTED: was 10
    PID = 14  # CORRECTED: was 7
    MOTORS_MINIVALUE = 15
    UNIQUE_DEVICE_ID = 16


class MAVLinkParser:
    """Parse MAVLink 1.0 protocol messages"""

    MAVLINK_START_BYTE = 0xFE

    def __init__(self):
        self.buffer = bytearray()
        self.state = 'IDLE'
        self.msg_len = 0
        self.msg_seq = 0
        self.msg_sysid = 0
        self.msg_compid = 0
        self.msg_id = 0
        self.msg_payload = bytearray()
        self.msg_checksum = 0
        self.checksum_byte = 0

    def parse(self, data: bytes) -> list:
        """
        Parse incoming data and return list of complete messages
        Returns list of tuples: (message_id, payload)
        """
        messages = []
        self.buffer.extend(data)

        while len(self.buffer) > 0:
            msg = self._parse_byte(self.buffer.pop(0))
            if msg:
                messages.append(msg)

        return messages

    def _parse_byte(self, byte: int):
        """Parse a single byte in the MAVLink state machine"""

        if self.state == 'IDLE':
            if byte == self.MAVLINK_START_BYTE:
                self.state = 'GET_LENGTH'
                self.checksum_byte = 0

        elif self.state == 'GET_LENGTH':
            self.msg_len = byte
            self._update_checksum(byte)
            self.state = 'GET_SEQ'

        elif self.state == 'GET_SEQ':
            self.msg_seq = byte
            self._update_checksum(byte)
            self.state = 'GET_SYSID'

        elif self.state == 'GET_SYSID':
            self.msg_sysid = byte
            self._update_checksum(byte)
            self.state = 'GET_COMPID'

        elif self.state == 'GET_COMPID':
            self.msg_compid = byte
            self._update_checksum(byte)
            self.state = 'GET_MSGID'

        elif self.state == 'GET_MSGID':
            self.msg_id = byte
            self._update_checksum(byte)
            self.msg_payload = bytearray()

            if self.msg_len == 0:
                self.state = 'GET_CRC1'
            else:
                self.state = 'GET_PAYLOAD'

        elif self.state == 'GET_PAYLOAD':
            self.msg_payload.append(byte)
            self._update_checksum(byte)

            if len(self.msg_payload) >= self.msg_len:
                self.state = 'GET_CRC1'

        elif self.state == 'GET_CRC1':
            self.msg_checksum = byte
            self.state = 'GET_CRC2'

        elif self.state == 'GET_CRC2':
            self.msg_checksum |= (byte << 8)
            self.state = 'IDLE'

            # For simplicity, we're not validating CRC here
            # In production, you'd verify the checksum
            return (self.msg_id, bytes(self.msg_payload))

        return None

    def _update_checksum(self, byte: int):
        """Update CRC-16 checksum"""
        tmp = byte ^ (self.checksum_byte & 0xFF)
        tmp ^= (tmp << 4) & 0xFF
        self.checksum_byte = ((self.checksum_byte >> 8) ^
                               (tmp << 8) ^
                               (tmp << 3) ^
                               (tmp >> 4))
        self.checksum_byte &= 0xFFFF

    @staticmethod
    def parse_heartbeat(payload: bytes) -> dict:
        """Parse HEARTBEAT message"""
        if len(payload) >= 4:
            mode, armed_status, api_version, config_status = struct.unpack('<BBBB', payload[:4])
            return {
                'flight_mode': mode,
                'armed': bool(armed_status & 0x80),
                'api_version': api_version,
                'config_status': config_status
            }
        return {}

    @staticmethod
    def parse_attitude(payload: bytes) -> dict:
        """Parse ATTITUDE message"""
        if len(payload) >= 12:
            roll, pitch, yaw = struct.unpack('<fff', payload[:12])
            return {
                'roll': roll,  # radians
                'pitch': pitch,
                'yaw': yaw
            }
        return {}

    @staticmethod
    def parse_rc_channels(payload: bytes) -> dict:
        """Parse RC_CHANNELS message"""
        channels = []
        if len(payload) >= 24:
            for i in range(12):
                ch = struct.unpack('<H', payload[i*2:i*2+2])[0]
                channels.append(ch)

            rssi = payload[24] if len(payload) > 24 else 0

            return {
                'channels': channels,
                'rssi': rssi
            }
        return {}

    @staticmethod
    def parse_sys_status(payload: bytes) -> dict:
        """Parse SYS_STATUS message"""
        if len(payload) >= 8:
            sensor_health, voltage, current, battery, cpu_load = struct.unpack('<HHHBB', payload[:8])
            return {
                'sensor_health': sensor_health,
                'voltage': voltage / 100.0,  # Convert to volts
                'current': current / 100.0,  # Convert to amps
                'battery': battery,  # percentage
                'cpu_load': cpu_load
            }
        return {}

    @staticmethod
    def parse_imu(payload: bytes) -> dict:
        """Parse IMU message"""
        if len(payload) >= 36:
            data = struct.unpack('<fffffffff', payload[:36])
            return {
                'accel_x': data[0],
                'accel_y': data[1],
                'accel_z': data[2],
                'gyro_x': data[3],
                'gyro_y': data[4],
                'gyro_z': data[5],
                'mag_x': data[6],
                'mag_y': data[7],
                'mag_z': data[8]
            }
        return {}
