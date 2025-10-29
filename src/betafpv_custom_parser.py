"""
BETAFPV Custom Protocol Parser
Handles the proprietary 16-byte RC channel format from BETAFPV 2SE transmitter
"""

import struct


class BETAFPVCustomParser:
    """Parse BETAFPV custom 16-byte RC channel protocol"""

    def __init__(self):
        self.buffer = bytearray()

    def parse(self, data: bytes) -> list:
        """
        Parse incoming data for 16-byte RC channel packets
        Returns list of dictionaries with RC channel data
        """
        messages = []
        self.buffer.extend(data)

        # Look for 16-byte packets
        while len(self.buffer) >= 16:
            # Check if this looks like a valid packet
            # Pattern: repeating structure with values in typical RC range
            packet = self.buffer[:16]

            # Try to parse as 8 channels of 16-bit little-endian values
            try:
                channels = []
                for i in range(8):
                    offset = i * 2
                    value = struct.unpack('<H', packet[offset:offset+2])[0]
                    channels.append(value)

                # Validate: RC channels should be in reasonable range (0-2047 typical)
                # If all channels are in valid range, accept the packet
                if all(0 <= ch <= 4095 for ch in channels):
                    messages.append({
                        'type': 'RC_CHANNELS',
                        'channels': channels,
                        'raw_packet': bytes(packet)
                    })

                # Remove processed packet
                self.buffer = self.buffer[16:]

            except Exception:
                # If parsing fails, remove one byte and try again
                self.buffer.pop(0)

        return messages

    @staticmethod
    def parse_rc_channels(packet_data: dict) -> dict:
        """
        Convert raw RC channel values to standard format
        BETAFPV uses 0-2047 range, convert to standard 1000-2000
        """
        channels = packet_data.get('channels', [])

        # Convert from 0-2047 to 1000-2000 range
        # Assuming 0x03FF (1023) is center, map to 1500
        converted_channels = []
        for ch in channels:
            # Map 0-2047 to 1000-2000
            # 1023 (0x3FF) -> 1500 (center)
            # Formula: output = 1000 + (input / 2047) * 1000
            if ch <= 2047:
                converted = int(1000 + (ch / 2047.0) * 1000)
            else:
                converted = ch  # Keep as-is if already in normal range
            converted_channels.append(converted)

        # Calculate RSSI from data variability (approximation)
        # More variation = better signal
        if len(channels) > 0:
            rssi = min(100, max(0, int((sum(channels) / (len(channels) * 2047.0)) * 100)))
        else:
            rssi = 0

        return {
            'channels': converted_channels,
            'rssi': rssi,
            'raw_values': channels
        }
