#!/usr/bin/env python3
"""
Raw Serial Data Diagnostic Tool
Connects to the BETAFPV device and displays all incoming data in hex format
"""

import serial
import sys
import time

def test_serial_connection(port='/dev/cu.usbmodem568241534B002', baudrate=115200):
    """Test serial connection and display raw data"""

    print("=" * 70)
    print("BETAFPV Serial Diagnostic Tool")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Baud Rate: {baudrate}")
    print()

    try:
        # Open serial port
        ser = serial.Serial(port, baudrate, timeout=1.0)
        print(f"✓ Successfully opened {port}")
        print(f"  Port is open: {ser.is_open}")
        print()

        # Wait a moment for connection to stabilize
        time.sleep(0.5)

        print("Listening for data... (Press Ctrl+C to stop)")
        print("-" * 70)

        byte_count = 0
        last_report = time.time()

        while True:
            # Check if data is available
            if ser.in_waiting > 0:
                # Read available data
                data = ser.read(ser.in_waiting)
                byte_count += len(data)

                # Display as hex
                hex_str = ' '.join(f'{b:02X}' for b in data)

                # Display as ASCII (if printable)
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)

                print(f"[{len(data):4d} bytes] {hex_str}")
                print(f"             ASCII: {ascii_str}")
                print()

                # Check for MSP header
                if b'$M' in data:
                    print("    ⚠ DETECTED MSP HEADER ($M)")

                # Check for MAVLink header
                if b'\xFE' in data or data.startswith(b'\xFE'):
                    print("    ⚠ DETECTED MAVLink HEADER (0xFE)")

                print()

            # Report byte rate every second
            current_time = time.time()
            if current_time - last_report >= 1.0:
                bytes_per_sec = byte_count / (current_time - last_report)
                print(f">>> Receiving ~{bytes_per_sec:.1f} bytes/sec (Total: {byte_count})")
                byte_count = 0
                last_report = current_time

            time.sleep(0.01)  # Small delay

    except serial.SerialException as e:
        print(f"✗ Serial Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check that the device is plugged in")
        print("  2. Verify the correct port name")
        print("  3. Make sure no other program is using the port")
        print("  4. On Mac, try 'ls /dev/cu.* /dev/tty.*' to see available ports")
        sys.exit(1)

    except KeyboardInterrupt:
        print()
        print("-" * 70)
        print("Stopped by user")
        print(f"Total bytes received: {byte_count}")
        if ser and ser.is_open:
            ser.close()
            print("✓ Port closed")

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test BETAFPV serial connection')
    parser.add_argument('--port', default='/dev/cu.usbmodem568241534B002',
                       help='Serial port (default: /dev/cu.usbmodem568241534B002)')
    parser.add_argument('--baud', type=int, default=115200,
                       help='Baud rate (default: 115200)')

    args = parser.parse_args()

    test_serial_connection(args.port, args.baud)
