"""
BETAFPV Configurator GUI - UltraThink Edition
A Python GUI for reading data from BETAFPV devices via serial port
Supports both MSP and MAVLink protocols
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from datetime import datetime
from serial_comm import SerialConnection
from msp_parser import MSPParser, MSPCommands
from mavlink_parser import MAVLinkParser, MAVLinkMessages
from betafpv_custom_parser import BETAFPVCustomParser
from joystick_widget import JoystickWidget
import serial.tools.list_ports
import json
import os


class BETAFPVConfiguratorGUI:
    """Main GUI application for BETAFPV device configuration"""

    def __init__(self, root):
        self.root = root
        self.root.title("BETAFPV Configurator - UltraThink")
        self.root.geometry("1100x800")

        # Serial connection
        self.serial_conn = None
        self.msp_parser = MSPParser()
        self.mavlink_parser = MAVLinkParser()
        self.betafpv_custom_parser = BETAFPVCustomParser()

        # Config file for remembering settings
        self.config_file = os.path.join(os.path.expanduser('~'), '.betafpv_config.json')

        # Telemetry data
        self.telemetry_data = {
            'voltage': 0.0,
            'current': 0.0,
            'rssi': 0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'armed': False,
            'mode': 'Unknown',
            'protocol': 'Auto-detecting...',
            'last_data_time': 0,
            'link_quality': 0
        }

        # Update lock
        self.update_lock = threading.Lock()

        # Connection tracking
        self.last_telemetry_time = 0
        self.drone_connected = False

        # Available ports
        self.available_ports = []

        # Debug settings
        self.show_raw_data = False
        self.bytes_received = 0

        # Build UI
        self._build_ui()

        # Load saved settings
        self._load_config()

        # Scan for ports on startup
        self._scan_ports()

        # Start UI update loop
        self._update_ui_loop()

    def _build_ui(self):
        """Build the user interface"""

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')

        # Top frame - Connection controls
        connection_frame = ttk.LabelFrame(self.root, text="Connection Settings", padding=10)
        connection_frame.pack(fill='x', padx=10, pady=5)

        # Port settings with dropdown
        ttk.Label(connection_frame, text="Port:").grid(row=0, column=0, padx=5, sticky='e')

        port_frame = ttk.Frame(connection_frame)
        port_frame.grid(row=0, column=1, padx=5)

        self.port_var = tk.StringVar(value="")
        self.port_dropdown = ttk.Combobox(
            port_frame,
            textvariable=self.port_var,
            width=18,
            state='readonly'
        )
        self.port_dropdown.pack(side='left', padx=(0, 2))

        # Refresh ports button
        self.refresh_btn = ttk.Button(
            port_frame,
            text="🔄",
            width=2,
            command=self._scan_ports
        )
        self.refresh_btn.pack(side='left')

        ttk.Label(connection_frame, text="Baud Rate:").grid(row=0, column=2, padx=5, sticky='e')
        self.baud_var = tk.StringVar(value="115200")
        self.baud_dropdown = ttk.Combobox(
            connection_frame,
            textvariable=self.baud_var,
            width=10,
            state='readonly',
            values=['9600', '19200', '38400', '57600', '115200', '230400']
        )
        self.baud_dropdown.grid(row=0, column=3, padx=5)

        # Connect button
        self.connect_btn = ttk.Button(
            connection_frame,
            text="Connect",
            command=self._toggle_connection
        )
        self.connect_btn.grid(row=0, column=4, padx=10)

        # Status indicator
        self.status_label = ttk.Label(
            connection_frame,
            text="Disconnected",
            foreground="red",
            font=('Arial', 10, 'bold')
        )
        self.status_label.grid(row=0, column=5, padx=10)

        # Protocol indicator
        ttk.Label(connection_frame, text="Protocol:").grid(row=0, column=6, padx=5)
        self.protocol_label = ttk.Label(
            connection_frame,
            text="N/A",
            font=('Arial', 10, 'bold')
        )
        self.protocol_label.grid(row=0, column=7, padx=5)

        # Add second row for controller and drone status
        ttk.Separator(connection_frame, orient='horizontal').grid(row=1, column=0, columnspan=8, sticky='ew', pady=5)

        # Controller status
        ttk.Label(connection_frame, text="Controller Status:", font=('Arial', 9, 'bold')).grid(row=2, column=0, columnspan=2, padx=5, sticky='w')
        self.controller_status_label = ttk.Label(
            connection_frame,
            text="Not Connected",
            foreground="gray",
            font=('Arial', 9)
        )
        self.controller_status_label.grid(row=2, column=2, columnspan=2, padx=5, sticky='w')

        # Drone link status
        ttk.Label(connection_frame, text="Drone Link:", font=('Arial', 9, 'bold')).grid(row=2, column=4, padx=5, sticky='w')
        self.drone_link_label = ttk.Label(
            connection_frame,
            text="No Link",
            foreground="gray",
            font=('Arial', 9, 'bold')
        )
        self.drone_link_label.grid(row=2, column=5, padx=5, sticky='w')

        # Link quality indicator
        ttk.Label(connection_frame, text="Link Quality:", font=('Arial', 9)).grid(row=2, column=6, padx=5, sticky='w')
        self.link_quality_label = ttk.Label(
            connection_frame,
            text="---",
            font=('Arial', 9)
        )
        self.link_quality_label.grid(row=2, column=7, padx=5, sticky='w')

        # Main content area - split into two columns
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Left column - Telemetry data
        telemetry_frame = ttk.LabelFrame(content_frame, text="Telemetry Data", padding=10)
        telemetry_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Battery section
        battery_section = ttk.LabelFrame(telemetry_frame, text="Battery & Power", padding=10)
        battery_section.pack(fill='x', pady=5)

        self.voltage_label = self._create_telemetry_row(battery_section, "Voltage:", "0.00 V", 0)
        self.current_label = self._create_telemetry_row(battery_section, "Current:", "0.00 A", 1)
        self.rssi_label = self._create_telemetry_row(battery_section, "RSSI:", "0", 2)

        # Attitude section
        attitude_section = ttk.LabelFrame(telemetry_frame, text="Attitude", padding=10)
        attitude_section.pack(fill='x', pady=5)

        self.roll_label = self._create_telemetry_row(attitude_section, "Roll:", "0.0°", 0)
        self.pitch_label = self._create_telemetry_row(attitude_section, "Pitch:", "0.0°", 1)
        self.yaw_label = self._create_telemetry_row(attitude_section, "Yaw:", "0.0°", 2)

        # Status section
        status_section = ttk.LabelFrame(telemetry_frame, text="Flight Status", padding=10)
        status_section.pack(fill='x', pady=5)

        self.armed_label = self._create_telemetry_row(status_section, "Armed:", "NO", 0)
        self.mode_label = self._create_telemetry_row(status_section, "Mode:", "Unknown", 1)

        # Gimbal/Joystick section (Channels 1-4)
        gimbal_section = ttk.LabelFrame(telemetry_frame, text="Gimbals (CH1-4)", padding=10)
        gimbal_section.pack(fill='x', pady=5)

        # Create frame for two joysticks side by side
        joystick_frame = ttk.Frame(gimbal_section)
        joystick_frame.pack(fill='x', pady=5)

        # Right Gimbal (CH1 & CH2)
        right_gimbal_frame = ttk.Frame(joystick_frame)
        right_gimbal_frame.pack(side='left', padx=10)

        ttk.Label(right_gimbal_frame, text="Right Gimbal", font=('Arial', 10, 'bold')).pack()
        ttk.Label(right_gimbal_frame, text="(CH1: X, CH2: Y)", font=('Arial', 8)).pack()

        self.right_joystick = JoystickWidget(right_gimbal_frame, size=120)
        self.right_joystick.pack(pady=5)

        self.right_joystick_label = ttk.Label(right_gimbal_frame, text="X: +0.00  Y: +0.00",
                                               font=('Courier', 9))
        self.right_joystick_label.pack()

        # Throttle Gimbal (CH3 & CH4)
        throttle_gimbal_frame = ttk.Frame(joystick_frame)
        throttle_gimbal_frame.pack(side='left', padx=10)

        ttk.Label(throttle_gimbal_frame, text="Throttle Gimbal", font=('Arial', 10, 'bold')).pack()
        ttk.Label(throttle_gimbal_frame, text="(CH3: Y, CH4: X)", font=('Arial', 8)).pack()

        self.throttle_joystick = JoystickWidget(throttle_gimbal_frame, size=120)
        self.throttle_joystick.pack(pady=5)

        self.throttle_joystick_label = ttk.Label(throttle_gimbal_frame, text="X: +0.00  Y: +0.00",
                                                  font=('Courier', 9))
        self.throttle_joystick_label.pack()

        # RC Channels 5-8 section
        rc_section = ttk.LabelFrame(telemetry_frame, text="Channels 5-8 (Switches)", padding=10)
        rc_section.pack(fill='x', pady=5)

        self.rc_labels = []
        for i in range(4, 8):  # Channels 5-8
            label = self._create_telemetry_row(rc_section, f"CH{i+1}:", "----", i-4)
            self.rc_labels.append(label)

        # Right column - Data log
        log_frame = ttk.LabelFrame(content_frame, text="Data Log", padding=10)
        log_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Courier', 9)
        )
        self.log_text.pack(fill='both', expand=True)

        # Control buttons
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill='x', pady=(5, 0))

        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Request Status", command=self._request_status).pack(side='left', padx=5)

        # Debug options
        debug_frame = ttk.Frame(log_frame)
        debug_frame.pack(fill='x', pady=(5, 0))

        self.raw_data_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            debug_frame,
            text="Show Raw Data (Hex)",
            variable=self.raw_data_var,
            command=self._toggle_raw_data
        ).pack(side='left', padx=5)

        # Bytes received counter
        self.bytes_counter_label = ttk.Label(debug_frame, text="Bytes RX: 0", font=('Courier', 9))
        self.bytes_counter_label.pack(side='left', padx=10)

    def _create_telemetry_row(self, parent, label_text, value_text, row):
        """Create a telemetry data row"""
        ttk.Label(parent, text=label_text, width=12).grid(row=row, column=0, sticky='w', pady=2)
        value_label = ttk.Label(
            parent,
            text=value_text,
            font=('Arial', 11, 'bold'),
            width=20
        )
        value_label.grid(row=row, column=1, sticky='w', pady=2)
        return value_label

    def _load_config(self):
        """Load saved configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.preferred_port = config.get('last_port', '')
                    self.preferred_baud = config.get('last_baud', '115200')
                    self.log_message(f"Loaded config: preferred port = {self.preferred_port}")
            else:
                self.preferred_port = ''
                self.preferred_baud = '115200'
        except Exception as e:
            self.preferred_port = ''
            self.preferred_baud = '115200'
            print(f"Could not load config: {e}")

    def _save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'last_port': self.port_var.get(),
                'last_baud': self.baud_var.get()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")

    def _scan_ports(self):
        """Scan for available serial ports and update dropdown"""
        self.available_ports = []
        port_list = []

        # Get list of all serial ports
        ports = serial.tools.list_ports.comports()

        for port in sorted(ports):
            # Store both port device and description
            self.available_ports.append(port.device)
            # Display format: "COM2 - USB Serial Device"
            if port.description and port.description != 'n/a':
                display_name = f"{port.device} - {port.description}"
            else:
                display_name = port.device
            port_list.append(display_name)

        # Update dropdown values
        if port_list:
            self.port_dropdown['values'] = port_list

            # Try to select preferred port if available
            preferred_found = False
            if self.preferred_port:
                for i, port_device in enumerate(self.available_ports):
                    if port_device == self.preferred_port:
                        self.port_dropdown.current(i)
                        self.port_var.set(port_device)
                        preferred_found = True
                        self.log_message(f"✓ Auto-selected remembered port: {port_device}")
                        break

            # Otherwise select first port
            if not preferred_found:
                self.port_dropdown.current(0)
                self.port_var.set(self.available_ports[0])

            self.log_message(f"Found {len(port_list)} serial port(s)")
        else:
            self.port_dropdown['values'] = ["No ports found"]
            self.port_var.set("")
            self.log_message("No serial ports detected")

    def _toggle_connection(self):
        """Connect or disconnect from the serial port"""
        if self.serial_conn and self.serial_conn.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """Establish connection to the serial port"""
        # Extract just the port device name (in case dropdown shows "COM2 - Description")
        port_full = self.port_var.get()
        if ' - ' in port_full:
            port = port_full.split(' - ')[0]
        else:
            port = port_full

        if not port:
            self.log_message("Please select a port!")
            return

        baud = int(self.baud_var.get())

        self.log_message(f"Connecting to {port} at {baud} baud...")

        self.serial_conn = SerialConnection(port, baud)
        self.serial_conn.set_data_callback(self._on_serial_data)

        if self.serial_conn.connect():
            self.status_label.config(text="Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
            self.log_message(f"✓ Connected successfully!")

            # Save this port for next time
            self._save_config()

            # Request initial status
            time.sleep(0.5)
            self._request_status()
        else:
            self.log_message(f"✗ Connection failed!")
            self.serial_conn = None

    def _disconnect(self):
        """Disconnect from the serial port"""
        if self.serial_conn:
            self.serial_conn.disconnect()
            self.serial_conn = None

        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_btn.config(text="Connect")
        self.protocol_label.config(text="N/A")
        self.controller_status_label.config(text="Not Connected", foreground="gray")
        self.drone_link_label.config(text="No Link", foreground="gray")
        self.link_quality_label.config(text="---")
        self.telemetry_data['protocol'] = 'Auto-detecting...'
        self.last_telemetry_time = 0
        self.drone_connected = False
        self.bytes_received = 0
        self.log_message("Disconnected")

    def _on_serial_data(self, data: bytes):
        """Handle incoming serial data"""
        # Update byte counter
        self.bytes_received += len(data)

        # Log raw data if enabled
        if self.show_raw_data:
            hex_data = ' '.join(f'{b:02X}' for b in data)
            self.log_message(f"RX ({len(data)} bytes): {hex_data}")

        # Try to parse as BETAFPV Custom (16-byte RC packets)
        betafpv_messages = self.betafpv_custom_parser.parse(data)
        if betafpv_messages:
            self.telemetry_data['protocol'] = 'BETAFPV-Custom'
            for msg in betafpv_messages:
                self._process_betafpv_custom_message(msg)
            return  # Data successfully parsed

        # Try to parse as MSP
        msp_messages = self.msp_parser.parse(data)
        if msp_messages:
            self.telemetry_data['protocol'] = 'MSP'
            for cmd, payload in msp_messages:
                self._process_msp_message(cmd, payload)
            return  # Data successfully parsed

        # Try to parse as MAVLink
        mavlink_messages = self.mavlink_parser.parse(data)
        if mavlink_messages:
            self.telemetry_data['protocol'] = 'MAVLink'
            for msg_id, payload in mavlink_messages:
                self._process_mavlink_message(msg_id, payload)
            return  # Data successfully parsed

        # If we received data but no protocol detected, only log occasionally
        # to avoid spam (only log every 20th unrecognized packet)
        if len(data) > 0:
            if not hasattr(self, '_unrecognized_count'):
                self._unrecognized_count = 0
            self._unrecognized_count += 1

            if self._unrecognized_count % 20 == 1:  # Log first and every 20th
                hex_preview = ' '.join(f'{b:02X}' for b in data[:32])
                if len(data) > 32:
                    hex_preview += '...'
                self.log_message(f"⚠ Unrecognized data ({len(data)} bytes): {hex_preview}")

    def _process_msp_message(self, cmd, payload):
        """Process MSP protocol messages"""
        with self.update_lock:
            # Update last telemetry time for any valid message
            self.last_telemetry_time = time.time()

            if cmd == MSPCommands.MSP_ANALOG:
                data = MSPParser.parse_analog(payload)
                self.telemetry_data.update(data)
                self._update_link_status()
                self.log_message(f"MSP_ANALOG: {data}")

            elif cmd == MSPCommands.MSP_ATTITUDE:
                data = MSPParser.parse_attitude(payload)
                self.telemetry_data.update(data)
                self._update_link_status()
                self.log_message(f"MSP_ATTITUDE: {data}")

            elif cmd == MSPCommands.MSP_STATUS:
                data = MSPParser.parse_status(payload)
                self.telemetry_data.update(data)
                self._update_link_status()
                self.log_message(f"MSP_STATUS: {data}")

            elif cmd == MSPCommands.MSP_RC:
                # RC channels indicate active link
                self._update_link_status()
                self.log_message(f"MSP_RC: {len(payload)} bytes")

            else:
                self.log_message(f"MSP message {cmd}: {len(payload)} bytes")

    def _process_betafpv_custom_message(self, msg: dict):
        """Process BETAFPV custom protocol messages"""
        with self.update_lock:
            # Update last telemetry time
            self.last_telemetry_time = time.time()

            if msg.get('type') == 'RC_CHANNELS':
                # Parse the RC channels data
                data = BETAFPVCustomParser.parse_rc_channels(msg)

                # Update telemetry
                self.telemetry_data['rc_channels'] = data.get('channels', [])
                self.telemetry_data['rssi'] = data.get('rssi', 0)

                # Log first few messages, then only occasionally
                if not hasattr(self, '_rc_message_count'):
                    self._rc_message_count = 0
                self._rc_message_count += 1

                if self._rc_message_count <= 3 or self._rc_message_count % 50 == 0:
                    self.log_message(f"BETAFPV RC_CHANNELS: {data['channels'][:4]}... RSSI: {data['rssi']}")

                self._update_link_status()

    def _process_mavlink_message(self, msg_id, payload):
        """Process MAVLink protocol messages"""
        with self.update_lock:
            # Update last telemetry time for any valid message
            self.last_telemetry_time = time.time()

            if msg_id == MAVLinkMessages.HEARTBEAT:
                data = MAVLinkParser.parse_heartbeat(payload)
                self.telemetry_data.update(data)
                self._update_link_status()
                self.log_message(f"HEARTBEAT: {data}")

            elif msg_id == MAVLinkMessages.ATTITUDE:
                data = MAVLinkParser.parse_attitude(payload)
                if data:
                    # Convert radians to degrees
                    data['roll'] = data['roll'] * 57.2958
                    data['pitch'] = data['pitch'] * 57.2958
                    data['yaw'] = data['yaw'] * 57.2958
                    self.telemetry_data.update(data)
                    self._update_link_status()
                    self.log_message(f"ATTITUDE: {data}")

            elif msg_id == MAVLinkMessages.RC_CHANNELS:
                data = MAVLinkParser.parse_rc_channels(payload)
                self.telemetry_data['rc_channels'] = data.get('channels', [])
                self.telemetry_data['rssi'] = data.get('rssi', 0)
                self._update_link_status()
                self.log_message(f"RC_CHANNELS: {data}")

            elif msg_id == MAVLinkMessages.SYS_STATUS:
                data = MAVLinkParser.parse_sys_status(payload)
                self.telemetry_data.update(data)
                self._update_link_status()
                self.log_message(f"SYS_STATUS: {data}")

            elif msg_id == MAVLinkMessages.IMU:
                data = MAVLinkParser.parse_imu(payload)
                self._update_link_status()
                self.log_message(f"IMU: {data}")

            elif msg_id == MAVLinkMessages.FIRMWARE_INFO:
                self._update_link_status()
                self.log_message(f"FIRMWARE_INFO: {len(payload)} bytes")

            elif msg_id == MAVLinkMessages.LOCAL_POSITION:
                self._update_link_status()
                self.log_message(f"LOCAL_POSITION: {len(payload)} bytes")

            elif msg_id == MAVLinkMessages.ALTITUDE:
                self._update_link_status()
                self.log_message(f"ALTITUDE: {len(payload)} bytes")

            elif msg_id == MAVLinkMessages.PID:
                self._update_link_status()
                self.log_message(f"PID: {len(payload)} bytes")

            elif msg_id == MAVLinkMessages.RATE:
                self._update_link_status()
                self.log_message(f"RATE: {len(payload)} bytes")

            else:
                self.log_message(f"⚠ Unknown MAVLink message ID {msg_id}: {len(payload)} bytes")

    def _update_link_status(self):
        """Update controller and drone link status based on telemetry"""
        # Controller is connected if we're receiving data
        if self.serial_conn and self.serial_conn.is_connected:
            self.telemetry_data['controller_connected'] = True
        else:
            self.telemetry_data['controller_connected'] = False
            self.telemetry_data['drone_link'] = False
            return

        # Check if drone is connected based on:
        # 1. Recent telemetry data (within last 2 seconds)
        # 2. RSSI > 0 (signal present)
        # 3. Active RC channels or attitude data
        current_time = time.time()
        time_since_last_data = current_time - self.last_telemetry_time

        rssi = self.telemetry_data.get('rssi', 0)
        rc_channels = self.telemetry_data.get('rc_channels', [])

        # Drone is considered linked if:
        # - We received data within last 2 seconds AND
        # - RSSI > 0 OR we have active RC channels
        if time_since_last_data < 2.0 and (rssi > 0 or len(rc_channels) > 0):
            self.telemetry_data['drone_link'] = True

            # Calculate link quality percentage based on RSSI
            # RSSI typically ranges from 0-255, but can vary by protocol
            if rssi > 0:
                # Normalize RSSI to 0-100%
                if rssi <= 100:
                    link_quality = rssi
                else:
                    link_quality = min(100, int((rssi / 255.0) * 100))
                self.telemetry_data['link_quality'] = link_quality
            else:
                self.telemetry_data['link_quality'] = 0
        else:
            self.telemetry_data['drone_link'] = False
            self.telemetry_data['link_quality'] = 0

    def _request_status(self):
        """Request status data from the device"""
        if not self.serial_conn or not self.serial_conn.is_connected:
            return

        protocol = self.telemetry_data.get('protocol', 'Auto-detecting...')

        if protocol == 'MSP' or protocol == 'Auto-detecting...':
            # Send MSP status requests
            requests = [
                MSPCommands.MSP_STATUS,
                MSPCommands.MSP_ANALOG,
                MSPCommands.MSP_ATTITUDE,
                MSPCommands.MSP_RC
            ]

            for cmd in requests:
                msg = MSPParser.create_request(cmd)
                self.serial_conn.send_data(msg)
                time.sleep(0.05)

            self.log_message("Sent MSP status requests")

    def _update_ui_loop(self):
        """Update UI with latest telemetry data"""
        with self.update_lock:
            # Check link timeout (if no data for 2+ seconds, connection is lost)
            if self.last_telemetry_time > 0:
                time_since_data = time.time() - self.last_telemetry_time
                if time_since_data > 2.0:
                    self.telemetry_data['drone_link'] = False
                    self.telemetry_data['link_quality'] = 0

            # Update controller status
            controller_connected = self.telemetry_data.get('controller_connected', False)
            if controller_connected:
                self.controller_status_label.config(text="Connected & Active", foreground="green")
            else:
                self.controller_status_label.config(text="Not Connected", foreground="gray")

            # Update drone link status
            drone_link = self.telemetry_data.get('drone_link', False)
            if drone_link:
                self.drone_link_label.config(text="✓ LINKED", foreground="green")
            else:
                self.drone_link_label.config(text="✗ No Link", foreground="red")

            # Update link quality
            link_quality = self.telemetry_data.get('link_quality', 0)
            if link_quality > 0:
                # Color code based on quality
                if link_quality >= 70:
                    color = "green"
                elif link_quality >= 40:
                    color = "orange"
                else:
                    color = "red"
                self.link_quality_label.config(text=f"{link_quality}%", foreground=color)
            else:
                self.link_quality_label.config(text="---", foreground="gray")

            # Update battery & power
            voltage = self.telemetry_data.get('voltage', 0.0)
            current = self.telemetry_data.get('current', 0.0)
            rssi = self.telemetry_data.get('rssi', 0)

            self.voltage_label.config(text=f"{voltage:.2f} V")
            self.current_label.config(text=f"{current:.2f} A")
            self.rssi_label.config(text=f"{rssi}")

            # Update attitude
            roll = self.telemetry_data.get('roll', 0.0)
            pitch = self.telemetry_data.get('pitch', 0.0)
            yaw = self.telemetry_data.get('yaw', 0.0)

            self.roll_label.config(text=f"{roll:.1f}°")
            self.pitch_label.config(text=f"{pitch:.1f}°")
            self.yaw_label.config(text=f"{yaw:.1f}°")

            # Update status
            armed = self.telemetry_data.get('armed', False)
            mode = self.telemetry_data.get('flight_mode', 'Unknown')

            self.armed_label.config(
                text="YES" if armed else "NO",
                foreground="red" if armed else "green"
            )
            self.mode_label.config(text=str(mode))

            # Update RC channels and joysticks
            rc_channels = self.telemetry_data.get('rc_channels', [])

            # Update joystick visualizations (CH1-4)
            if len(rc_channels) >= 4:
                # Right Gimbal: CH1 (X) and CH2 (Y)
                ch1 = rc_channels[0] if len(rc_channels) > 0 else 1500
                ch2 = rc_channels[1] if len(rc_channels) > 1 else 1500
                self.right_joystick.update_from_rc(ch1, ch2)
                self.right_joystick_label.config(text=self.right_joystick.get_position_text())

                # Throttle Gimbal: CH4 (X) and CH3 (Y)
                # Note: CH3 is typically throttle (vertical), CH4 is yaw (horizontal)
                ch3 = rc_channels[2] if len(rc_channels) > 2 else 1500
                ch4 = rc_channels[3] if len(rc_channels) > 3 else 1500
                self.throttle_joystick.update_from_rc(ch4, ch3)
                self.throttle_joystick_label.config(text=self.throttle_joystick.get_position_text())

            # Update channels 5-8 display
            for i, label in enumerate(self.rc_labels):
                ch_index = i + 4  # Channels 5-8 (indices 4-7)
                if ch_index < len(rc_channels):
                    label.config(text=f"{rc_channels[ch_index]}")
                else:
                    label.config(text="----")

            # Update protocol
            protocol = self.telemetry_data.get('protocol', 'N/A')
            self.protocol_label.config(text=protocol)

            # Update bytes counter
            self.bytes_counter_label.config(text=f"Bytes RX: {self.bytes_received}")

        # Schedule next update
        self.root.after(100, self._update_ui_loop)

    def log_message(self, message):
        """Add message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {message}\n"

        # Update log text (must be done in main thread)
        def update_log():
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)

            # Limit log size
            if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
                self.log_text.delete('1.0', '100.0')

        self.root.after(0, update_log)

    def _toggle_raw_data(self):
        """Toggle raw data display"""
        self.show_raw_data = self.raw_data_var.get()
        if self.show_raw_data:
            self.log_message("✓ Raw data logging enabled")
        else:
            self.log_message("Raw data logging disabled")

    def _clear_log(self):
        """Clear the log text"""
        self.log_text.delete('1.0', tk.END)
        self.log_message("Log cleared")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BETAFPVConfiguratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
