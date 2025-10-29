"""
Drone Control GUI for Air65 via MSP
Manual control interface with live camera feed and telemetry
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import serial.tools.list_ports
from serial_comm import SerialConnection
from msp_parser import MSPParser, MSPCommands
from msp_control_loop import MSPControlLoop
from camera_widget import CameraWidget


class DroneControlGUI:
    """Main GUI for drone control via MSP"""

    def __init__(self, root):
        self.root = root
        self.root.title("Air65 Drone Controller - MSP Control")
        self.root.geometry("1000x720")

        # Serial connection
        self.serial_conn = None
        self.available_ports = []
        self.preferred_port = None

        # MSP components
        self.msp_parser = MSPParser()
        self.control_loop = None

        # Telemetry data
        self.telemetry_data = {
            'voltage': 0.0,
            'current': 0.0,
            'mah_drawn': 0,
            'rssi': 0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0,
            'armed': False,
            'connected': False
        }
        self.update_lock = threading.Lock()

        # UI state
        self.control_enabled = False

        # Slider variables (initialize before building UI)
        self.slider_vars = []

        # Value labels for sliders
        self.throttle_value_label = None
        self.roll_value_label = None
        self.pitch_value_label = None
        self.yaw_value_label = None

        # Build UI
        self._build_ui()

        # Start telemetry request loop
        self._start_telemetry_loop()

    def _build_ui(self):
        """Build the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Air65 Drone Controller - MSP",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=5)

        # Connection section
        self._build_connection_section(main_frame)

        # Camera feed section
        self._build_camera_section(main_frame)

        # Telemetry section
        self._build_telemetry_section(main_frame)

        # Control section
        self._build_control_section(main_frame)

        # Buttons section
        self._build_buttons_section(main_frame)

        # Status bar
        self._build_status_bar(main_frame)

    def _build_connection_section(self, parent):
        """Build connection controls"""
        connection_frame = ttk.LabelFrame(parent, text="Connection", padding="10")
        connection_frame.pack(fill='x', pady=5)

        # Port selection
        ttk.Label(connection_frame, text="Serial Port:").pack(side='left', padx=5)

        self.port_var = tk.StringVar()
        self.port_dropdown = ttk.Combobox(
            connection_frame,
            textvariable=self.port_var,
            width=50,
            state='readonly'
        )
        self.port_dropdown.pack(side='left', padx=5)

        # Refresh button
        refresh_btn = ttk.Button(
            connection_frame,
            text="🔄",
            command=self._scan_ports,
            width=3
        )
        refresh_btn.pack(side='left', padx=2)

        # Connect button
        self.connect_btn = ttk.Button(
            connection_frame,
            text="Connect",
            command=self._toggle_connection,
            width=15
        )
        self.connect_btn.pack(side='left', padx=5)

        # Connection status indicator
        self.conn_status_label = ttk.Label(
            connection_frame,
            text="● Disconnected",
            foreground='red',
            font=('Arial', 10, 'bold')
        )
        self.conn_status_label.pack(side='left', padx=10)

        # Initial port scan
        self._scan_ports()

    def _build_camera_section(self, parent):
        """Build camera feed section"""
        camera_frame = ttk.LabelFrame(parent, text="FPV Camera Feed", padding="5")
        camera_frame.pack(fill='x', pady=5)

        # Camera widget - smaller size
        self.camera_widget = CameraWidget(
            camera_frame,
            width=480,
            height=360,
            camera_index=0,
            fps=30
        )
        self.camera_widget.pack()

    def _build_telemetry_section(self, parent):
        """Build telemetry display"""
        telemetry_frame = ttk.LabelFrame(parent, text="Telemetry", padding="5")
        telemetry_frame.pack(fill='x', pady=3)

        # Create telemetry grid
        tel_grid = ttk.Frame(telemetry_frame)
        tel_grid.pack(fill='x')

        # Battery info
        battery_frame = ttk.Frame(tel_grid)
        battery_frame.grid(row=0, column=0, padx=10, pady=2, sticky='w')
        ttk.Label(battery_frame, text="Battery:", font=('Arial', 10, 'bold')).pack(side='left')
        self.voltage_label = ttk.Label(battery_frame, text="0.00 V", font=('Arial', 10))
        self.voltage_label.pack(side='left', padx=5)
        self.current_label = ttk.Label(battery_frame, text="0.00 A", font=('Arial', 10))
        self.current_label.pack(side='left', padx=5)
        self.mah_label = ttk.Label(battery_frame, text="0 mAh", font=('Arial', 10))
        self.mah_label.pack(side='left', padx=5)

        # Signal strength
        signal_frame = ttk.Frame(tel_grid)
        signal_frame.grid(row=0, column=1, padx=10, pady=2, sticky='w')
        ttk.Label(signal_frame, text="RSSI:", font=('Arial', 10, 'bold')).pack(side='left')
        self.rssi_label = ttk.Label(signal_frame, text="0", font=('Arial', 10))
        self.rssi_label.pack(side='left', padx=5)

        # Attitude
        attitude_frame = ttk.Frame(tel_grid)
        attitude_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=2, sticky='w')
        ttk.Label(attitude_frame, text="Attitude:", font=('Arial', 10, 'bold')).pack(side='left')
        self.roll_label = ttk.Label(attitude_frame, text="Roll: 0.0°", font=('Arial', 10))
        self.roll_label.pack(side='left', padx=5)
        self.pitch_label = ttk.Label(attitude_frame, text="Pitch: 0.0°", font=('Arial', 10))
        self.pitch_label.pack(side='left', padx=5)
        self.yaw_label = ttk.Label(attitude_frame, text="Yaw: 0°", font=('Arial', 10))
        self.yaw_label.pack(side='left', padx=5)

        # Armed status
        status_frame = ttk.Frame(tel_grid)
        status_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky='w')
        ttk.Label(status_frame, text="Status:", font=('Arial', 10, 'bold')).pack(side='left')
        self.armed_label = ttk.Label(
            status_frame,
            text="DISARMED",
            font=('Arial', 12, 'bold'),
            foreground='orange'
        )
        self.armed_label.pack(side='left', padx=5)

    def _build_control_section(self, parent):
        """Build manual control sliders"""
        control_frame = ttk.LabelFrame(parent, text="Manual Controls", padding="5")
        control_frame.pack(fill='x', pady=3)

        # Initialize slider variables BEFORE creating sliders
        self.slider_vars = [
            tk.IntVar(value=1000),  # Throttle
            tk.IntVar(value=1500),  # Roll
            tk.IntVar(value=1500),  # Pitch
            tk.IntVar(value=1500)   # Yaw
        ]

        # Throttle
        self._create_slider(control_frame, "Throttle", 0, 1000)

        # Roll
        self._create_slider(control_frame, "Roll", 1, 1500)

        # Pitch
        self._create_slider(control_frame, "Pitch", 2, 1500)

        # Yaw
        self._create_slider(control_frame, "Yaw", 3, 1500)

    def _create_slider(self, parent, label, row, default_value):
        """Create a control slider"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)

        # Label
        lbl = ttk.Label(frame, text=f"{label}:", width=10, font=('Arial', 10))
        lbl.pack(side='left', padx=5)

        # Use the pre-created slider variable
        var = self.slider_vars[row]

        # Slider
        slider = ttk.Scale(
            frame,
            from_=1000,
            to=2000,
            orient='horizontal',
            variable=var,
            command=lambda v: self._on_slider_change(row)
        )
        slider.pack(side='left', fill='x', expand=True, padx=5)

        # Value label
        value_label = ttk.Label(frame, text=str(default_value), width=6, font=('Arial', 10))
        value_label.pack(side='left', padx=5)

        # Store value label reference
        if row == 0:
            self.throttle_value_label = value_label
        elif row == 1:
            self.roll_value_label = value_label
        elif row == 2:
            self.pitch_value_label = value_label
        elif row == 3:
            self.yaw_value_label = value_label

        # Center button (not for throttle)
        if row != 0:
            center_btn = ttk.Button(
                frame,
                text="Center",
                command=lambda: self._center_slider(row),
                width=8
            )
            center_btn.pack(side='left', padx=2)

    def _on_slider_change(self, slider_index):
        """Handle slider value change"""
        value = self.slider_vars[slider_index].get()

        # Update value label
        if slider_index == 0:
            self.throttle_value_label.config(text=str(value))
        elif slider_index == 1:
            self.roll_value_label.config(text=str(value))
        elif slider_index == 2:
            self.pitch_value_label.config(text=str(value))
        elif slider_index == 3:
            self.yaw_value_label.config(text=str(value))

        # Update control loop if connected
        if self.control_loop and self.control_enabled:
            self._update_control_loop()

    def _center_slider(self, slider_index):
        """Center a slider to 1500"""
        self.slider_vars[slider_index].set(1500)
        self._on_slider_change(slider_index)

    def _build_buttons_section(self, parent):
        """Build control buttons"""
        button_frame = ttk.Frame(parent, padding="5")
        button_frame.pack(fill='x', pady=5)

        # Enable/Disable controls
        self.enable_btn = ttk.Button(
            button_frame,
            text="Enable Controls",
            command=self._toggle_controls,
            state='disabled',
            width=15
        )
        self.enable_btn.pack(side='left', padx=5)

        # ARM button
        self.arm_btn = ttk.Button(
            button_frame,
            text="ARM",
            command=self._arm_drone,
            state='disabled',
            width=12
        )
        self.arm_btn.pack(side='left', padx=5)

        # DISARM button
        self.disarm_btn = ttk.Button(
            button_frame,
            text="DISARM",
            command=self._disarm_drone,
            state='disabled',
            width=12
        )
        self.disarm_btn.pack(side='left', padx=5)

        # Emergency stop button (RED)
        self.emergency_btn = ttk.Button(
            button_frame,
            text="EMERGENCY STOP",
            command=self._emergency_stop,
            state='disabled',
            width=18
        )
        self.emergency_btn.pack(side='left', padx=5)

    def _build_status_bar(self, parent):
        """Build status bar at bottom"""
        status_frame = ttk.Frame(parent, relief='sunken', borderwidth=1)
        status_frame.pack(fill='x', side='bottom', pady=5)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready - Connect to drone",
            font=('Arial', 9)
        )
        self.status_label.pack(side='left', padx=5, pady=2)

        # Control loop stats
        self.stats_label = ttk.Label(
            status_frame,
            text="Control Loop: Stopped",
            font=('Arial', 9)
        )
        self.stats_label.pack(side='right', padx=5, pady=2)

    def _scan_ports(self):
        """Scan for available serial ports and update dropdown"""
        self.available_ports = []
        port_list = []

        # Get list of all serial ports
        ports = serial.tools.list_ports.comports()

        for port in sorted(ports):
            # Store both port device and description
            self.available_ports.append(port.device)
            # Display format: "/dev/cu.usbmodem123 - Betaflight STM32G47x"
            if port.description and port.description != 'n/a':
                display_name = f"{port.device} - {port.description}"
            else:
                display_name = port.device
            port_list.append(display_name)

        # Update dropdown values
        if port_list:
            self.port_dropdown['values'] = port_list

            # Auto-select first usbmodem port (likely the drone)
            selected = False
            for i, port_device in enumerate(self.available_ports):
                if 'usbmodem' in port_device.lower() or 'ttyUSB' in port_device or 'ttyACM' in port_device:
                    self.port_dropdown.current(i)
                    selected = True
                    self._log(f"Auto-selected: {port_list[i]}")
                    break

            # Otherwise select first port
            if not selected:
                self.port_dropdown.current(0)

            self._log(f"Found {len(port_list)} serial port(s)")
        else:
            self.port_dropdown['values'] = ["No ports found"]
            self.port_var.set("")
            self._log("No serial ports detected - Connect drone and click refresh")

    def _toggle_connection(self):
        """Connect or disconnect from drone"""
        if self.serial_conn and self.serial_conn.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """Connect to drone via serial"""
        # Extract port device from dropdown (format: "port - description")
        port_full = self.port_var.get()
        if ' - ' in port_full:
            port = port_full.split(' - ')[0]
        else:
            port = port_full

        if not port or port == "No ports found":
            self.status_label.config(text="Please select a valid port")
            self._log("No port selected - scan for ports first")
            return

        try:
            self.serial_conn = SerialConnection(port=port, baudrate=115200)
            self.serial_conn.set_data_callback(self._on_serial_data)

            if self.serial_conn.connect():
                # Create control loop
                self.control_loop = MSPControlLoop(self.serial_conn)

                # Update UI
                self.connect_btn.config(text="Disconnect")
                self.conn_status_label.config(text="● Connected", foreground='green')
                self.enable_btn.config(state='normal')
                self.status_label.config(text=f"Connected to {port}")

                with self.update_lock:
                    self.telemetry_data['connected'] = True

                self._log("Connected to drone at {}".format(port))

            else:
                self.status_label.config(text="Failed to connect to {}".format(port))
                self._log("Connection failed")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self._log(f"Connection error: {str(e)}")

    def _disconnect(self):
        """Disconnect from drone"""
        # Stop control loop first
        if self.control_loop:
            self.control_loop.stop()
            self.control_loop = None

        # Disconnect serial
        if self.serial_conn:
            self.serial_conn.disconnect()
            self.serial_conn = None

        # Update UI
        self.connect_btn.config(text="Connect")
        self.conn_status_label.config(text="● Disconnected", foreground='red')
        self.enable_btn.config(state='disabled')
        self.arm_btn.config(state='disabled')
        self.disarm_btn.config(state='disabled')
        self.emergency_btn.config(state='disabled')
        self.control_enabled = False
        self.enable_btn.config(text="Enable Controls")

        with self.update_lock:
            self.telemetry_data['connected'] = False

        self.status_label.config(text="Disconnected")
        self._log("Disconnected from drone")

    def _toggle_controls(self):
        """Enable or disable manual controls"""
        if self.control_enabled:
            # Disable controls
            self.control_loop.stop()
            self.control_enabled = False
            self.enable_btn.config(text="Enable Controls")
            self.arm_btn.config(state='disabled')
            self.disarm_btn.config(state='disabled')
            self.emergency_btn.config(state='disabled')
            self.status_label.config(text="Controls disabled")
            self._log("Controls disabled")
        else:
            # Enable controls
            self.control_loop.start()
            self.control_enabled = True
            self.enable_btn.config(text="Disable Controls")
            self.arm_btn.config(state='normal')
            self.disarm_btn.config(state='normal')
            self.emergency_btn.config(state='normal')
            self.status_label.config(text="Controls enabled - sending at 50 Hz")
            self._log("Controls enabled - Control loop running at 50 Hz")

    def _update_control_loop(self):
        """Update control loop with current slider values"""
        if not self.control_loop:
            return

        throttle = self.slider_vars[0].get()
        roll = self.slider_vars[1].get()
        pitch = self.slider_vars[2].get()
        yaw = self.slider_vars[3].get()

        self.control_loop.set_channels(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            throttle=throttle
        )

    def _arm_drone(self):
        """Send ARM command"""
        if self.control_loop:
            self.control_loop.arm()
            channels = self.control_loop.get_channels()
            self.status_label.config(text="Arming command sent - Check drone status")
            self._log(f"ARM command sent - Channels: Throttle={channels[3]}, AUX1={channels[4]}")

    def _disarm_drone(self):
        """Send DISARM command"""
        if self.control_loop:
            self.control_loop.disarm()
            self.status_label.config(text="Disarming command sent")
            self._log("DISARM command sent")

    def _emergency_stop(self):
        """Emergency stop - reset all controls"""
        if self.control_loop:
            self.control_loop.emergency_stop()

            # Reset sliders
            self.slider_vars[0].set(1000)  # Throttle to min
            self.slider_vars[1].set(1500)  # Roll center
            self.slider_vars[2].set(1500)  # Pitch center
            self.slider_vars[3].set(1500)  # Yaw center

            self.status_label.config(text="EMERGENCY STOP ACTIVATED")
            self._log("EMERGENCY STOP - All channels reset")

    def _on_serial_data(self, data: bytes):
        """Handle incoming serial data"""
        # Parse MSP messages
        messages = self.msp_parser.parse(data)

        for cmd, payload in messages:
            if cmd == MSPCommands.MSP_ANALOG:
                analog_data = MSPParser.parse_analog(payload)
                with self.update_lock:
                    self.telemetry_data.update(analog_data)

            elif cmd == MSPCommands.MSP_ATTITUDE:
                attitude_data = MSPParser.parse_attitude(payload)
                with self.update_lock:
                    self.telemetry_data.update(attitude_data)

            elif cmd == MSPCommands.MSP_STATUS:
                status_data = MSPParser.parse_status(payload)
                with self.update_lock:
                    self.telemetry_data.update(status_data)

    def _start_telemetry_loop(self):
        """Start requesting telemetry periodically"""
        self._request_telemetry()
        self._update_telemetry_display()
        self._update_stats_display()

    def _request_telemetry(self):
        """Request telemetry from drone"""
        if self.serial_conn and self.serial_conn.is_connected:
            # Request status
            self.serial_conn.send_data(MSPParser.create_request(MSPCommands.MSP_STATUS))

            # Request analog (battery)
            self.serial_conn.send_data(MSPParser.create_request(MSPCommands.MSP_ANALOG))

            # Request attitude
            self.serial_conn.send_data(MSPParser.create_request(MSPCommands.MSP_ATTITUDE))

        # Schedule next request
        self.root.after(200, self._request_telemetry)  # Request every 200ms

    def _update_telemetry_display(self):
        """Update telemetry labels"""
        with self.update_lock:
            data = self.telemetry_data.copy()

        # Update battery
        self.voltage_label.config(text=f"{data['voltage']:.2f} V")
        self.current_label.config(text=f"{data['current']:.2f} A")
        self.mah_label.config(text=f"{data['mah_drawn']} mAh")

        # Update RSSI
        self.rssi_label.config(text=str(data['rssi']))

        # Update attitude
        self.roll_label.config(text=f"Roll: {data['roll']:.1f}°")
        self.pitch_label.config(text=f"Pitch: {data['pitch']:.1f}°")
        self.yaw_label.config(text=f"Yaw: {data['yaw']}°")

        # Update armed status
        if data['armed']:
            self.armed_label.config(text="ARMED", foreground='red')
        else:
            self.armed_label.config(text="DISARMED", foreground='orange')

        # Schedule next update
        self.root.after(100, self._update_telemetry_display)  # Update every 100ms

    def _update_stats_display(self):
        """Update control loop statistics"""
        if self.control_loop:
            stats = self.control_loop.get_stats()
            if stats['running']:
                self.stats_label.config(
                    text=f"Control Loop: {stats['frequency']:.1f} Hz | Commands: {stats['commands_sent']}"
                )
            else:
                self.stats_label.config(text="Control Loop: Stopped")
        else:
            self.stats_label.config(text="Control Loop: Not initialized")

        # Schedule next update
        self.root.after(500, self._update_stats_display)  # Update every 500ms

    def _log(self, message):
        """Log message (can be extended to show in GUI)"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def cleanup(self):
        """Cleanup on exit"""
        if self.control_loop:
            self.control_loop.stop()

        if self.serial_conn:
            self.serial_conn.disconnect()

        if hasattr(self, 'camera_widget'):
            self.camera_widget.cleanup()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DroneControlGUI(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
