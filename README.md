# BETAFPV Configurator - UltraThink Edition

A Python GUI application for reading and monitoring data from BETAFPV devices via serial port. Supports both **MSP (MultiWii Serial Protocol)** and **MAVLink 1.0** protocols.

## Features

- **Auto port detection**: Automatically scans and lists all available serial ports with descriptions
- **One-click refresh**: Rescan for ports without restarting the app
- **Dual connection monitoring**: Shows both PC-to-Controller and Controller-to-Drone link status
- **Real-time link quality indicator**: Color-coded signal strength display
- Real-time telemetry monitoring
- Support for MSP and MAVLink protocols with auto-detection
- Battery voltage, current, and RSSI monitoring
- Attitude display (Roll, Pitch, Yaw)
- RC channel monitoring (8 channels)
- Flight status indication (armed/disarmed)
- Data logging with timestamps
- Dropdown menus for port and baud rate selection
- Automatic link timeout detection (2-second threshold)

## Hardware Configuration

- **Device**: BETAFPV 2SE Radio Transmitter
- **Port**: COM2 (configurable)
- **Baud Rate**: 115200 (configurable)

## Installation

### Option 1: Using Anaconda (Recommended)

1. Create the Anaconda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate betafpv-ultrathink
```

3. Run the application:
```bash
python main.py
```

### Option 2: Using pip

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. **Connect Your Device**:
   - Plug in your BETAFPV 2SE transmitter to your computer
   - Verify it's connected to COM2 (or note the correct port)

2. **Launch the Application**:
   ```bash
   python main.py
   # Or use the convenience script:
   ./run.sh          # Mac/Linux
   run.bat           # Windows
   ```

3. **Configure Connection**:
   - Select your COM port from the dropdown menu (auto-detects all available ports)
   - Click the 🔄 refresh button to rescan for ports if needed
   - Select baud rate from dropdown (default: 115200)
   - Click "Connect"

4. **Monitor Data**:
   - The GUI will automatically detect whether your device uses MSP or MAVLink protocol
   - Real-time telemetry data will be displayed
   - All messages are logged in the data log panel

5. **Request Status**:
   - Click "Request Status" to manually query the device for updated information
   - This sends protocol-specific status request messages

## GUI Overview

### Connection Settings

**Top Row:**
- **Port Dropdown**: Auto-detects and lists all available serial ports with descriptions
  - Displays format: "COM2 - USB Serial Device" or "/dev/ttyUSB0 - CP2102 USB to UART"
  - 🔄 **Refresh Button**: Click to rescan for newly connected devices
- **Baud Rate Dropdown**: Select communication speed (default: 115200)
  - Options: 9600, 19200, 38400, 57600, 115200, 230400
- **Connect/Disconnect**: Toggle connection to selected port
- **Status Indicator**: PC connection to controller (green=connected, red=disconnected)
- **Protocol**: Displays detected protocol (MSP or MAVLink)

**Bottom Row - Controller & Link Status:**
- **Controller Status**: Shows if the controller is connected and actively transmitting data
  - *Not Connected* (gray) - No controller detected
  - *Connected & Active* (green) - Controller is communicating with PC

- **Drone Link**: Shows if the controller has an active radio link to a drone
  - *✗ No Link* (red) - Controller not connected to any drone
  - *✓ LINKED* (green) - Active connection between controller and drone

- **Link Quality**: Signal strength as a percentage with color coding
  - 70-100% (green) - Excellent signal
  - 40-69% (orange) - Fair signal
  - 1-39% (red) - Poor signal
  - --- (gray) - No signal

### Telemetry Data Panel

#### Battery & Power
- **Voltage**: Battery voltage in volts
- **Current**: Current draw in amperes
- **RSSI**: Received Signal Strength Indicator

#### Attitude
- **Roll**: Roll angle in degrees
- **Pitch**: Pitch angle in degrees
- **Yaw**: Yaw/heading angle in degrees

#### Flight Status
- **Armed**: YES (red) when armed, NO (green) when disarmed
- **Mode**: Current flight mode

#### RC Channels
- Displays values for 8 RC channels
- Shows raw PWM values or channel data

### Data Log
- Real-time log of all received messages
- Timestamps for each entry
- Clear Log button to reset
- Auto-scrolls to latest messages

## Protocol Support

### MSP (MultiWii Serial Protocol)
The application supports MSP V1 protocol with the following messages:
- `MSP_STATUS` - Flight status and sensor data
- `MSP_ANALOG` - Battery, current, RSSI
- `MSP_ATTITUDE` - Roll, pitch, yaw angles
- `MSP_RC` - RC channel inputs
- `MSP_MOTOR` - Motor outputs
- And more...

### MAVLink 1.0
The application supports MAVLink with the following messages:
- `HEARTBEAT` - System heartbeat
- `SYS_STATUS` - System status and battery
- `ATTITUDE` - Attitude information
- `RC_CHANNELS` - RC channel values
- `IMU` - Inertial measurement unit data
- And more...

## File Structure

```
drone_simulator/
├── src/                       # Source code
│   ├── betafpv_gui.py        # Main GUI application
│   ├── serial_comm.py        # Serial communication handler
│   ├── msp_parser.py         # MSP protocol parser
│   ├── mavlink_parser.py     # MAVLink protocol parser
│   ├── betafpv_custom_parser.py  # Custom BETAFPV protocol parser
│   └── joystick_widget.py    # 2D joystick visualization widget
├── docs/                      # Documentation
│   ├── BETAFPV_PROTOCOL.md   # Protocol documentation
│   ├── JOYSTICK_GUIDE.md     # Joystick usage guide
│   └── TROUBLESHOOTING.md    # Troubleshooting guide
├── tests/                     # Test files
│   └── test_serial_raw.py    # Serial communication tests
├── main.py                    # Application launcher
├── run.sh                     # Quick start script (Mac/Linux)
├── run.bat                    # Quick start script (Windows)
├── environment.yml            # Anaconda environment config
├── requirements.txt           # pip requirements
├── CHANGELOG.md               # Version history
└── README.md                  # This file
```

## Troubleshooting

### Connection Issues

1. **"Failed to connect" error**:
   - Verify the device is plugged in
   - Check the COM port number in Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)
   - Ensure no other application is using the serial port
   - Try unplugging and replugging the device

2. **No data received**:
   - Check baud rate is set to 115200
   - Verify the device is powered on
   - Try clicking "Request Status" to manually request data
   - Check the data log for any error messages

3. **Permission denied (Linux/Mac)**:
   ```bash
   sudo chmod 666 /dev/ttyUSB0  # Replace with your port
   # Or add user to dialout group:
   sudo usermod -a -G dialout $USER
   # Then logout and login again
   ```

4. **pyserial not found**:
   - Make sure you activated the Anaconda environment or virtual environment
   - Reinstall dependencies: `pip install -r requirements.txt`

## Advanced Usage

### Selecting the Correct Port

**The GUI now auto-detects all available serial ports!**

1. **Automatic Detection**: When you launch the app, it automatically scans for all connected serial devices
2. **Port Dropdown**: Select your BETAFPV device from the dropdown list
   - Each port shows a description (e.g., "COM2 - USB Serial Device")
3. **Refresh Ports**: If you plug in your device after launching the app, click the 🔄 button to rescan
4. **Windows**: Ports appear as COM1, COM2, COM3, etc.
5. **Mac/Linux**: Ports appear as /dev/tty.*, /dev/cu.*, /dev/ttyUSB*, etc.

### Port Not Detected?

If your BETAFPV device doesn't appear in the dropdown:

**On Windows:**
1. Open Device Manager → Ports (COM & LPT)
2. Look for "USB Serial Device" or similar
3. If you see a yellow warning icon, update the driver
4. Click 🔄 refresh button in the app

**On Linux:**
1. Check permissions: `ls -l /dev/ttyUSB0` or `ls -l /dev/ttyACM0`
2. Add user to dialout group: `sudo usermod -a -G dialout $USER`
3. Logout and login again
4. Click 🔄 refresh button in the app

**On Mac:**
1. List ports: `ls /dev/tty.* /dev/cu.*`
2. Look for `/dev/tty.usbserial-*` or `/dev/cu.usbserial-*`
3. Check if drivers are installed (may need CP210x or FTDI drivers)
4. Click 🔄 refresh button in the app

### Custom Baud Rates
While 115200 is standard, you can change the baud rate if your device uses a different speed:
- 9600
- 19200
- 38400
- 57600
- 115200 (default)
- 230400

## Development

### Adding New Message Parsers

To add support for additional MSP or MAVLink messages:

1. **For MSP**: Edit `src/msp_parser.py`
   - Add message ID to `MSPCommands` enum
   - Create a static parser method
   - Call it in `_process_msp_message()` in `src/betafpv_gui.py`

2. **For MAVLink**: Edit `src/mavlink_parser.py`
   - Add message ID to `MAVLinkMessages` enum
   - Create a static parser method
   - Call it in `_process_mavlink_message()` in `src/betafpv_gui.py`

### Contributing
Feel free to fork and submit pull requests!

## References

- [BETAFPV Configurator](https://github.com/BETAFPV/BETAFPV_Configurator)
- [MSP Protocol Documentation](https://github.com/iNavFlight/inav/wiki/MSP-V2)
- [MAVLink Protocol](https://mavlink.io/)

## License

This project is provided as-is for educational and personal use.

## Author

UltraThink Edition - Custom BETAFPV configuration tool
