# Changelog - BETAFPV Configurator UltraThink Edition

## Version 2.0 - Enhanced Port Detection & Connection Status

### New Features

#### 1. Automatic Port Detection
- **Auto-scan on startup**: The application now automatically scans for all available serial ports when launched
- **Dropdown menu**: Replaced text entry with a user-friendly dropdown that lists all detected ports
- **Port descriptions**: Each port displays helpful information:
  - Windows: "COM2 - USB Serial Device"
  - Mac: "/dev/cu.usbserial-1234 - CP2102 USB to UART Bridge Controller"
  - Linux: "/dev/ttyUSB0 - USB-Serial Controller"

#### 2. One-Click Refresh
- **🔄 Refresh Button**: Instantly rescan for serial ports without restarting the application
- Perfect for when you plug in your device after the app has already started
- Shows the number of detected ports in the log

#### 3. Baud Rate Dropdown
- Replaced text entry with dropdown menu
- Pre-populated with common baud rates: 9600, 19200, 38400, 57600, 115200, 230400
- Default: 115200 (standard for BETAFPV devices)

#### 4. Dual Connection Status Display
- **PC ↔ Controller Status**: Shows if your computer is connected to the BETAFPV 2SE transmitter
  - "Connected & Active" (green) when receiving data
  - "Not Connected" (gray) when disconnected

- **Controller ↔ Drone Link Status**: Shows if the controller has an active radio link to a drone
  - "✓ LINKED" (green) when drone is connected
  - "✗ No Link" (red) when no drone detected

#### 5. Real-Time Link Quality
- Displays signal strength as a percentage (0-100%)
- Color-coded indicators:
  - **Green** (70-100%): Excellent signal
  - **Orange** (40-69%): Fair signal
  - **Red** (1-39%): Poor signal, connection may be unstable
  - **Gray** (---): No signal detected

#### 6. Intelligent Link Detection
- Automatically determines if a drone is linked based on:
  - Active RSSI (Received Signal Strength Indicator)
  - Recent telemetry data (within last 2 seconds)
  - RC channel activity
  - Attitude/IMU data streaming

### Technical Improvements

- Added `serial.tools.list_ports` for cross-platform port detection
- Implemented real-time link monitoring with 2-second timeout detection
- Enhanced error handling for port selection
- Improved UI layout with better spacing and organization
- Added visual separator between connection settings and status indicators

### User Interface Changes

**Before:**
```
Port: [COM2____] Baud: [115200__] [Connect] Status Protocol
```

**After:**
```
Port: [COM2 - USB Serial Device ▼] 🔄  Baud: [115200 ▼] [Connect] Connected Protocol: MSP
─────────────────────────────────────────────────────────────────────────────────
Controller Status: Connected & Active    Drone Link: ✓ LINKED    Link Quality: 87% (green)
```

### Platform Support

- **Windows**: Full support with COM port detection
- **macOS**: Full support with /dev/tty.* and /dev/cu.* detection
- **Linux**: Full support with /dev/ttyUSB* and /dev/ttyACM* detection

### Bug Fixes

- Fixed issue where manually typed port names could include extra spaces
- Improved handling of disconnection states
- Better validation before attempting connection

### Documentation Updates

- Updated README.md with new port detection features
- Added troubleshooting section for port detection issues
- Included platform-specific port naming conventions
- Created CHANGELOG.md to track version history

---

## Version 1.0 - Initial Release

### Features
- Basic serial communication with BETAFPV devices
- MSP and MAVLink protocol support
- Real-time telemetry display
- Battery monitoring
- Attitude display (Roll, Pitch, Yaw)
- RC channel monitoring
- Data logging
- Manual port and baud rate entry
