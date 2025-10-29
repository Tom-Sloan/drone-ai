# Air65 Drone Control Guide

Complete guide for controlling the Air65 drone via MSP protocol using USB connection.

## Overview

This system provides manual control of your Air65 drone via the MSP (MultiWii Serial Protocol) through USB connection. It includes:

- **Live FPV camera feed** from USB camera
- **Real-time telemetry** (battery, attitude, RSSI, armed state)
- **Manual control** via sliders (throttle, roll, pitch, yaw)
- **50 Hz control loop** for smooth, continuous command transmission
- **Safety features** (emergency stop, arm/disarm)

## Hardware Setup

### Requirements

1. **Air65 Drone** (BETAFPV Air65 or compatible)
2. **USB Cable** - Air65 uses JST connector, needs USB-C adapter
3. **USB Camera** (optional) - For FPV feed display

### Betaflight Configuration

Before using this control system, configure your Air65 in Betaflight Configurator:

1. **Enable MSP RX Mode:**
   ```
   CLI Command: set serialrx_provider = MSP
   CLI Command: feature RX_MSP
   CLI Command: save
   ```

2. **Configure UART for MSP:**
   - Open Betaflight Configurator
   - Go to Ports tab
   - Enable MSP on UART1 (usually the USB port)
   - Baud rate: 115200

3. **Set Up Arming:**
   - Go to Modes tab
   - Configure ARM mode on AUX1 channel
   - Set ARM range: 1800-2100

4. **IMPORTANT SAFETY:**
   - Configure failsafe behavior
   - Set failsafe to DROP or LAND
   - Test failsafe before flight

## Running the Application

### Method 1: Using Launch Script (Recommended)

```bash
./run_drone.sh
```

This automatically activates the conda environment and launches the GUI.

### Method 2: Direct Python

```bash
conda activate betafpv-ultrathink
python3 drone_control.py
```

## Using the GUI

### 1. Connection

1. **Find Serial Port:**
   - Mac: `/dev/cu.usbmodem*` (check with `ls /dev/cu.usb*`)
   - Windows: `COM3`, `COM4`, etc.

2. **Connect:**
   - Enter port name in connection field
   - Click "Connect"
   - Green "Connected" indicator should appear

### 2. FPV Camera

- Camera feed appears automatically
- Select camera index if multiple cameras available (0, 1, 2...)
- Start/Stop camera as needed

### 3. Telemetry

Monitor real-time data:
- **Battery:** Voltage, current draw, mAh consumed
- **RSSI:** Signal strength (0-1023)
- **Attitude:** Roll, pitch, yaw angles
- **Status:** Armed/Disarmed state

### 4. Manual Control

**IMPORTANT: Remove props for initial testing!**

1. **Enable Controls:**
   - Click "Enable Controls" button
   - Control loop starts at 50 Hz
   - Status shows "Controls enabled"

2. **Control Sliders:**
   - **Throttle:** 1000 (min) to 2000 (max)
   - **Roll:** 1000 (left) to 2000 (right), center = 1500
   - **Pitch:** 1000 (forward) to 2000 (back), center = 1500
   - **Yaw:** 1000 (left) to 2000 (right), center = 1500
   - Use "Center" buttons to reset roll/pitch/yaw to 1500

3. **Arming Sequence:**
   - Set throttle to minimum (1000)
   - Click "ARM" button
   - Wait for telemetry to show "ARMED" status
   - Check that motors spin with small throttle increase

4. **Flying:**
   - Slowly increase throttle
   - Adjust roll/pitch/yaw as needed
   - All commands sent continuously at 50 Hz

5. **Landing:**
   - Reduce throttle gradually
   - After landing, click "DISARM"

6. **Emergency Stop:**
   - Click "EMERGENCY STOP" any time
   - Immediately resets all channels to safe values
   - Sets throttle to minimum, disarms drone

## Control Loop Details

### How It Works

The MSP Control Loop runs in a background thread at 50 Hz:

1. Reads current slider values
2. Encodes as MSP_SET_RAW_RC command
3. Sends via serial at 20ms intervals
4. Continues until "Disable Controls" clicked

### Why 50 Hz?

- Betaflight expects continuous RC input
- No input for >300ms triggers failsafe
- 50 Hz (20ms period) provides smooth control
- Matches typical RC receiver update rate

### Channel Mapping

```
Channel 0 (ROLL):     1000-2000 (left to right)
Channel 1 (PITCH):    1000-2000 (forward to back)
Channel 2 (YAW):      1000-2000 (left to right)
Channel 3 (THROTTLE): 1000-2000 (min to max)
Channel 4 (AUX1):     1000=disarm, 2000=arm
Channel 5 (AUX2):     1500 (unused)
Channel 6 (AUX3):     1500 (unused)
Channel 7 (AUX4):     1500 (unused)
```

## Safety Guidelines

### Pre-Flight Checks

1. **Props Off Testing:**
   - Always test with propellers removed first
   - Verify arming/disarming works
   - Test all control axes
   - Confirm emergency stop works

2. **Connection Test:**
   - Verify telemetry updates
   - Check battery voltage reading
   - Confirm attitude indicators respond

3. **Control Response:**
   - Test throttle response
   - Verify roll/pitch/yaw directions
   - Ensure sliders control correct axes

### During Flight

1. **Stay Ready:**
   - Keep hand near emergency stop
   - Monitor battery voltage
   - Watch for any unusual behavior

2. **Controlled Movements:**
   - Make small, gradual slider adjustments
   - Don't slam throttle to max
   - Center roll/pitch when uncertain

3. **Emergency Procedures:**
   - If control loss: Click EMERGENCY STOP
   - If battery low (<3.5V): Land immediately
   - If unusual behavior: DISARM and land

### Limitations

1. **USB Connection:**
   - USB timing can be inconsistent
   - Not recommended for complex maneuvers
   - Best for tethered testing and development

2. **Control Latency:**
   - Slider to drone: ~20-40ms typical
   - Not suitable for racing or acrobatics
   - Fine for hover, slow flight, testing

3. **No Transmitter:**
   - Can't switch back to manual RC
   - Must reboot FC to use normal transmitter
   - Emergency stop is only backup

## Troubleshooting

### Connection Issues

**Problem:** Can't connect to drone
- Check USB cable connection
- Verify correct serial port name
- Confirm Betaflight not running (closes port)
- Try different USB port

**Problem:** Connected but no telemetry
- Verify MSP enabled on correct UART
- Check baud rate is 115200
- Reconnect USB cable
- Restart drone

### Control Issues

**Problem:** Drone won't arm
- Check throttle is at minimum (1000)
- Verify AUX1 configured for ARM in Betaflight
- Check for Betaflight error messages
- Ensure battery voltage sufficient (>3.7V for 1S)

**Problem:** Controls don't respond
- Verify "Enable Controls" clicked
- Check control loop running (status bar)
- Confirm RX_MSP feature enabled in Betaflight
- Restart control loop (disable/enable)

**Problem:** Jittery control
- Check USB cable quality
- Reduce USB hub usage
- Monitor control loop frequency (should be ~50 Hz)
- Close other serial applications

### Camera Issues

**Problem:** No camera feed
- Verify camera plugged in
- Check camera index (try 0, 1, 2)
- Test camera in other app
- Grant camera permissions if prompted

## Technical Details

### MSP Commands Used

- **MSP_SET_RAW_RC (200):** Send control inputs
- **MSP_STATUS (101):** Get armed state, flags
- **MSP_ANALOG (110):** Get battery voltage, current
- **MSP_ATTITUDE (108):** Get roll, pitch, yaw

### File Structure

```
src/
├── msp_parser.py           # MSP protocol encoding/parsing
├── msp_control_loop.py     # 50 Hz control loop
├── drone_control_gui.py    # Main GUI application
├── serial_comm.py          # Serial port communication
└── camera_widget.py        # Camera feed display
```

### Control Flow

```
User moves slider
    ↓
GUI updates channel value
    ↓
Control loop reads channel values (50 Hz)
    ↓
Encode as MSP_SET_RAW_RC packet
    ↓
Send via serial to drone
    ↓
Betaflight processes as RC input
    ↓
Motors respond
```

## Development and Extension

### Adding New Features

The system is designed to be extensible:

1. **Custom Controllers:**
   - Implement autopilot modes
   - Add PID controllers
   - Create waypoint navigation

2. **Data Logging:**
   - Log telemetry for analysis
   - Record control inputs
   - Export flight data

3. **Advanced Control:**
   - Integrate game controller
   - Add auto-hover mode
   - Implement altitude hold

### Integration Points

- `msp_control_loop.py` - Modify `set_channels()` for custom control
- `drone_control_gui.py` - Add new UI elements, modes
- `msp_parser.py` - Add more MSP commands

## Resources

- **Betaflight MSP Protocol:** https://github.com/betaflight/betaflight/wiki/MSP-V2
- **Air65 Manual:** BETAFPV website
- **MSP Command Reference:** `/docs/MSP_COMMANDS.md`

## Support

For issues specific to this control system:
- Check troubleshooting section above
- Review console output for error messages
- Verify Betaflight configuration

For Betaflight/drone issues:
- Consult Betaflight documentation
- Check BETAFPV support resources
- Use Betaflight Configurator for diagnostics

---

**Remember: Safety first! Always test with props off initially.**
