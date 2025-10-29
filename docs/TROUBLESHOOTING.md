# Troubleshooting Guide - BETAFPV Configurator

## Issue: Protocol stays on "Auto-detecting..." and values don't update

### ✅ **FIXED** - MAVLink Message ID Mismatch

**Root Cause:** The MAVLink message IDs were incorrectly mapped, preventing the parser from recognizing messages from the BETAFPV 2SE controller.

**What was wrong:**
- RC_CHANNELS was 6, should be 7
- PID was 7, should be 14
- MOTORS_TEST was 8, should be 11
- ALTITUDE was 9, should be 12
- RATE was 10, should be 13

**Status:** ✅ **FIXED in latest version**

The correct message IDs from the official BETAFPV_Configurator are now implemented.

---

## How to Test the Fix

### Step 1: Restart the Application

```bash
cd /Users/Sloan/Desktop/Project_Desktop/Self/drone_simulator
python betafpv_gui.py
```

### Step 2: Connect to Your Device

1. Select `/dev/cu.usbmodem568241534B002` from the port dropdown
2. Baud rate: 115200
3. Click **Connect**

### Step 3: Enable Debug Logging

1. Check the **"Show Raw Data (Hex)"** checkbox at the bottom of the Data Log
2. Watch the **"Bytes RX"** counter - it should be increasing if data is being received
3. Look for messages in the log like:
   - `HEARTBEAT: {...}`
   - `RC_CHANNELS: {...}`
   - `ATTITUDE: {...}`
   - `SYS_STATUS: {...}`

### Step 4: Check Protocol Detection

- The **Protocol** field should change from "Auto-detecting..." to **"MAVLink"**
- The **Controller Status** should show "Connected & Active" (green)

---

## Debugging Tools

### 1. **Built-in Debugging** (in GUI)

The GUI now includes debugging features:

- ✅ **"Show Raw Data (Hex)" checkbox** - Shows all incoming data in hexadecimal
- ✅ **Bytes RX counter** - Displays total bytes received
- ✅ **Warning messages** - Shows "⚠ Received X bytes but no valid MSP/MAVLink" if data isn't recognized
- ✅ **Protocol-specific logging** - Each recognized message is logged with its type

### 2. **Raw Serial Diagnostic Tool**

Use the diagnostic script to see raw data directly:

```bash
python test_serial_raw.py --port /dev/cu.usbmodem568241534B002 --baud 115200
```

This will:
- Display all incoming bytes in hex format
- Show ASCII representation
- Detect MAVLink headers (0xFE)
- Show data rate (bytes/second)
- Help identify if data is being received at all

**Example output:**
```
==================================================================
BETAFPV Serial Diagnostic Tool
==================================================================
Port: /dev/cu.usbmodem568241534B002
Baud Rate: 115200

✓ Successfully opened /dev/cu.usbmodem568241534B002
  Port is open: True

Listening for data... (Press Ctrl+C to stop)
------------------------------------------------------------------
[  32 bytes] FE 14 01 00 00 07 18 05 E0 05 DC 05 DC 05 ...
             ASCII: .................

    ⚠ DETECTED MAVLink HEADER (0xFE)

>>> Receiving ~256.3 bytes/sec (Total: 256)
```

---

## Expected Behavior After Fix

### ✅ When Working Correctly:

1. **Protocol field**: Shows "MAVLink" within 1-2 seconds
2. **Controller Status**: "Connected & Active" (green)
3. **Data Log shows**:
   - `HEARTBEAT: {'flight_mode': X, 'armed': False, ...}`
   - `RC_CHANNELS: {'channels': [1500, 1500, ...], 'rssi': 85}`
   - `ATTITUDE: {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}`
   - `SYS_STATUS: {'voltage': 3.8, 'current': 0.0, ...}`

4. **Telemetry updates**:
   - Voltage displays battery level
   - RSSI shows signal strength
   - Roll/Pitch/Yaw change when you move controller sticks
   - RC channels show stick positions (typically 1000-2000 range)

5. **Bytes RX counter**: Continuously increasing

### ❌ If Still Not Working:

#### Symptom: Bytes RX = 0 (no data received)
**Possible causes:**
- Wrong port selected
- Device not powered on
- USB cable issue
- Port locked by another application

**Solution:**
1. Try the diagnostic tool: `python test_serial_raw.py`
2. Check port in Terminal: `ls /dev/cu.* /dev/tty.*`
3. Close other apps that might be using the port (FPV.skydive, BETAFPV_Configurator)
4. Unplug and replug the device
5. Click the 🔄 refresh button to rescan ports

#### Symptom: Bytes RX increasing, but Protocol = "Auto-detecting..."
**Possible causes:**
- Data is being received but format doesn't match MSP or MAVLink
- Checksum validation failing (currently disabled for debugging)

**Solution:**
1. Enable "Show Raw Data (Hex)"
2. Look for hex pattern starting with `FE` (MAVLink header)
3. Share first 50 bytes in hex for analysis
4. Try the diagnostic tool to verify data format

#### Symptom: ⚠ Warning messages about unrecognized data
**Means:**
- Data is being received
- MAVLink/MSP headers not found in stream
- Possible data corruption or different protocol

**Solution:**
1. Check that the device is in configurator mode (not in bootloader/DFU mode)
2. Try power cycling the device
3. Verify baud rate is 115200
4. Use diagnostic tool to see raw hex data

---

## MAVLink Message IDs (Correct Values)

For reference, these are the correct BETAFPV MAVLink message IDs:

| Message Type | ID | Description |
|---|---|---|
| HEARTBEAT | 1 | Device status heartbeat |
| FIRMWARE_INFO | 2 | Firmware version info |
| SYS_STATUS | 3 | System status, battery |
| IMU | 4 | Accelerometer, gyro, magnetometer |
| ATTITUDE | 5 | Roll, pitch, yaw angles |
| LOCAL_POSITION | 6 | Local position data |
| **RC_CHANNELS** | **7** | RC inputs and RSSI |
| COMMAND | 8 | Command message |
| COMMAND_ACK | 9 | Command acknowledgment |
| STATUS_TEXT | 10 | Status text messages |
| MOTORS_TEST | 11 | Motor test data |
| **ALTITUDE** | **12** | Altitude data |
| **RATE** | **13** | Rate settings |
| **PID** | **14** | PID parameters |
| MOTORS_MINIVALUE | 15 | Motor minimum values |
| UNIQUE_DEVICE_ID | 16 | Device unique ID |

---

## Need More Help?

1. **Run the diagnostic tool** to capture raw data:
   ```bash
   python test_serial_raw.py > serial_output.txt
   ```
   (Let it run for 5 seconds, then Ctrl+C)

2. **Enable raw data logging** in GUI and save the log

3. **Check official configurator** works with same device on same port

4. **Verify USB drivers** are installed (CP2102 or FTDI drivers on macOS)
