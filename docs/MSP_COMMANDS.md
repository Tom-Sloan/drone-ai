# MSP Command Reference

Complete reference for MSP (MultiWii Serial Protocol) commands used in Air65 drone control.

## Protocol Overview

MSP is a binary protocol for communication between flight controllers and external devices. Messages follow this structure:

```
Header: $M<    (3 bytes)
Size:          (1 byte)  - Payload length
Command:       (1 byte)  - Command ID
Payload:       (N bytes) - Command data
Checksum:      (1 byte)  - XOR checksum
```

### Direction Indicators

- `$M<` - Message TO flight controller (command)
- `$M>` - Message FROM flight controller (response)
- `$M!` - Error response

### Checksum Calculation

```python
checksum = size ^ command
for byte in payload:
    checksum ^= byte
```

## Commands Used in This System

### MSP_SET_RAW_RC (200)

**Send RC channel data to flight controller**

**Direction:** To FC (`$M<`)

**Payload:** 16 bytes (8 channels × 2 bytes each, little-endian uint16)

**Channel Order:**
1. Roll (AIL)
2. Pitch (ELE)
3. Yaw (RUD)
4. Throttle (THR)
5. AUX1 (Arming)
6. AUX2
7. AUX3
8. AUX4

**Value Range:** 1000-2000 (center = 1500, except throttle min = 1000)

**Example:**
```python
channels = [1500, 1500, 1500, 1000, 1000, 1500, 1500, 1500]
payload = struct.pack('<8H', *channels)
```

**Packet Example (hex):**
```
24 4D 3C 10 C8 DC 05 DC 05 DC 05 E8 03 E8 03 DC 05 DC 05 DC 05 XX
│  │  │  │  │  └─────┴─────┴─────┴─────┴─────┴─────┴───── 8 channels
│  │  │  │  └─ Command (200 = 0xC8)
│  │  │  └─ Size (16 = 0x10)
│  │  └─ Direction (<)
│  └─ M
└─ $ (header)
```

**Usage:**
- Must be sent continuously (50-100 Hz recommended)
- If not received for >300ms, triggers failsafe
- Used for all manual control

**Python Implementation:**
```python
def create_set_raw_rc(channels):
    payload = struct.pack('<8H', *channels)
    return create_msp_request(200, payload)
```

---

### MSP_STATUS (101)

**Get flight controller status**

**Direction:** From FC (`$M>`)

**Request:** No payload (size = 0)

**Response Payload:** 11 bytes
- `cycle_time` (2 bytes): Loop time in microseconds
- `i2c_error_count` (2 bytes): I2C errors
- `sensor` (2 bytes): Available sensors bitmask
- `flag` (4 bytes): Status flags
- `current_setting` (1 byte): Current PID profile

**Status Flags:**
- Bit 0: ARMED
- Bit 1: WAS_EVER_ARMED
- Bit 2-5: Flight mode flags

**Example Request:**
```python
request = create_msp_request(101, b'')  # Empty payload
```

**Example Response Parsing:**
```python
cycle_time, i2c_err, sensor, flag = struct.unpack('<HHHI', payload[:11])
armed = bool(flag & 0x01)
```

**Usage:**
- Poll periodically (5-10 Hz)
- Check armed state before sending control
- Monitor for errors

---

### MSP_ANALOG (110)

**Get battery and power information**

**Direction:** From FC (`$M>`)

**Request:** No payload

**Response Payload:** 7+ bytes
- `vbat` (1 byte): Battery voltage × 10 (e.g., 42 = 4.2V)
- `mah_drawn` (2 bytes): mAh consumed
- `rssi` (2 bytes): Signal strength (0-1023)
- `amperage` (2 bytes): Current × 100 (e.g., 150 = 1.5A)

**Example Parsing:**
```python
vbat, mah, rssi, amps = struct.unpack('<BHHH', payload[:7])
voltage = vbat / 10.0
current = amps / 100.0
```

**Usage:**
- Poll every 200ms for battery monitoring
- Alert if voltage drops below safe level (3.5V for 1S)
- Track power consumption

---

### MSP_ATTITUDE (108)

**Get drone orientation**

**Direction:** From FC (`$M>`)

**Request:** No payload

**Response Payload:** 6 bytes
- `roll` (2 bytes): Roll angle × 10 (degrees)
- `pitch` (2 bytes): Pitch angle × 10 (degrees)
- `yaw` (2 bytes): Yaw heading (degrees)

**Example Parsing:**
```python
roll, pitch, yaw = struct.unpack('<hhh', payload[:6])
roll_deg = roll / 10.0
pitch_deg = pitch / 10.0
```

**Angle Convention:**
- Roll: Positive = right wing down
- Pitch: Positive = nose up
- Yaw: 0-360 degrees (magnetic heading)

**Usage:**
- Poll at 10-20 Hz for attitude display
- Use for stabilization algorithms
- Monitor for unusual angles

---

### MSP_RC (105)

**Get current RC channel values**

**Direction:** From FC (`$M>`)

**Request:** No payload

**Response Payload:** 16 bytes (8 channels)
- 8 × uint16 channel values (1000-2000)

**Usage:**
- Read what FC is receiving
- Verify MSP_SET_RAW_RC working
- Compare sent vs received values

---

### MSP_RAW_IMU (102)

**Get raw IMU sensor data**

**Direction:** From FC (`$M>`)

**Response Payload:** 18 bytes
- `acc_x`, `acc_y`, `acc_z` (6 bytes): Accelerometer
- `gyro_x`, `gyro_y`, `gyro_z` (6 bytes): Gyroscope
- `mag_x`, `mag_y`, `mag_z` (6 bytes): Magnetometer

**Usage:**
- Advanced control algorithms
- Custom stabilization
- Sensor debugging

---

### MSP_MOTOR (104)

**Get motor output values**

**Direction:** From FC (`$M>`)

**Response:** Motor PWM values (1000-2000)

**Usage:**
- Verify motors responding
- Check motor mixing
- Debug control issues

---

## Command Sequences

### Arming Sequence

1. Ensure throttle at minimum (1000)
2. Send MSP_SET_RAW_RC with AUX1 = 2000
3. Continue sending at 50 Hz
4. Poll MSP_STATUS to confirm armed

```python
channels = [1500, 1500, 1500, 1000, 2000, 1500, 1500, 1500]
#           roll  pitch yaw   throt arm
```

### Disarming Sequence

1. Reduce throttle to minimum (1000)
2. Send MSP_SET_RAW_RC with AUX1 = 1000
3. Poll MSP_STATUS to confirm disarmed

### Normal Flight Control Loop

```python
while flying:
    # 1. Send control (50 Hz)
    send_msp_set_raw_rc(channels)

    # 2. Request telemetry (10 Hz)
    if time_for_telemetry():
        request_msp_status()
        request_msp_analog()
        request_msp_attitude()

    # 3. Update display (30 Hz)
    if time_for_display():
        update_gui()

    sleep(0.02)  # 50 Hz loop
```

## Timing Requirements

### Critical

- **MSP_SET_RAW_RC:** 50-100 Hz (every 10-20ms)
  - Below 3 Hz: Failsafe triggered
  - Below 10 Hz: Jittery control
  - 50 Hz: Smooth, recommended

### Telemetry

- **MSP_STATUS:** 5-10 Hz (every 100-200ms)
- **MSP_ANALOG:** 5 Hz (every 200ms)
- **MSP_ATTITUDE:** 10-20 Hz (every 50-100ms)

### Display Updates

- GUI refresh: 30-60 Hz
- Telemetry labels: 10 Hz adequate
- Camera feed: 30 fps

## Error Handling

### Checksum Errors

If received checksum doesn't match:
- Discard message
- Don't update telemetry
- Continue sending control

### Timeout

If no response to telemetry request:
- Continue control loop (critical)
- Mark telemetry as stale
- Retry next cycle

### Failsafe

If FC doesn't receive MSP_SET_RAW_RC:
- After 300ms: Triggers failsafe
- FC behavior: Configured in Betaflight (DROP/LAND)
- Recovery: Resume sending commands

## Betaflight Configuration

### Enable MSP RX

```
# In Betaflight CLI:
set serialrx_provider = MSP
feature RX_MSP
save
```

### Configure Arming

In Betaflight Configurator:
1. Modes tab → Add ARM mode
2. Assign to AUX1
3. Range: 1800-2100

### Failsafe Settings

```
# Recommended failsafe:
set failsafe_delay = 3        # 0.3 seconds
set failsafe_procedure = DROP # or LAND
```

## Python Implementation Examples

### Basic Request

```python
def create_msp_request(command, payload=b''):
    size = len(payload)
    checksum = size ^ command
    for byte in payload:
        checksum ^= byte

    msg = bytearray(b'$M<')
    msg.append(size)
    msg.append(command)
    msg.extend(payload)
    msg.append(checksum)
    return bytes(msg)
```

### Send Control

```python
def send_rc_channels(serial, channels):
    payload = struct.pack('<8H', *channels)
    request = create_msp_request(200, payload)
    serial.write(request)
```

### Request Telemetry

```python
def request_status(serial):
    request = create_msp_request(101, b'')
    serial.write(request)

def request_battery(serial):
    request = create_msp_request(110, b'')
    serial.write(request)
```

## Troubleshooting

### Commands Not Working

1. Verify RX_MSP feature enabled
2. Check correct serial port configured
3. Confirm baud rate 115200
4. Test with Betaflight Configurator CLI

### Checksum Errors

1. Verify byte order (little-endian)
2. Check payload size matches
3. Confirm XOR checksum calculation

### No Response

1. Ensure request format correct ($M<)
2. Check command ID valid
3. Verify FC not in configurator mode
4. Monitor with serial debugger

## References

- **Betaflight MSP Wiki:** https://github.com/betaflight/betaflight/wiki/MSP-V2
- **MSP Protocol Spec:** https://github.com/betaflight/betaflight/blob/master/src/main/msp/msp_protocol.h
- **Command Codes:** https://github.com/betaflight/betaflight/blob/master/src/main/msp/msp.h

---

**Note:** This reference covers MSP V1 protocol. MSP V2 uses different header ($X<) and extended features but is backward compatible.
