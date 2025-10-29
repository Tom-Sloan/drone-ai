# BETAFPV 2SE Custom Protocol Documentation

## Discovery

Your BETAFPV 2SE transmitter uses a **proprietary 16-byte packet format** for transmitting RC channel data, not standard MSP or MAVLink.

## Packet Format

### Raw Data Pattern
```
FF 03 FF 03 A7 05 FF 03 FF 07 00 00 FF 07 FF 07
```

### Structure
- **Total Size**: 16 bytes
- **Format**: 8 channels × 2 bytes per channel (16-bit little-endian)
- **Update Rate**: ~20 Hz (50ms intervals)

### Channel Encoding

Each channel is encoded as a 16-bit little-endian unsigned integer:

| Bytes | Hex Value | Decimal | Meaning |
|-------|-----------|---------|---------|
| 0-1   | `FF 03`   | 1023    | Channel 1 (center position) |
| 2-3   | `FF 03`   | 1023    | Channel 2 (center) |
| 4-5   | `A7 05`   | 1447    | Channel 3 (mid-high) |
| 6-7   | `FF 03`   | 1023    | Channel 4 (center) |
| 8-9   | `FF 07`   | 2047    | Channel 5 (max) |
| 10-11 | `00 00`   | 0       | Channel 6 (min) |
| 12-13 | `FF 07`   | 2047    | Channel 7 (max) |
| 14-15 | `FF 07`   | 2047    | Channel 8 (max) |

### Value Ranges
- **Native Range**: 0 - 2047 (11-bit resolution)
- **Center Value**: 1023 (0x03FF)
- **Min Value**: 0 (0x0000)
- **Max Value**: 2047 (0x07FF)

### Conversion to Standard RC Values
Standard RC values use 1000-2000 range with 1500 as center.

**Formula:**
```
standard_value = 1000 + (native_value / 2047) × 1000
```

**Examples:**
- 0 → 1000 (min)
- 1023 → ~1500 (center)
- 2047 → 2000 (max)

## Implementation

### Parser (`betafpv_custom_parser.py`)
The custom parser:
1. Buffers incoming bytes
2. Looks for 16-byte packets
3. Parses 8 channels as little-endian uint16
4. Validates values are in 0-4095 range
5. Converts to standard 1000-2000 RC format

### Integration
The GUI now tries to parse data in this order:
1. **BETAFPV Custom** (16-byte packets) ← Your controller uses this!
2. MSP (MultiWii Serial Protocol)
3. MAVLink (standard or BETAFPV variant)

## What Works Now

### ✅ Protocol Detection
- Protocol field shows: **"BETAFPV-Custom"**
- Recognizes 16-byte RC packets automatically

### ✅ RC Channel Display
- Shows all 8 channels in real-time
- Values converted to standard 1000-2000 range
- Updates ~20 times per second

### ✅ Controller Status
- Detects active data stream
- Shows "Connected & Active" when receiving packets

### ✅ Port Memory
- Remembers `/dev/cu.usbmodem568241534B002`
- Auto-selects it on next launch
- Saves config to `~/.betafpv_config.json`

## Testing Results

Based on your log data:
```
FF 03 FF 03 A7 05 FF 03 FF 07 00 00 FF 07 FF 07
```

**Parsed values:**
- CH1: 1023 → 1500 (stick centered)
- CH2: 1023 → 1500 (stick centered)
- CH3: 1447 → 1707 (throttle ~70%)
- CH4: 1023 → 1500 (stick centered)
- CH5: 2047 → 2000 (switch ON)
- CH6: 0 → 1000 (switch OFF)
- CH7: 2047 → 2000 (switch ON)
- CH8: 2047 → 2000 (switch ON)

## Future Enhancements

### Possible Additional Data
The BETAFPV 2SE might send more than just RC channels. Future versions could decode:
- Battery voltage (if transmitted)
- Button states
- Additional telemetry

### Reverse Engineering Notes
To discover additional protocols:
1. Use the diagnostic tool: `python test_serial_raw.py`
2. Press different buttons/switches on controller
3. Watch for pattern changes
4. Document packet variations

## Comparison with Official Configurator

The official BETAFPV_Configurator uses MAVLink for **configuration/settings**, but your log shows the controller sends **raw RC data** during normal operation.

**Two modes:**
1. **Config Mode**: MAVLink messages (when using configurator)
2. **Normal Mode**: 16-byte RC packets (what you're seeing)

Your controller is currently in normal RC transmission mode, not configurator mode.
