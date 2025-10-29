# Joystick Visualization Guide

## Overview

The BETAFPV Configurator now displays your RC channels as **2D joystick visualizations** for the gimbal sticks and numerical values for switches.

## Layout

```
┌────────────────────────────────────────────────────────┐
│  Gimbals (CH1-4)                                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌─ Right Gimbal ──┐      ┌─ Throttle Gimbal ──┐    │
│  │ (CH1: X, CH2: Y) │      │ (CH3: Y, CH4: X)    │    │
│  │                  │      │                     │    │
│  │       +Y         │      │        +Y           │    │
│  │        │         │      │         │           │    │
│  │   -X───●───+X    │      │    -X───●───+X      │    │
│  │        │         │      │         │           │    │
│  │       -Y         │      │        -Y           │    │
│  │                  │      │                     │    │
│  │  X: +0.00        │      │   X: +0.00          │    │
│  │  Y: +0.00        │      │   Y: +0.00          │    │
│  └──────────────────┘      └─────────────────────┘    │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Channels 5-8 (Switches)                               │
├────────────────────────────────────────────────────────┤
│  CH5:  2000                                            │
│  CH6:  1000                                            │
│  CH7:  2000                                            │
│  CH8:  2000                                            │
└────────────────────────────────────────────────────────┘
```

## Joystick Display Features

### Visual Elements
- **Dark background** - Easy to see in any lighting
- **Crosshairs** - Show center position (0, 0)
- **Green indicator** - Current stick position
- **Axis labels** - +X, -X, +Y, -Y markers
- **Numeric display** - Shows exact normalized values

### Coordinate System

The joysticks use a normalized coordinate system:

| Value | Position | RC Value |
|-------|----------|----------|
| **-1.00** | Minimum (left/down) | 1000 |
| **0.00** | Center | 1500 |
| **+1.00** | Maximum (right/up) | 2000 |

### Example Readings

#### Right Gimbal (CH1 & CH2)
- **X: +0.50, Y: +0.00** = Stick pushed halfway right, centered vertically
- **X: -1.00, Y: +1.00** = Stick in top-left corner
- **X: +0.00, Y: -0.50** = Stick centered horizontally, halfway down

#### Throttle Gimbal (CH3 & CH4)
- **X: +0.00, Y: +1.00** = Full throttle, centered yaw
- **X: +1.00, Y: +0.00** = Mid throttle, full right yaw
- **X: -0.50, Y: -1.00** = Zero throttle, half left yaw

## Channel Mapping

### Gimbal Channels (Visual Display)

| Channel | Gimbal | Axis | Function |
|---------|--------|------|----------|
| **CH1** | Right | X | Aileron (Roll) - Left/Right tilt |
| **CH2** | Right | Y | Elevator (Pitch) - Forward/Back tilt |
| **CH3** | Throttle | Y | Throttle - Up/Down power |
| **CH4** | Throttle | X | Rudder (Yaw) - Left/Right rotation |

### Switch Channels (Numerical Display)

| Channel | Typical Use | Values |
|---------|-------------|--------|
| **CH5** | Aux 1 / Arm | 1000 (OFF) / 2000 (ON) |
| **CH6** | Aux 2 / Mode | 1000 (OFF) / 2000 (ON) |
| **CH7** | Aux 3 | 1000 (OFF) / 2000 (ON) |
| **CH8** | Aux 4 | 1000 (OFF) / 2000 (ON) |

## Real-Time Updates

The joystick visualizations update in **real-time** at ~20 Hz (50ms refresh):

- **Move sticks** → See green indicator move instantly
- **Release sticks** → Watch them spring back to center
- **Numeric values update** → Shows exact position in -1.00 to +1.00 range

## Testing Your Controller

### 1. Center All Sticks
All joysticks should show:
- Green indicator at center
- X: +0.00, Y: +0.00

### 2. Right Gimbal Test
Move the right stick:
- **Up** → Y increases (+0.50 to +1.00)
- **Down** → Y decreases (-0.50 to -1.00)
- **Right** → X increases (+0.50 to +1.00)
- **Left** → X decreases (-0.50 to -1.00)

### 3. Throttle Gimbal Test
Move the left stick:
- **Up** → Y increases (throttle up)
- **Down** → Y decreases (throttle down)
- **Right** → X increases (yaw right)
- **Left** → X decreases (yaw left)

### 4. Switch Test
Flip switches on controller:
- **Switch ON** → Channel shows **2000**
- **Switch OFF** → Channel shows **1000**

## Calibration Check

Your controller is properly calibrated if:

✅ **Sticks centered** → X: ~0.00, Y: ~0.00 (within ±0.05)
✅ **Sticks at extremes** → ±1.00 (within ±0.05)
✅ **Sticks spring back** → Return to 0.00 when released
✅ **No drift** → Centered position stays stable

If values are significantly off (e.g., center = 0.20), you may need to calibrate your controller in the official BETAFPV configurator.

## Color Coding

- **Green indicator** - Normal operation, receiving data
- **Crosshairs (gray)** - Center reference lines
- **Dark background** - High contrast for visibility

## Tips

1. **Smooth movement** - The indicator should follow your stick movements smoothly without lag
2. **Spring return** - When you release sticks, they should snap back to center (0, 0)
3. **Corner test** - Move sticks to all 4 corners to verify full range
4. **Switch test** - All switches should toggle cleanly between 1000 and 2000

## Troubleshooting

### Indicator stuck at center
- Check "Bytes RX" counter is increasing
- Verify "Protocol: BETAFPV-Custom" is displayed
- Try moving sticks vigorously

### Values seem inverted
- This is normal! Some axes may be inverted depending on controller configuration
- The display shows the raw values from your controller

### Jittery indicator
- Normal for some controllers
- Should be within ±0.02 of target position
- Significant jitter (±0.10) may indicate electrical noise or low battery

## Data Format

Behind the scenes, your controller sends:
- **Raw values**: 0-2047 (11-bit resolution)
- **Converted to**: 1000-2000 (standard RC range)
- **Displayed as**: -1.00 to +1.00 (normalized)

This gives you smooth, precise control visualization! 🎮
