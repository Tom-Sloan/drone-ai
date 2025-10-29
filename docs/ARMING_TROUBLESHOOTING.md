# Arming Troubleshooting Guide

## Why Won't My Drone Arm?

If your drone isn't arming when you click the ARM button, here's how to diagnose and fix the issue.

---

## Quick Checklist

Before troubleshooting, verify:
- [ ] **Props removed** (for safety)
- [ ] **Battery connected** and voltage > 3.5V
- [ ] **USB connected** and showing "Connected" in GUI
- [ ] **Controls enabled** (button shows "Disable Controls")
- [ ] **Throttle at minimum** (slider all the way left = 1000)

---

## Step-by-Step Troubleshooting

### 1. Check Betaflight Configuration

**Open Betaflight Configurator and verify:**

#### A. RX_MSP Feature Enabled
1. Go to **Configuration** tab
2. Under **Receiver**, check:
   - Receiver Mode: **MSP RX**
3. Under **Other Features**, enable:
   - **RX_MSP**
4. Click **Save and Reboot**

#### B. ARM Mode Configuration
1. Go to **Modes** tab
2. Find **ARM** mode
3. Verify it's assigned to **AUX 1** channel
4. Set range: **1700 - 2100**
   ```
   ARM: |------[====ARM====]|
        1000  1700         2100  2000
        AUX 1
   ```
5. Click **Save**

#### C. Check Arming Flags
1. Go to **CLI** tab
2. Type: `status`
3. Look for **"Arming disabled flags"**

Common flags that prevent arming:
```
NOGYRO          - Gyro not detected/calibrated
FAILSAFE        - In failsafe mode
RXLOSS          - No RC signal (should be OK with MSP)
CLI             - In CLI mode (exit with 'exit')
THROTTLE        - Throttle not at minimum
ANGLE           - Drone tilted too much
LOAD            - CPU load too high
CALIBRATING     - Sensors calibrating
RPMFILTER       - RPM filter not ready
```

**Fix common flags:**
- `NOGYRO` or `CALIBRATING`: Wait 10 seconds after power-on
- `THROTTLE`: Make sure throttle slider is at 1000
- `ANGLE`: Place drone level on flat surface
- `RXLOSS`: Ignore if using MSP (expected)

---

### 2. Test Receiver Inputs in Betaflight

1. Go to **Receiver** tab in Betaflight Configurator
2. **Disconnect** our GUI (to avoid conflict)
3. **Connect** in Betaflight Configurator
4. Watch the channel bars

**Test manually:**
- Open your GUI
- Click "Enable Controls"
- Move sliders and click ARM
- Watch Betaflight Receiver tab

**What you should see:**
- **CH1 (Roll)**: Moves when you move Roll slider
- **CH2 (Pitch)**: Moves when you move Pitch slider
- **CH3 (Throttle)**: Should be at **~1000** (left side)
- **CH4 (Yaw)**: Moves when you move Yaw slider
- **AUX1 (CH5)**: Jumps to **~2000** when you click ARM

**If channels don't move:**
- Control loop may not be running
- Check "Enable Controls" is clicked
- Verify MSP_SET_RAW_RC commands are being sent (check console)

---

### 3. Verify Control Loop is Sending Commands

**Check console output:**

When you click "Enable Controls", you should see:
```
MSP Control Loop started at 50 Hz
Controls enabled - Control loop running at 50 Hz
```

When you click "ARM", you should see:
```
Arming command sent (AUX1 = 2000)
ARM command sent - Channels: Throttle=1000, AUX1=2000
```

**If you don't see these messages:**
- Control loop may not have started
- Try clicking "Disable Controls" then "Enable Controls" again

---

### 4. Check Arming Sequence Timing

Some flight controllers require the arm signal to be held for a moment.

**Proper sequence:**
1. **Set throttle to minimum** (1000) FIRST
2. **Enable controls**
3. Wait 2 seconds
4. **Click ARM**
5. Keep controls enabled for 3-5 seconds
6. Check telemetry - "Status: ARMED" should appear in red

**If still not arming:**
- The AUX1 range in Betaflight might be wrong
- Try adjusting to 1800-2100 instead of 1700-2100

---

### 5. Alternative: Manual Arming with Sliders

If the ARM button doesn't work, you can arm manually:

1. **Enable controls**
2. **Set throttle to 1000** (minimum)
3. **Move Yaw slider far right** (to 2000)
4. **Hold for 1-2 seconds**
5. **Return Yaw to center** (1500)
6. Drone should arm

This is the traditional "throttle low, yaw right" arming method.

---

### 6. Check for Betaflight Version Issues

Different Betaflight versions handle MSP differently:

**Betaflight 4.3+:**
- Uses MSPv2 by default
- Our code uses MSPv1 (should still work)
- May need explicit MSP_SET_RAW_RC support

**Check in CLI:**
```
get msp_override
```

If it shows a value, try:
```
set msp_override_channels_mask = 0
save
```

Then try arming again.

---

### 7. Enable Debug Logging

Add debug output to see exactly what's being sent:

**In console, you should see:**
```
[15:08:25] Controls enabled
Commands sent: 150 (shows it's sending at 50 Hz)
Throttle=1000, AUX1=2000 (when ARM clicked)
```

**If commands aren't being sent:**
- Serial connection may have dropped
- Click "Disconnect" then "Connect" again
- Check USB cable

---

## Advanced Troubleshooting

### Check MSP Packet Structure

The ARM command sends:
```
MSP_SET_RAW_RC (200):
  CH1 (Roll):    1500
  CH2 (Pitch):   1500
  CH3 (Yaw):     1500
  CH4 (Throttle): 1000  ← Must be low
  AUX1:          2000  ← Arm signal
  AUX2-4:        1500
```

### Verify in Betaflight CLI

While our GUI is connected and ARM is clicked:

1. **Disconnect GUI**
2. **Connect Betaflight Configurator**
3. **CLI tab**
4. Type: `get rc`

You should see:
```
rxmsp_channel_count = 8
```

Type: `get aux`
Look for AUX1 configuration

---

## Still Not Working?

### Try These Steps:

1. **Full Betaflight reset:**
   ```
   defaults nosave
   save
   ```
   Then reconfigure everything

2. **Try a different AUX channel:**
   - Configure ARM on AUX2 instead
   - Modify `msp_control_loop.py`:
     ```python
     self.channels[5] = 2000  # Use AUX2 instead of AUX1
     ```

3. **Check if manual RC transmitter works:**
   - If your normal RC transmitter can arm, but MSP can't
   - The issue is definitely in MSP configuration

4. **Use Betaflight Blackbox:**
   - Enable blackbox logging
   - Try to arm with MSP
   - Check logs to see what's preventing arming

---

## Success! Drone Armed

When arming succeeds, you should see:

- **Telemetry Status**: Changes from "DISARMED" (orange) to "ARMED" (red)
- **Motors**: Spin up slightly (props off, you'll hear it)
- **Console**: Shows "Arming command sent"

**Next steps:**
- Test throttle response (slowly increase throttle slider)
- Test roll/pitch/yaw response
- When ready, disarm and install props for flight

---

## Common Solutions Summary

| Problem | Solution |
|---------|----------|
| "Unexpected error: unpack" | Fixed in latest version |
| Channels don't move | Enable RX_MSP feature |
| ARM button does nothing | Configure ARM on AUX1, range 1700-2100 |
| "THROTTLE" flag | Set throttle to 1000 before arming |
| "NOGYRO" flag | Wait 10 seconds after power-on |
| "RXLOSS" flag | Normal with MSP, can be ignored |
| Drone arms then immediately disarms | Failsafe triggering - increase control loop rate or check connection |

---

## Getting Help

If you're still stuck:

1. **Capture console output:**
   - Connect, enable controls, try to arm
   - Copy all console messages

2. **Check Betaflight status:**
   - CLI command: `status`
   - Note all arming flags

3. **Verify configuration:**
   - Post your Modes tab screenshot
   - Post Receiver tab screenshot

4. **Test with Betaflight Configurator:**
   - Can you arm using Betaflight's MSP control?
   - Go to Motors tab (props off!)
   - Click "I understand the risks"
   - Try to arm there

With this information, we can diagnose the specific issue!

---

**Remember: Safety first! Always remove propellers when testing arming.**
