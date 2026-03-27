"""
hardware_reader.py
==================
Background thread that manages all serial communication with the Arduino.

Arduino data format (one JSON object per line, every 500 ms):
    {"vibration":145,"temperature":28.50,"humidity":62.00}

    vibration  — integer 0-1023 (analogRead)
    temperature — float °C, or -1 if DHT11 failed
    humidity    — float %, or -1 if DHT11 failed

Commands sent back to Arduino (accepted strings + newline):
    "GREEN\n"   — green LED on, buzzer off
    "YELLOW\n"  — yellow LED on, buzzer off
    "RED\n"     — red LED on, buzzer at 1 kHz

Design goals
------------
* Never blocks the main server — runs entirely in a daemon thread.
* Auto-discovers the Arduino across common COM ports.
* Maintains rolling history so averaged values are stable (ignores single spikes).
* Produces a single hardware_risk_score in [0, 1] combining vibration + temp/hum trend.
* Returns zeros and sets is_connected=False silently on any hardware failure.
"""

import json
import math
import threading
import time
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Optional import — if pyserial is absent the class still loads but hardware
# will never connect (is_connected stays False).
# ---------------------------------------------------------------------------
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[HardwareReader] WARNING: pyserial not installed. "
          "Hardware sensors disabled. Run: pip install pyserial")


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

BAUD_RATE           = 9600          # Must match Arduino sketch
CONNECT_TIMEOUT_S   = 2.0           # How long to wait for port open
READ_TIMEOUT_S      = 1.0           # Serial read timeout per line
RECONNECT_DELAY_S   = 5.0           # Pause between reconnect attempts

# Smaller history = rolling average reacts faster to individual taps.
# 8 samples @ one reading per 500 ms = ~4 s response window.
# A single firm tap dominates for ~2–3 s then falls back naturally.
HISTORY_SIZE        = 8             # Rolling window (was 30 — shortened for demo)
TREND_WINDOW_OLD    = 5             # Older slice for trend comparison
TREND_WINDOW_NEW    = 3             # Latest N samples for trend comparison

# ---------------------------------------------------------------------------
# Vibration thresholds — demo calibrated
# ---------------------------------------------------------------------------
# The classification compares the ROLLING AVERAGE (vib_avg), not raw peaks.
# With HISTORY_SIZE=8, a single tap that spikes to ~400 ADC moves the average
# by roughly 400/8 = 50 ADC counts — so thresholds are set accordingly.
#
#   Resting / no touch      →  avg ≤ 50   →  LOW    (GREEN LED)
#   Light tap / gentle hold →  avg ≤ 200  →  MEDIUM (YELLOW LED)
#   Firm / sustained tap    →  avg > 200  →  HIGH   (RED LED + buzzer)
#
# Gaps between bands:   LOW→MEDIUM = 150 ADC apart
#                       MEDIUM→HIGH = 150 ADC apart
# Wide gaps prevent accidentally jumping between states.
VIB_LOW_MAX         = 50            # ≤ this  → LOW   (resting, no touch)
VIB_MED_MAX         = 200           # ≤ this  → MEDIUM (light tap)
# > VIB_MED_MAX                     →           HIGH  (firm / sustained)

# Erratic std threshold: if avg is in the MEDIUM zone but std is very high,
# promote straight to HIGH (erratic = panic-like motion).
VIB_ERRATIC_STD     = 40            # High std within medium zone → HIGH

# Temperature / humidity rising-trend thresholds
TEMP_RISE_THRESHOLD = 1.0           # °C rise (old-avg → new-avg) = crowd heat signal
HUM_RISE_THRESHOLD  = 3.0           # % rise  (old-avg → new-avg) = crowd humidity signal

# Risk weighting: vibration reacts faster so it carries more weight
WEIGHT_VIBRATION    = 0.65
WEIGHT_ENV_TREND    = 0.35

# Common Arduino COM ports to probe (Windows + Linux/Mac)
# COM9 is listed first because that is where this Arduino is detected by the OS.
# The auto-discovery at runtime will also prepend any live OS-reported ports,
# so this list is just a sensible fallback ordering.
CANDIDATE_PORTS = [
    # Highest-priority — known Arduino port on this machine
    "COM9",
    # Windows (rest)
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
    "COM7", "COM8", "COM10", "COM11", "COM12",
    # Linux / macOS
    "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2",
    "/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2",
    "/dev/cu.usbmodem1401", "/dev/cu.usbserial-1410",
]


# ---------------------------------------------------------------------------
# Vibration level enumeration
# ---------------------------------------------------------------------------

class VibrationLevel:
    NORMAL  = "normal"    # Low, consistent — normal walking
    MEDIUM  = "medium"    # Sustained — heavy crowd movement
    HIGH    = "high"      # Erratic spikes — panic / stampede


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HardwareReader:
    """
    Reads vibration and environmental (temp/hum) data from an Arduino over
    serial in a background daemon thread.

    Usage
    -----
        reader = HardwareReader()
        reader.start()

        # Later, in your main loop:
        data = reader.get_current_data()
        score = data['hardware_risk_score']   # 0.0 – 1.0
    """

    def __init__(self):
        # Connection state
        self.is_connected: bool = False
        self._port_name: Optional[str] = None
        self._serial: Optional["serial.Serial"] = None  # type: ignore[name-defined]
        self._lock = threading.Lock()

        # Rolling history deques
        self._vib_history:  deque = deque(maxlen=HISTORY_SIZE)
        self._temp_history: deque = deque(maxlen=HISTORY_SIZE)
        self._hum_history:  deque = deque(maxlen=HISTORY_SIZE)

        # Latest computed metrics (safe to read from any thread)
        self._latest: dict = self._zero_data()

        # Background thread
        self._thread = threading.Thread(
            target=self._run_loop,
            name="HardwareReaderThread",
            daemon=True          # Dies automatically when main process exits
        )
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread.start()
        print("[HardwareReader] " + "=" * 50)
        print("[HardwareReader] Background thread started.")
        print("[HardwareReader] Scanning for Arduino on all serial ports...")
        print("[HardwareReader] IMPORTANT: Close Arduino IDE Serial Monitor")
        print("[HardwareReader]            before running this — both cannot")
        print("[HardwareReader]            hold the port at the same time.")
        print("[HardwareReader] " + "=" * 50)

    def stop(self) -> None:
        """Signal the background thread to stop and close the port."""
        self._stop_event.set()
        self._close_port()
        print("[HardwareReader] Stopped.")

    def get_current_data(self) -> dict:
        """
        Thread-safe snapshot of the latest hardware metrics.

        Returns
        -------
        dict with keys:
            is_connected       : bool
            port               : str | None
            vibration_raw      : float   — latest raw ADC value
            vibration_avg      : float   — rolling average
            vibration_std      : float   — rolling std-dev (erratic indicator)
            vibration_level    : str     — "normal" | "medium" | "high"
            temperature_avg    : float   — rolling average °C
            humidity_avg       : float   — rolling average %
            temp_trend         : float   — °C rise over trend window (negative = falling)
            hum_trend          : float   — % rise over trend window
            env_risk           : float   — 0-1 score from temperature/humidity trend
            vibration_risk     : float   — 0-1 score from vibration classification
            hardware_risk_score: float   — combined weighted score (0-1)
            history_size       : int     — number of readings in buffer so far
        """
        with self._lock:
            return dict(self._latest)

    def send_command(self, command: str) -> bool:
        """
        Send a newline-terminated command string to the Arduino.

        Parameters
        ----------
        command : str  — one of "GREEN", "YELLOW", "RED"
                         (Arduino ignores anything else)

        Returns
        -------
        True if sent successfully, False if not connected or write failed.
        """
        if not self.is_connected or self._serial is None:
            return False
        try:
            # Arduino reads until newline — encode as bytes
            payload = (command.strip() + "\n").encode("utf-8")
            with self._lock:
                self._serial.write(payload)
            return True
        except Exception as exc:
            print(f"[HardwareReader] send_command failed: {exc}")
            self._handle_disconnect()
            return False

    # ------------------------------------------------------------------
    # Internal — background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main loop: connect → read → reconnect on failure."""
        while not self._stop_event.is_set():
            if not self.is_connected:
                self._attempt_connect()
                if not self.is_connected:
                    # Pause before retrying so we don't spam log
                    self._stop_event.wait(RECONNECT_DELAY_S)
                    continue

            # --- connected: read one line ---
            # _read_line returns:
            #   str  (possibly empty) — successful readline (empty = 1-s timeout, normal)
            #   None                  — serial exception → must disconnect
            line = self._read_line()
            if line is None:
                # A real serial exception occurred — disconnect and retry
                self._handle_disconnect()
                continue

            # Empty string just means readline() timed out (1 s) with no data —
            # the Arduino sends every 500 ms so this is normal; loop again.
            if line == "":
                continue

            self._parse_and_update(line)
            # Print every received line so the terminal confirms data flow
            # (remove or comment this out once confirmed working)
            print(f"[HardwareReader] RAW << {line}")

        # Thread exiting cleanly
        self._close_port()

    # ------------------------------------------------------------------
    # Internal — serial communication helpers
    # ------------------------------------------------------------------

    def _attempt_connect(self) -> None:
        """Try every candidate port until one responds with valid Arduino data."""
        if not SERIAL_AVAILABLE:
            return

        # Build union of candidate ports + any port currently listed by OS
        # Live OS ports are prepended — they are far more likely to be the Arduino
        ports_to_try = list(CANDIDATE_PORTS)
        try:
            live_ports = [p.device for p in serial.tools.list_ports.comports()]
            for p in live_ports:
                if p not in ports_to_try:
                    ports_to_try.insert(0, p)
        except Exception:
            pass

        for port in ports_to_try:
            if self._stop_event.is_set():
                return
            try:
                print(f"[HardwareReader] Trying {port} @ {BAUD_RATE} baud …")
                # 2-second open timeout — move to next port immediately on failure
                ser = serial.Serial(
                    port=port,
                    baudrate=BAUD_RATE,
                    timeout=2.0,          # 2 s read timeout during probe
                )
                # Give Arduino time to reset after port open, then clear junk
                # Arduino resets when DTR is asserted (port opened) — needs ~2 s
                time.sleep(2.0)
                ser.reset_input_buffer()

                # Probe: attempt up to 3 lines; accept first valid JSON
                found = False
                for _ in range(3):
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()
                    if self._is_valid_arduino_json(raw):
                        found = True
                        break

                if found:
                    # Switch read timeout to normal operating value
                    ser.timeout = READ_TIMEOUT_S
                    with self._lock:
                        self._serial = ser
                        self._port_name = port
                        self.is_connected = True
                    print(f"[HardwareReader] ✅ Connected on {port}")
                    return  # Success — exit port scan

                ser.close()   # No valid data on this port — try next

            except Exception:
                # Port busy, does not exist, or any other error — move on
                pass

        print("[HardwareReader] No Arduino found on any port. Will retry.")

    def _read_line(self) -> Optional[str]:
        """
        Read one newline-terminated line from the Arduino.

        Returns
        -------
        str   — decoded, stripped line (may be "" if readline timed out normally)
        None  — a real serial exception occurred; caller must disconnect
        """
        if self._serial is None:
            return None
        try:
            # readline() blocks up to READ_TIMEOUT_S then returns b""
            raw = self._serial.readline()
            # Decode, strip whitespace and newline characters
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            print(f"[HardwareReader] Read error: {exc}")
            # Return None to signal a real error — caller will disconnect
            return None

    def _handle_disconnect(self) -> None:
        """Mark as disconnected and reset metrics to zero."""
        with self._lock:
            self.is_connected = False
            self._latest = self._zero_data()
        self._close_port()
        print("[HardwareReader] ⚠️  Disconnected from Arduino. Will retry.")

    def _close_port(self) -> None:
        """Safely close the serial port."""
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass
        self._serial = None

    # ------------------------------------------------------------------
    # Internal — data parsing and metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_arduino_json(line: str) -> bool:
        """Return True if line is a JSON object with the 'vibration' key."""
        if not line or not line.startswith("{"):
            return False
        try:
            obj = json.loads(line)
            # Arduino sends exactly: vibration, temperature, humidity
            return "vibration" in obj
        except json.JSONDecodeError:
            return False

    def _parse_and_update(self, line: str) -> None:
        """
        Parse one JSON line from the Arduino, update rolling histories,
        and recompute all derived metrics.

        Arduino format: {"vibration":145,"temperature":28.50,"humidity":62.00}
        DHT failure:    temperature and/or humidity will be -1
        """
        if not line:
            return

        # ---- JSON parse — silently skip empty or malformed lines ----
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return  # Malformed line — wait for next one

        # ---- Extract values using the exact Arduino key names ----
        if "vibration" not in obj:
            return  # Not an Arduino data line — ignore

        vib  = float(obj["vibration"])
        temp = float(obj.get("temperature", -1))
        hum  = float(obj.get("humidity",    -1))

        with self._lock:
            # Always append vibration — it is never invalid
            self._vib_history.append(vib)

            # DHT11 sends -1 on sensor failure — keep the last known good value
            # by only appending when the reading is valid (>= 0)
            if temp >= 0:
                self._temp_history.append(temp)
            if hum >= 0:
                self._hum_history.append(hum)

            # Recompute all metrics immediately so get_current_data() is fresh
            self._latest = self._compute_metrics(vib)
            # Ensure connected flag is visible to readers
            self._latest["is_connected"] = True

    def _compute_metrics(self, latest_vib: float) -> dict:
        """
        Compute all derived metrics from the current rolling histories.
        Must be called with self._lock held.
        """
        vib_list  = list(self._vib_history)
        temp_list = list(self._temp_history)
        hum_list  = list(self._hum_history)

        n = len(vib_list)

        # ---- Vibration statistics ----------------------------------------
        vib_avg = _safe_mean(vib_list)
        vib_std = _safe_std(vib_list)

        vib_level, vib_risk = self._classify_vibration(vib_avg, vib_std)

        # ---- Temperature / humidity averages ------------------------------
        temp_avg = _safe_mean(temp_list)
        hum_avg  = _safe_mean(hum_list)

        # ---- Temperature / humidity trend ----------------------------------
        temp_trend, hum_trend, env_risk = self._compute_env_trend(
            temp_list, hum_list, n
        )

        # ---- Combined hardware risk score ----------------------------------
        hardware_risk_score = min(
            1.0,
            WEIGHT_VIBRATION * vib_risk + WEIGHT_ENV_TREND * env_risk
        )

        return {
            "is_connected":        True,
            "port":                self._port_name,
            "vibration_raw":       float(latest_vib),
            "vibration_avg":       float(round(vib_avg, 2)),
            "vibration_std":       float(round(vib_std, 2)),
            "vibration_level":     vib_level,
            "temperature_avg":     float(round(temp_avg, 2)),
            "humidity_avg":        float(round(hum_avg, 2)),
            "temp_trend":          float(round(temp_trend, 3)),
            "hum_trend":           float(round(hum_trend, 3)),
            "env_risk":            float(round(env_risk, 3)),
            "vibration_risk":      float(round(vib_risk, 3)),
            "hardware_risk_score": float(round(hardware_risk_score, 3)),
            "history_size":        n,
        }

    # ------------------------------------------------------------------
    # Internal — classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_vibration(avg: float, std: float) -> tuple:
        """
        Classify vibration level and return (level_str, risk_score 0-1).

        Rules
        -----
        HIGH  : mean > VIB_MED_MAX  OR  (mean > VIB_LOW_MAX AND std > VIB_ERRATIC_STD)
        MEDIUM: mean > VIB_LOW_MAX  AND std <= VIB_ERRATIC_STD
        NORMAL: everything else
        """
        if avg > VIB_MED_MAX or (avg > VIB_LOW_MAX and std > VIB_ERRATIC_STD):
            # Panic / stampede — map linearly but floor at 0.6
            # Clamp so that values well above threshold reach 1.0
            cap = 800.0          # ADC value at which risk saturates
            raw_score = 0.60 + 0.40 * min(1.0, (avg - VIB_MED_MAX) / (cap - VIB_MED_MAX))
            # Erratic spikes boost the score slightly
            erratic_bonus = min(0.10, std / (VIB_ERRATIC_STD * 4))
            return VibrationLevel.HIGH, min(1.0, raw_score + erratic_bonus)

        elif avg > VIB_LOW_MAX:
            # Heavy crowd movement — medium, consistent (low std)
            # Score range: 0.25 – 0.59
            t = (avg - VIB_LOW_MAX) / (VIB_MED_MAX - VIB_LOW_MAX)  # 0→1
            return VibrationLevel.MEDIUM, 0.25 + 0.35 * t

        else:
            # Normal walking
            t = avg / max(VIB_LOW_MAX, 1.0)       # 0→1 within normal range
            return VibrationLevel.NORMAL, 0.0 + 0.24 * t

    @staticmethod
    def _compute_env_trend(temp_list: list, hum_list: list, n: int) -> tuple:
        """
        Detect a rising trend in temperature and humidity.

        Strategy
        --------
        Compare the mean of the oldest TREND_WINDOW_OLD readings (before
        index -TREND_WINDOW_NEW) with the mean of the most recent
        TREND_WINDOW_NEW readings.  A positive delta signals a crowd-heat
        buildup.

        Returns (temp_trend, hum_trend, env_risk 0-1).
        """
        if n < TREND_WINDOW_NEW + 2:
            # Not enough data yet — no penalty
            return 0.0, 0.0, 0.0

        recent_temp = temp_list[-TREND_WINDOW_NEW:]
        recent_hum  = hum_list[-TREND_WINDOW_NEW:]

        if n >= TREND_WINDOW_OLD + TREND_WINDOW_NEW:
            old_end   = n - TREND_WINDOW_NEW
            old_start = max(0, old_end - TREND_WINDOW_OLD)
            older_temp = temp_list[old_start:old_end]
            older_hum  = hum_list[old_start:old_end]
        else:
            # Use whatever old data we have
            cutoff = n - TREND_WINDOW_NEW
            older_temp = temp_list[:cutoff] if cutoff > 0 else [temp_list[0]]
            older_hum  = hum_list[:cutoff]  if cutoff > 0 else [hum_list[0]]

        temp_trend = _safe_mean(recent_temp) - _safe_mean(older_temp)
        hum_trend  = _safe_mean(recent_hum)  - _safe_mean(older_hum)

        # Score contribution: positive trend above thresholds → nonzero risk
        temp_component = max(0.0, temp_trend - TEMP_RISE_THRESHOLD) / 5.0   # 5 °C rise → 1.0
        hum_component  = max(0.0, hum_trend  - HUM_RISE_THRESHOLD)  / 15.0  # 15 % rise → 1.0

        env_risk = min(1.0, 0.5 * temp_component + 0.5 * hum_component)
        return temp_trend, hum_trend, env_risk

    # ------------------------------------------------------------------
    # Internal — zero / disconnected state
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_data() -> dict:
        return {
            "is_connected":        False,
            "port":                None,
            "vibration_raw":       0.0,
            "vibration_avg":       0.0,
            "vibration_std":       0.0,
            "vibration_level":     VibrationLevel.NORMAL,
            "temperature_avg":     0.0,
            "humidity_avg":        0.0,
            "temp_trend":          0.0,
            "hum_trend":           0.0,
            "env_risk":            0.0,
            "vibration_risk":      0.0,
            "hardware_risk_score": 0.0,
            "history_size":        0,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (no lock needed — pure functions)
# ---------------------------------------------------------------------------

def _safe_mean(data: list) -> float:
    """Return mean of a list, or 0.0 if empty."""
    return sum(data) / len(data) if data else 0.0


def _safe_std(data: list) -> float:
    """Return population std-dev of a list, or 0.0 if fewer than 2 elements."""
    if len(data) < 2:
        return 0.0
    mean = _safe_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python hardware_reader.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=== HardwareReader self-test (simulated data) ===\n")

    reader = HardwareReader()

    # Inject simulated readings directly (bypasses serial)
    def _simulate():
        phase = 0
        while True:
            phase += 1
            # Simulate escalating scenario
            if phase < 10:
                vib  = random.uniform(50, 120)      # Normal walking
                temp = random.uniform(27, 28)
                hum  = random.uniform(60, 62)
            elif phase < 20:
                vib  = random.uniform(200, 380)     # Heavy movement
                temp = random.uniform(28, 29)
                hum  = random.uniform(63, 66)
            else:
                vib  = random.uniform(450, 900)     # Panic
                temp = random.uniform(30, 32)
                hum  = random.uniform(68, 75)

            fake_line = json.dumps({"vibration": int(round(vib)),
                                    "temperature": round(temp, 2),
                                    "humidity":    round(hum, 2)})
            with reader._lock:
                reader._vib_history.append(vib)
                reader._temp_history.append(temp)
                reader._hum_history.append(hum)
                reader._latest = reader._compute_metrics(vib)
                reader.is_connected = True

            d = reader.get_current_data()
            print(
                f"[t={phase:02d}] vib_avg={d['vibration_avg']:6.1f}  "
                f"std={d['vibration_std']:5.1f}  level={d['vibration_level']:6s}  "
                f"temp={d['temperature_avg']:5.1f}°C  hum={d['humidity_avg']:5.1f}%  "
                f"temp_trend={d['temp_trend']:+.2f}  "
                f"hw_risk={d['hardware_risk_score']:.3f}"
            )
            time.sleep(0.3)

    sim_thread = threading.Thread(target=_simulate, daemon=True)
    sim_thread.start()

    try:
        sim_thread.join(timeout=15)
    except KeyboardInterrupt:
        pass

    print("\nSelf-test complete.")
