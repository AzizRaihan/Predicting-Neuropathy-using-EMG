"""
sensor.py — EMG sensor interface layer

Two modes:
  - SIMULATION : generates synthetic EMG data (use when sensor not connected)
  - LIVE       : reads raw voltage values from Arduino over USB serial

To switch to live mode in app.py, set mode='live' and pass the correct port.

Arduino sketch should send one float per line at a fixed sample rate, e.g.:
    Serial.println(analogRead(A0) * (5.0 / 1023.0));  // voltage in volts
"""

import numpy as np
import serial
import serial.tools.list_ports
import time
from scipy.signal import welch
from scipy.integrate import simpson


# ── Port utilities ────────────────────────────────────────────────────────────

def list_ports() -> list[str]:
    """Return a list of available serial port names."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports] if ports else []


def default_port() -> str:
    """Return the first available port, or empty string if none found."""
    ports = list_ports()
    return ports[0] if ports else ''


# ── EMG feature extraction ────────────────────────────────────────────────────

def compute_features(samples: np.ndarray, fs: int = 1000) -> tuple[float, float]:
    """
    Compute EMG RMS and median frequency from a raw sample array.

    Args:
        samples : 1-D numpy array of raw voltage readings
        fs      : sampling frequency in Hz (default 1000 Hz)

    Returns:
        (emg_rms, emg_median_freq)
    """
    emg_rms = float(np.sqrt(np.mean(samples ** 2)))

    nperseg = min(len(samples), 512)
    f, Pxx = welch(samples, fs=fs, nperseg=nperseg)

    total_power = simpson(Pxx, x=f)
    if total_power > 0:
        cumulative = np.cumsum(Pxx)
        idx = np.where(cumulative >= total_power / 2)[0]
        emg_median_freq = float(f[idx[0]]) if len(idx) > 0 else 0.0
    else:
        emg_median_freq = 0.0

    return emg_rms, emg_median_freq


# ── Simulation mode ───────────────────────────────────────────────────────────

def simulate_recording(duration: int = 60, fs: int = 1000,
                       callback=None) -> tuple[float, float]:
    """
    Simulate a sensor recording with synthetic EMG data.
    callback(elapsed, total) is called each second for progress updates.

    Returns:
        (emg_rms, emg_median_freq)
    """
    total_samples = duration * fs
    t = np.linspace(0, duration, total_samples)

    # Synthetic EMG: broadband noise + low-freq muscle component
    np.random.seed(int(time.time()) % 1000)
    signal = (
        np.random.normal(0, 1, total_samples) * 15        # broadband noise
        + np.sin(2 * np.pi * 60 * t) * 8                  # 60 Hz component
        + np.sin(2 * np.pi * 120 * t) * 4                 # 120 Hz harmonic
        + np.random.normal(0, 0.5, total_samples) * 5     # low-freq drift
    ) + 100  # offset to µV range

    # Simulate time passing with progress callbacks
    for elapsed in range(1, duration + 1):
        time.sleep(1)
        if callback:
            callback(elapsed, duration)

    return compute_features(signal, fs=fs)


# ── Live sensor mode ──────────────────────────────────────────────────────────

def live_recording(port: str, duration: int = 60, fs: int = 1000,
                   baud: int = 9600, callback=None) -> tuple[float, float]:
    """
    Read raw EMG voltage values from Arduino over serial for `duration` seconds.
    Expects one float value per line from the Arduino.

    callback(elapsed, total) is called each second for progress updates.

    Returns:
        (emg_rms, emg_median_freq)
    """
    samples = []
    start_time = time.time()
    last_tick = 0

    with serial.Serial(port, baudrate=baud, timeout=1) as ser:
        time.sleep(2)  # wait for Arduino to reset after connection
        ser.reset_input_buffer()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # Progress callback once per second
            tick = int(elapsed)
            if tick > last_tick and callback:
                callback(tick, duration)
                last_tick = tick

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    samples.append(float(line))
                except ValueError:
                    pass  # skip malformed lines

    if callback:
        callback(duration, duration)

    if len(samples) < 100:
        raise ValueError(f"Only {len(samples)} samples received — check sensor connection.")

    return compute_features(np.array(samples), fs=fs)
