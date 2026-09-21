"""Synthetic 2.4 GHz captures for testing the heuristics without hardware.

All levels are given as in-band PSD SNR over the noise floor, in dB, matching how find_signals
measures them.
"""
import numpy as np

from signallm.capture import Capture

FS = 20e6
CENTER = 2437e6


def _rng(seed):
    return np.random.default_rng(seed)


def noise(n, rng, tilt_db=3.0):
    """Complex white noise (unit power per sample) with a mild passband tilt like the real front end."""
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) / np.sqrt(2)
    if tilt_db:
        X = np.fft.fft(x)
        f = np.fft.fftfreq(n, 1 / FS)
        X *= (10 ** (tilt_db * (f / (FS / 2)) / 20)).astype(np.float32)
        x = np.fft.ifft(X).astype(np.complex64)
        x /= x.std()
    return x


def band_noise(n, rng, offset_hz, bw_hz, snr_db):
    """Noise confined to [offset - bw/2, offset + bw/2] (offset from the capture center)."""
    X = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    f = np.fft.fftfreq(n, 1 / FS)
    X[np.abs(f - offset_hz) > bw_hz / 2] = 0
    x = np.fft.ifft(X)
    x /= x.std()
    return (x * np.sqrt(10 ** (snr_db / 10) * bw_hz / FS)).astype(np.complex64)


def gate(n, on_s, period_s, jitter=0.0, rng=None, start_s=0.0):
    """On/off envelope: bursts of on_s seconds every period_s."""
    g = np.zeros(n, np.float32)
    on, per = int(on_s * FS), int(period_s * FS)
    t = int(start_s * FS)
    while t < n:
        g[t:t + on] = 1
        t += per + (int(rng.uniform(-jitter, jitter) * per) if jitter else 0)
    return g


def make(*components, secs=0.3, seed=0, center=CENTER, tilt_db=3.0):
    """components: callables (n, rng) -> complex array, summed onto the noise."""
    rng = _rng(seed)
    n = int(secs * FS)
    x = noise(n, rng, tilt_db)
    for c in components:
        x = x + c(n, rng)
    return Capture(x.astype(np.complex64), center, FS)


# --- signal models -------------------------------------------------------------------------

def wifi(offset_hz=0.0, snr_db=25, bw_hz=16.6e6, duty=0.15, packet_s=300e-6):
    """OFDM-like plateau, bursty traffic: packets of ~300 us with random gaps."""
    def c(n, rng):
        x = band_noise(n, rng, offset_hz, bw_hz, snr_db)
        g = np.zeros(n, np.float32)
        t = 0
        while t < n:
            t += int(rng.exponential(packet_s / duty) * FS)
            g[t:t + int(packet_s * FS)] = 1
        return x * g
    return c


def ble_adv(offset_hz, snr_db=25, bw_hz=1.6e6, burst_s=250e-6, period_s=20e-3):
    """GFSK-ish advertising burst: narrow band, short, sparse."""
    def c(n, rng):
        return band_noise(n, rng, offset_hz, bw_hz, snr_db) * gate(n, burst_s, period_s, 0.1, rng, 1e-3)
    return c


def microwave(offset_hz=0.0, snr_db=20, bw_hz=12e6, period_s=20e-3):
    """Magnetron on for half of each mains cycle (50 Hz): 10 ms on, 10 ms off."""
    def c(n, rng):
        return band_noise(n, rng, offset_hz, bw_hz, snr_db) * gate(n, period_s / 2, period_s)
    return c


def carrier(offset_hz, snr_db=30):
    """Unmodulated CW tone."""
    def c(n, rng):
        t = np.arange(n) / FS
        return (np.exp(2j * np.pi * offset_hz * t) * np.sqrt(10 ** (snr_db / 10) * 5e3 / FS)).astype(np.complex64)
    return c
