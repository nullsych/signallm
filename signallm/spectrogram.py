"""Spectrogram and burst statistics: bursty signals are invisible in the averaged spectrum,
while duty cycle and burst length distinguish WiFi from a sensor or a microwave oven."""
import numpy as np
from scipy import signal

from .capture import Capture
from .spectrum import Signal


def stft_db(cap: Capture, nfft: int = 512, hop: int | None = None):
    """Returns (t_sec, freq_hz, dB[t, f]) with the frequency axis fft-shifted."""
    hop = hop or nfft
    f, t, z = signal.stft(cap.iq, fs=cap.fs, nperseg=nfft, noverlap=nfft - hop,
                          return_onesided=False, boundary=None, padded=False)
    z = np.fft.fftshift(z, axes=0)
    p = np.abs(z.T) ** 2
    return t, np.fft.fftshift(f) + cap.center_hz, 10 * np.log10(p + 1e-20)


def burst_stats(t: np.ndarray, f: np.ndarray, db: np.ndarray, sig: Signal, thresh_db: float = 6.0) -> dict:
    """In-band power over time -> threshold -> bursts, duty cycle, periodicity."""
    band = np.abs(f - sig.center_hz) <= max(sig.bandwidth_hz, 2 * (f[1] - f[0])) / 2
    if not band.any():
        return {}
    lin = 10 ** (db[:, band] / 10)
    pw = 10 * np.log10(lin.mean(axis=1) + 1e-20)
    dt = t[1] - t[0]
    # on/off split: noise ~ lower percentile, threshold thresh_db above it
    floor = np.percentile(pw, 20)
    on = pw > floor + thresh_db
    # close single-frame gaps inside a burst
    on = np.convolve(on.astype(int), np.ones(3, int), "same") >= 2 if on.sum() > 3 else on

    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dur = (ends - starts) * dt
    out = {
        "duty_cycle": round(float(on.mean()), 3),
        "burst_count": int(len(starts)),
        "modulation_depth_db": round(float(np.percentile(pw, 95) - floor), 1),
    }
    if len(starts):
        out["burst_ms_median"] = round(float(np.median(dur)) * 1e3, 3)
        out["burst_ms_max"] = round(float(dur.max()) * 1e3, 3)
    if len(starts) >= 4:
        period = np.median(np.diff(starts)) * dt
        out["period_ms"] = round(float(period) * 1e3, 3)
    # envelope periodicity (50 Hz for a microwave oven): autocorrelation peak
    x = pw - pw.mean()
    if x.std() > 1.0 and len(x) > 64:
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac /= ac[0]
        lo, hi = int(2e-3 / dt), min(int(0.1 / dt), len(ac) - 1)
        if hi > lo:
            k = lo + int(np.argmax(ac[lo:hi]))
            if ac[k] > 0.4:
                out["envelope_period_ms"] = round(k * dt * 1e3, 2)
                out["envelope_period_strength"] = round(float(ac[k]), 2)
    return out


def save_png(t, f, db, path: str, signals=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # decimate in time for the image, otherwise tens of thousands of rows
    step = max(1, len(t) // 1500)
    d = db[::step]
    lo = np.percentile(d, 5)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(d, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=lo + 45,
              extent=[f[0] / 1e6, f[-1] / 1e6, t[0] * 1e3, t[-1] * 1e3])
    for s in signals:
        ax.axvline(s.center_hz / 1e6, color="w", lw=0.5, ls=":")
    ax.set_xlabel("MHz"); ax.set_ylabel("ms")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
