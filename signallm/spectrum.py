"""Spectrum -> facts: PSD, noise floor estimate, peak grouping."""
from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal
from scipy import stats
from scipy.ndimage import median_filter, uniform_filter1d

from .capture import Capture

USABLE = 0.8       # fraction of the band free of analog filter roll-off at the edges
DC_GUARD_HZ = 60e3  # LO leakage / DC offset around the center


def psd(cap: Capture, rbw_hz: float = 5e3):
    """Welch PSD. Returns (absolute freq in Hz, dB relative to full scale), sorted by frequency."""
    nperseg = int(2 ** np.round(np.log2(cap.fs / rbw_hz)))
    f, p = signal.welch(cap.iq, fs=cap.fs, nperseg=nperseg, noverlap=nperseg // 2,
                        return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    return f + cap.center_hz, 10 * np.log10(p + 1e-20)


def noise_floor(f: np.ndarray, db: np.ndarray, win_hz: float = 4e6) -> np.ndarray:
    """Noise floor as a running median of the lower half of values: stays below wide signals
    and follows the tilt of the filter passband."""
    df = f[1] - f[0]
    k = max(3, int(win_hz / df) | 1)
    low = np.minimum(db, median_filter(db, size=k, mode="nearest"))  # clip the tops
    return median_filter(low, size=k, mode="nearest")


def time_floor(cap: Capture, f_out: np.ndarray, max_secs: float = 0.5, nperseg: int = 512,
               smooth_bins: int = 16, quantile: float = 0.1) -> np.ndarray:
    """Noise floor from the quiet moments in time: a low quantile of the (frequency-smoothed)
    spectrogram per bin, interpolated onto f_out (dB, same scale as psd()).

    Unlike the frequency-median floor this stays correct for bursty signals that fill the whole
    span (WiFi, a microwave oven): the gaps between bursts show the true noise. A signal that is
    on continuously still raises it, which no single capture can tell apart from noise.
    """
    iq = cap.iq[: int(max_secs * cap.fs)]
    f, _, sxx = signal.spectrogram(iq, fs=cap.fs, nperseg=nperseg, noverlap=0,
                                   return_onesided=False, detrend=False, mode="psd")
    sxx = uniform_filter1d(sxx, smooth_bins, axis=0, mode="nearest")
    low = np.quantile(sxx, quantile, axis=1)
    # smoothed noise power is ~ chi2(2k)/(2k) around its mean: undo the bias of the low quantile
    k = smooth_bins
    low = low / (stats.chi2.ppf(quantile, 2 * k) / (2 * k))
    order = np.argsort(f)
    fa = f[order] + cap.center_hz
    return np.interp(f_out, fa, 10 * np.log10(low[order] + 1e-20))


@dataclass
class Signal:
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    snr_db: float
    flatness_db: float  # peak - mean in band: small for a WiFi plateau, large for a carrier
    confirmed: bool | None = None  # None = not checked with a second capture
    burst: dict | None = None      # time-domain stats, see spectrogram.py

    def to_dict(self):
        d = asdict(self)
        d = {k: round(v, 1) if isinstance(v, float) else v for k, v in d.items()}
        return {k: v for k, v in d.items() if v is not None}


def find_signals(cap: Capture, thresh_db: float = 6.0, merge_hz: float = 150e3,
                 min_bw_hz: float = 0.0, rbw_hz: float = 5e3) -> dict:
    f, db = psd(cap, rbw_hz)
    df = f[1] - f[0]
    floor = np.minimum(noise_floor(f, db), time_floor(cap, f))
    sm = uniform_filter1d(db, max(1, int(3 * rbw_hz / df) | 1))  # smoothing against noise ripple
    snr = sm - floor

    half = cap.fs * USABLE / 2
    valid = np.abs(f - cap.center_hz) <= half
    valid &= np.abs(f - cap.center_hz) > DC_GUARD_HZ
    hot = (snr > thresh_db) & valid

    # merge nearby regions; bandwidth is the extent above the threshold
    idx = np.flatnonzero(hot)
    groups = []
    if len(idx):
        start = prev = idx[0]
        gap = int(merge_hz / df)
        for i in idx[1:]:
            if i - prev > gap:
                groups.append((start, prev)); start = i
            prev = i
        groups.append((start, prev))

    sigs = []
    for a, b in groups:
        seg = slice(a, b + 1)
        pk = int(np.argmax(sm[seg])) + a
        peak = float(sm[pk])
        above = np.flatnonzero(snr[seg] > thresh_db)
        bw = (above[-1] - above[0] + 1) * df
        if bw < min_bw_hz:
            continue
        lin = 10 ** (db[seg] / 10)
        center = float(np.sum(f[seg] * lin) / np.sum(lin))
        sigs.append(Signal(center, bw, peak, float(snr[pk]),
                           float(peak - 10 * np.log10(np.mean(lin)))))
    sigs.sort(key=lambda s: -s.snr_db)
    return {
        "center_hz": cap.center_hz,
        "span_hz": cap.fs * USABLE,
        "noise_floor_db": round(float(np.median(floor[valid])), 1),
        "signals": sigs,
        "_psd": (f, db, floor),
    }
