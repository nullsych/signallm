"""Spur rejection using two captures with different centers.

A real signal stays at the same absolute frequency whatever the center. A spur (DC, LO leakage,
IQ image, clock harmonics) stays at the same offset from the center, so when the center moves it
shifts too and is absent at the original frequency in the second capture.
"""
from .spectrum import Signal


def _covers(res: dict, f: float) -> bool:
    return abs(f - res["center_hz"]) <= res["span_hz"] / 2


def _confirm(s: Signal, other: dict, tol_hz: float, snr_tol_db: float) -> bool | None:
    if not _covers(other, s.center_hz):
        return None  # outside the second capture's span: cannot be verified
    tol = max(tol_hz, s.bandwidth_hz / 4)
    return any(abs(t.center_hz - s.center_hz) <= tol and abs(t.snr_db - s.snr_db) <= snr_tol_db
               for t in other["signals"])


def dual_capture_filter(a: dict, b: dict, tol_hz: float = 100e3, snr_tol_db: float = 8.0):
    """Returns (signals, spurs), merging both captures without duplicates."""
    signals, spurs = [], []
    for res, other in ((a, b), (b, a)):
        for s in res["signals"]:
            s.confirmed = _confirm(s, other, tol_hz, snr_tol_db)
            if s.confirmed is False:
                spurs.append(s)
            elif res is a or not _covers(a, s.center_hz):  # from b keep only what a does not cover
                signals.append(s)
    signals.sort(key=lambda s: s.center_hz)
    return signals, spurs
