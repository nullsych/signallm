"""Step 1 orchestrator: capture(s) -> JSON facts."""
import argparse
import json

from .capture import Capture, SoapyRadio, load_sc16
from .heuristics import interpret
from .spectrogram import burst_stats, save_png, stft_db
from .spectrum import find_signals
from .spurs import dual_capture_filter

STFT_MAX_SECS = 0.5  # bound memory: 0.5 s at 20 MS/s ~ 20k frames


def analyze(cap_a: Capture, cap_b: Capture | None = None, png: str | None = None) -> dict:
    ra = find_signals(cap_a)
    spurs = []
    if cap_b is not None:
        rb = find_signals(cap_b)
        signals, spurs = dual_capture_filter(ra, rb)
    else:
        signals = ra["signals"]

    n = int(STFT_MAX_SECS * cap_a.fs)
    t, f, db = stft_db(Capture(cap_a.iq[:n], cap_a.center_hz, cap_a.fs))
    for s in signals:
        if abs(s.center_hz - cap_a.center_hz) <= ra["span_hz"] / 2:
            s.burst = burst_stats(t, f, db, s) or None
    if png:
        save_png(t, f, db, png, signals)

    facts = {
        "center_hz": cap_a.center_hz,
        "span_hz": ra["span_hz"],
        "duration_s": round(cap_a.duration, 3),
        "noise_floor_db": ra["noise_floor_db"],
        "dual_capture": cap_b is not None,
        "signals": [s.to_dict() for s in signals],
        "spurs_rejected": [s.to_dict() for s in spurs],
    }
    facts["interpretation"] = interpret(facts)
    return facts


def main():
    ap = argparse.ArgumentParser(description="Spectrum -> JSON facts")
    ap.add_argument("--file", help="sc16 file instead of a live capture")
    ap.add_argument("--file-b", help="second sc16 file with a different center")
    ap.add_argument("--center", type=float, required=True, help="center frequency, Hz")
    ap.add_argument("--center-b", type=float, help="second capture center, Hz (default: center + 0.2*fs)")
    ap.add_argument("--fs", type=float, default=20e6)
    ap.add_argument("--secs", type=float, default=1.0)
    ap.add_argument("--gain", type=float)
    ap.add_argument("--single", action="store_true", help="skip spur rejection")
    ap.add_argument("--png", help="save the spectrogram")
    a = ap.parse_args()

    if a.file:
        cap_a = load_sc16(a.file, a.center, a.fs)
        cap_b = load_sc16(a.file_b, a.center_b, a.fs) if a.file_b else None
    else:
        radio = SoapyRadio()
        cap_a = radio.capture(a.center, a.fs, a.secs, a.gain)
        cap_b = None if a.single else radio.capture(a.center_b or a.center + a.fs * 0.2, a.fs, a.secs, a.gain)
    print(json.dumps(analyze(cap_a, cap_b, a.png), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
