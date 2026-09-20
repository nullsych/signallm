"""IQ sources: live BladeRF via SoapySDR, or an sc16 file (for debugging and replay)."""
import ctypes
import glob
import os
from dataclasses import dataclass

import numpy as np

# The apt SoapySDR bladerf module is built against the distro libbladeRF; a custom build in
# /usr/local shadows it and corrupts the IQ stream (ABI mismatch). Loading the distro library
# first makes the dynamic loader reuse it for the module (same soname).
_SYSTEM_LIBBLADERF_GLOBS = ["/usr/lib/*/libbladeRF.so.2", "/usr/lib64/libbladeRF.so.2"]


def _preload_system_libbladerf():
    path = os.environ.get("SIGNALLM_LIBBLADERF")
    if path is None:
        found = [p for g in _SYSTEM_LIBBLADERF_GLOBS for p in glob.glob(g)]
        path = found[0] if found else None
    if path:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


@dataclass
class Capture:
    iq: np.ndarray  # complex64, normalized to [-1, 1]
    center_hz: float
    fs: float

    @property
    def duration(self) -> float:
        return len(self.iq) / self.fs


def load_sc16(path: str, center_hz: float, fs: float, scale: float = 2048.0) -> Capture:
    """Interleaved int16 I/Q (bladerf-cli / SC16_Q11 format)."""
    raw = np.fromfile(path, dtype=np.int16)
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / scale
    return Capture(iq, center_hz, fs)


class SoapyRadio:
    """Sole owner of the device. SoapySDR is imported lazily so the rest of the code works without it."""

    def __init__(self, args: str = "driver=bladerf", channel: int = 0):
        _preload_system_libbladerf()
        import SoapySDR
        self._S = SoapySDR
        self.dev = SoapySDR.Device(args)
        self.ch = channel

    def probe(self) -> dict:
        r = self.dev.getFrequencyRange(self._S.SOAPY_SDR_RX, self.ch)
        sr = self.dev.getSampleRateRange(self._S.SOAPY_SDR_RX, self.ch)
        bw = self.dev.getBandwidthRange(self._S.SOAPY_SDR_RX, self.ch)
        return {
            "freq_hz": [min(x.minimum() for x in r), max(x.maximum() for x in r)],
            "sample_rate_hz": [min(x.minimum() for x in sr), max(x.maximum() for x in sr)],
            "bandwidth_hz": [min(x.minimum() for x in bw), max(x.maximum() for x in bw)],
        }

    def capture(self, center_hz: float, fs: float, secs: float, gain_db: float | None = None,
                settle_secs: float = 0.05, max_overflows: int = 20) -> Capture:
        S, dev, ch = self._S, self.dev, self.ch
        dev.setSampleRate(S.SOAPY_SDR_RX, ch, fs)
        dev.setBandwidth(S.SOAPY_SDR_RX, ch, fs * 0.8)
        dev.setFrequency(S.SOAPY_SDR_RX, ch, center_hz)
        if gain_db is None:
            dev.setGainMode(S.SOAPY_SDR_RX, ch, True)
        else:
            dev.setGainMode(S.SOAPY_SDR_RX, ch, False)
            dev.setGain(S.SOAPY_SDR_RX, ch, gain_db)
        n = int(secs * fs)
        skip = int(settle_secs * fs)  # discard PLL/AGC transient after retuning
        out = np.empty(n + skip, np.complex64)
        st = dev.setupStream(S.SOAPY_SDR_RX, S.SOAPY_SDR_CF32, [ch])
        dev.activateStream(st)
        try:
            got, overflows, buf = 0, 0, np.empty(1 << 16, np.complex64)
            while got < n + skip:
                r = dev.readStream(st, [buf], min(len(buf), n + skip - got), timeoutUs=1_000_000)
                if r.ret == S.SOAPY_SDR_OVERFLOW:  # host was too slow; samples were dropped
                    overflows += 1
                    if overflows > max_overflows:
                        raise RuntimeError(f"too many RX overflows ({overflows})")
                    continue
                if r.ret < 0:
                    raise RuntimeError(f"readStream error {r.ret}")
                out[got:got + r.ret] = buf[:r.ret]
                got += r.ret
        finally:
            dev.deactivateStream(st)
            dev.closeStream(st)
        return Capture(out[skip:], center_hz, fs)
