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


class DeviceError(RuntimeError):
    """The SDR is missing, busy or was unplugged. The message is written for the end user."""


NOT_FOUND_HELP = (
    "No BladeRF target connected. Check that it is plugged in (use a USB 3.0 port and a good cable) and that "
    "its LEDs are on. On Linux make sure your user may access it (udev rules / plugdev group); on "
    "Windows that the Nuand driver is installed. Run `bladeRF-cli -p` to see what the system sees.")
BINDINGS_HELP = (
    "The SoapySDR Python bindings are not installed. Linux: sudo apt install python3-soapysdr "
    "soapysdr0.8-module-bladerf. Windows: install PothosSDR (it includes SoapySDR and the bladeRF module).")


# BladeRF 2.0 limits, used to validate requests while no device is connected
DEFAULT_LIMITS = {"freq_hz": [70e6, 6e9], "sample_rate_hz": [520834.0, 61.44e6],
                  "bandwidth_hz": [200e3, 56e6]}


class SoapyRadio:
    """Sole owner of the device. Nothing touches the hardware until it is needed.

    Creating the object never fails and never opens anything. Every operation goes through
    _open(), which first checks that the device is on the bus and (re)opens it if necessary, so
    the SDR can be plugged in, unplugged and plugged in again while the program runs.
    SoapySDR is imported lazily so the rest of the code works without it.
    """

    def __init__(self, args: str = "driver=bladerf", channel: int = 0, soapy=None):
        _preload_system_libbladerf()
        self.args, self.ch = args, channel
        self._S, self._bindings_error, self.dev = soapy, None, None
        if soapy is None:
            try:
                import SoapySDR
                self._S = SoapySDR
            except ImportError:
                self._bindings_error = DeviceError(BINDINGS_HELP)

    def is_present(self) -> bool:
        """True if the device shows up on the bus right now (also works while we hold it open)."""
        if self._S is None:
            return False
        try:
            return len(self._S.Device.enumerate(self.args)) > 0
        except Exception:
            return False

    def status(self) -> str:
        if self._bindings_error:
            return "SoapySDR Python bindings are not installed"
        return "BladeRF connected" if self.is_present() else "BladeRF not connected"

    def _open(self):
        """Check the device is there before using it; open it on first use or after a re-plug."""
        if self._bindings_error:
            raise self._bindings_error
        if not self.is_present():
            self.dev = None  # a handle to an unplugged device is dead; forget it
            raise DeviceError(NOT_FOUND_HELP)
        if self.dev is None:
            try:
                self.dev = self._S.Device(self.args)
            except Exception as e:
                raise DeviceError(f"BladeRF found but could not be opened ({e}). Another program "
                                  "(bladeRF-cli, GNU Radio, another signallm) may be using it; close it "
                                  "and try again.") from None
        return self.dev

    def probe(self) -> dict:
        """Real limits if the device is already open, otherwise the BladeRF 2.0 defaults.
        Never opens the device itself: the hardware is only touched by real measurements."""
        try:
            dev = self.dev
            if dev is None:
                raise DeviceError("not open")
            r = dev.getFrequencyRange(self._S.SOAPY_SDR_RX, self.ch)
            sr = dev.getSampleRateRange(self._S.SOAPY_SDR_RX, self.ch)
            bw = dev.getBandwidthRange(self._S.SOAPY_SDR_RX, self.ch)
        except Exception:
            return {k: list(v) for k, v in DEFAULT_LIMITS.items()}
        return {
            "freq_hz": [min(x.minimum() for x in r), max(x.maximum() for x in r)],
            "sample_rate_hz": [min(x.minimum() for x in sr), max(x.maximum() for x in sr)],
            "bandwidth_hz": [min(x.minimum() for x in bw), max(x.maximum() for x in bw)],
        }

    def capture(self, center_hz: float, fs: float, secs: float, gain_db: float | None = None,
                settle_secs: float = 0.05, max_overflows: int = 20) -> Capture:
        self._open()
        try:
            return self._capture(center_hz, fs, secs, gain_db, settle_secs, max_overflows)
        except DeviceError:
            raise
        except Exception as e:
            if not self.is_present():
                self.dev = None
                raise DeviceError("The BladeRF was disconnected. Plug it back in and try again.") from None
            raise RuntimeError(str(e)) from e

    def _capture(self, center_hz, fs, secs, gain_db, settle_secs, max_overflows) -> Capture:
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


class ReplayRadio:
    """Radio backed by a recorded sc16 file: lets the agent run without hardware (demos, tests).

    Implements the same probe()/capture() interface as SoapyRadio. Only requests whose span
    overlaps the recorded band can be served.
    """

    def __init__(self, path: str, center_hz: float, fs: float):
        self._cap = load_sc16(path, center_hz, fs)

    def is_present(self) -> bool:
        return True

    def status(self) -> str:
        return "replay mode (recorded file)"

    def probe(self) -> dict:
        lo, hi = self._cap.center_hz - self._cap.fs * 0.4, self._cap.center_hz + self._cap.fs * 0.4
        return {"freq_hz": [lo, hi], "sample_rate_hz": [self._cap.fs, self._cap.fs],
                "bandwidth_hz": [self._cap.fs * 0.8, self._cap.fs * 0.8]}

    def capture(self, center_hz: float, fs: float, secs: float, gain_db: float | None = None,
                **_) -> Capture:
        lo, hi = self.probe()["freq_hz"]
        if not lo <= center_hz <= hi:
            raise RuntimeError(f"replay file only covers {lo / 1e6:.1f}-{hi / 1e6:.1f} MHz")
        return Capture(self._cap.iq[: int(secs * self._cap.fs)], self._cap.center_hz, self._cap.fs)
