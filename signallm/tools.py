"""Agent tools: JSON schemas plus the code behind them.

The LLM never sees IQ. Every tool returns a small JSON document of facts; problems are returned
as {"error": "..."} in plain words so the model can correct its arguments and retry.
Frequencies are in MHz at this boundary because models make far fewer unit mistakes with them.
"""
import json
import math

from .facts import analyze

MAX_FS = 20e6            # sample rate we use; USB 3 headroom and the tested configuration
USABLE = 0.8             # fraction of fs that is clean (see spectrum.USABLE)
MAX_SCAN_STEPS = 60
MAX_SIGNALS_IN_RESULT = 40

TOOLS = [
    {"type": "function", "function": {
        "name": "describe",
        "description": "Describe the SDR: tunable frequency range, maximum bandwidth per capture, "
                       "and what was measured last. Call this if unsure what the hardware can do.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "record",
        "description": "Tune the SDR to a frequency, capture the air for a moment, and return what "
                       "is there: signals with center, width, SNR, a label (wifi, ble_advertising, "
                       "microwave_oven, carrier, ...) and confidence. Use this to answer "
                       "'what is on frequency X'. Covers about 16 MHz around the frequency.",
        "parameters": {"type": "object", "properties": {
            "freq_mhz": {"type": "number", "description": "Center frequency in MHz, e.g. 2437 or 868.3"},
            "bandwidth_mhz": {"type": "number", "description": "Capture bandwidth in MHz, 1-20 (default 20)"},
            "secs": {"type": "number", "description": "Capture length in seconds, 0.2-3 (default 0.5). "
                                                      "Longer catches rarer bursts."}},
            "required": ["freq_mhz"]}}},
    {"type": "function", "function": {
        "name": "scan",
        "description": "Sweep a frequency range in steps and list every signal found. Use it to "
                       "find where activity is when the user gives a range or a band rather than a "
                       "single frequency. Each step takes about a second; at most 60 steps "
                       "(about 900 MHz).",
        "parameters": {"type": "object", "properties": {
            "start_mhz": {"type": "number"}, "stop_mhz": {"type": "number"}},
            "required": ["start_mhz", "stop_mhz"]}}},
]


class ToolError(Exception):
    """Bad arguments or a failed capture; the message is shown to the model."""


def _num(args: dict, key: str, default=None):
    v = args.get(key, default)
    if v is None:
        raise ToolError(f"missing required argument '{key}'")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(v)  # models sometimes send numbers as strings
        except (TypeError, ValueError):
            raise ToolError(f"argument '{key}' must be a number, got {v!r}") from None
    if not math.isfinite(v):
        raise ToolError(f"argument '{key}' must be a finite number")
    return float(v)


def _signal_view(s: dict, evidence: bool = True) -> dict:
    out = {"center_mhz": round(s["center_hz"] / 1e6, 3), "bandwidth_mhz": round(s["bandwidth_hz"] / 1e6, 3),
           "snr_db": round(s["snr_db"], 1), "label": s["label"], "confidence": s["confidence"]}
    if evidence:  # scans skip it: dozens of signals would overflow small context windows
        out["evidence"] = s["evidence"]
    if s.get("channel"):
        out["wifi_channel"] = s["channel"]
    return out


class Toolbox:
    def __init__(self, radio, gain_db: float | None = None):
        self.radio = radio
        self.gain_db = gain_db
        self.limits = radio.probe()
        self.last: dict | None = None

    # -- validation ---------------------------------------------------------------------
    def _check_freq(self, name: str, mhz: float) -> float:
        lo, hi = self.limits["freq_hz"]
        if not lo / 1e6 <= mhz <= hi / 1e6:
            raise ToolError(f"{name}={mhz} MHz is outside the tunable range "
                            f"{lo / 1e6:.0f}-{hi / 1e6:.0f} MHz")
        return mhz * 1e6

    def _fs_for(self, bandwidth_mhz: float) -> float:
        lo, hi = self.limits["sample_rate_hz"]
        if bandwidth_mhz < 1 or bandwidth_mhz > MAX_FS / 1e6:
            raise ToolError(f"bandwidth_mhz={bandwidth_mhz} is not supported; use 1-{MAX_FS / 1e6:.0f} MHz")
        return min(max(bandwidth_mhz * 1e6, lo), min(hi, MAX_FS))

    # -- tools ----------------------------------------------------------------------------
    def describe(self, args: dict) -> dict:
        lo, hi = self.limits["freq_hz"]
        return {"device": "BladeRF", "freq_range_mhz": [lo / 1e6, hi / 1e6],
                "max_bandwidth_mhz": MAX_FS / 1e6, "clean_span_per_capture_mhz": MAX_FS * USABLE / 1e6,
                "heuristics_cover": "2.4 GHz ISM band (WiFi, BLE advertising, microwave oven); "
                                    "elsewhere signals only get width-based labels",
                "last_measurement": self.last}

    def record(self, args: dict) -> dict:
        f = self._check_freq("freq_mhz", _num(args, "freq_mhz"))
        fs = self._fs_for(_num(args, "bandwidth_mhz", 20))
        secs = _num(args, "secs", 0.5)
        if not 0.2 <= secs <= 3:
            raise ToolError(f"secs={secs} is out of range; use 0.2-3")
        try:
            a = self.radio.capture(f, fs, secs, self.gain_db)
            # second capture at a shifted center to reject the SDR's own spurs
            b = self.radio.capture(f + fs * 0.2, fs, secs, self.gain_db)
            facts = analyze(a, b)
        except RuntimeError as e:
            raise ToolError(f"capture failed: {e}") from None
        interp = facts["interpretation"]
        res = {"center_mhz": f / 1e6, "span_mhz": facts["span_hz"] / 1e6, "seconds": secs,
               "noise_floor_db": facts["noise_floor_db"], "verdict": interp["verdict"],
               "summary": interp["summary"],
               "signals": [_signal_view(s) for s in interp["signals"]],
               "spurs_rejected": len(facts["spurs_rejected"]), "spur_checked": True}
        self.last = {"tool": "record", "center_mhz": res["center_mhz"], "verdict": res["verdict"]}
        return res

    def scan(self, args: dict) -> dict:
        start = self._check_freq("start_mhz", _num(args, "start_mhz"))
        stop = self._check_freq("stop_mhz", _num(args, "stop_mhz"))
        if stop <= start:
            raise ToolError("stop_mhz must be greater than start_mhz")
        fs = MAX_FS
        span = fs * USABLE
        step = span * 0.9  # small overlap so nothing falls between windows
        n = max(1, math.ceil((stop - start - span * 0.1) / step))
        if n > MAX_SCAN_STEPS:
            raise ToolError(f"range {start / 1e6:.0f}-{stop / 1e6:.0f} MHz needs {n} steps, the limit is "
                            f"{MAX_SCAN_STEPS} (about {MAX_SCAN_STEPS * step / 1e6:.0f} MHz). Narrow it.")
        signals, empty = [], []
        try:
            for i in range(n):
                c = start + span / 2 + i * step
                facts = analyze(self.radio.capture(c, fs, 0.25, self.gain_db))
                interp = facts["interpretation"]
                if not interp["signals"]:
                    empty.append([round((c - span / 2) / 1e6, 1), round((c + span / 2) / 1e6, 1)])
                signals += [_signal_view(s, evidence=False) for s in interp["signals"]]
        except RuntimeError as e:
            raise ToolError(f"capture failed: {e}") from None
        signals.sort(key=lambda s: s["center_mhz"])
        res = {"start_mhz": start / 1e6, "stop_mhz": stop / 1e6, "steps": n,
               "signals": signals[:MAX_SIGNALS_IN_RESULT], "empty_windows_mhz": empty,
               "note": "single capture per step, spurs are not cross-checked; a wide signal may appear "
                       "in two adjacent windows"}
        if len(signals) > MAX_SIGNALS_IN_RESULT:
            res["truncated"] = f"{len(signals)} signals found, showing the first {MAX_SIGNALS_IN_RESULT}"
        self.last = {"tool": "scan", "range_mhz": [res["start_mhz"], res["stop_mhz"]], "signals": len(signals)}
        return res

    def call(self, name: str, arguments) -> str:
        """Run a tool for the agent loop. Always returns a JSON string, never raises."""
        try:
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    raise ToolError(f"arguments are not valid JSON: {arguments[:100]!r}") from None
            if not isinstance(arguments, dict):
                raise ToolError("arguments must be a JSON object")
            fn = {"describe": self.describe, "record": self.record, "scan": self.scan}.get(name)
            if fn is None:
                raise ToolError(f"unknown tool '{name}'; available: describe, record, scan")
            return json.dumps(fn(arguments), ensure_ascii=False)
        except ToolError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
