"""Level-0 heuristics: rules over the step-1 facts, no models and no decoders.

Input is the dict produced by facts.analyze(); output is a label, a rough confidence and
human-readable evidence for every signal, plus a verdict for the whole capture. The rules
only know the 2.4 GHz ISM band; elsewhere signals get generic width-based labels.
"""
from dataclasses import dataclass, field

ISM_LO, ISM_HI = 2.400e9, 2.500e9
WIFI_OCCUPIED_HZ = 16.6e6           # 52 OFDM subcarriers x 312.5 kHz
BLE_ADV_HZ = (2402e6, 2426e6, 2480e6)
EDGE_TOL_HZ = 0.4e6                 # a signal this close to the span edge is cut off by it


def wifi_channel_center(n: int) -> float:
    return (2412 + 5 * (n - 1)) * 1e6


@dataclass
class Verdict:
    label: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def _extent(sig: dict, facts: dict):
    """Visible [lo, hi] of the signal and whether each side is cut off by the span edge."""
    lo = sig["center_hz"] - sig["bandwidth_hz"] / 2
    hi = sig["center_hz"] + sig["bandwidth_hz"] / 2
    span_lo = facts["center_hz"] - facts["span_hz"] / 2
    span_hi = facts["center_hz"] + facts["span_hz"] / 2
    return lo, hi, lo <= span_lo + EDGE_TOL_HZ, hi >= span_hi - EDGE_TOL_HZ


def _wifi_channel(sig: dict, facts: dict):
    """Estimate the channel from whichever edges of the plateau are actually visible."""
    lo, hi, cut_lo, cut_hi = _extent(sig, facts)
    span_lo = facts["center_hz"] - facts["span_hz"] / 2
    span_hi = facts["center_hz"] + facts["span_hz"] / 2
    half = WIFI_OCCUPIED_HZ / 2
    if cut_lo and cut_hi:
        # the plateau covers the whole span, so its center is within (half) of the span center
        c = facts["center_hz"]
    elif cut_lo:
        c = hi - half
    elif cut_hi:
        c = lo + half
    else:
        c = (lo + hi) / 2
    n = min(range(1, 14), key=lambda k: abs(wifi_channel_center(k) - c))
    err = abs(wifi_channel_center(n) - c)
    return (n if err <= 2.5e6 else None), c, (cut_lo or cut_hi), span_lo, span_hi


def _rule_microwave(sig, facts, b):
    period = b.get("envelope_period_ms")
    strength = b.get("envelope_period_strength", 0)
    duty = b.get("duty_cycle", 0)
    if sig["bandwidth_hz"] < 6e6 or period is None or strength < 0.5:
        return None
    # magnetron conducts on one half of each mains cycle: 20 ms (50 Hz) or 16.7 ms (60 Hz)
    if not (14 <= period <= 22 and 0.3 <= duty <= 0.7):
        return None
    ev = [f"wide ({sig['bandwidth_hz'] / 1e6:.1f} MHz)",
          f"on/off envelope with {period:.1f} ms period (mains cycle), strength {strength:.2f}",
          f"duty cycle {duty:.0%}"]
    return Verdict("microwave_oven", 0.85 if abs(duty - 0.5) < 0.15 else 0.7, ev)


def _rule_wifi(sig, facts, b):
    bw = sig["bandwidth_hz"]
    _, _, cut_lo, cut_hi = _extent(sig, facts)
    truncated = cut_lo or cut_hi
    if bw < 8e6 and not (truncated and bw >= 3e6):
        return None
    if sig["flatness_db"] > 6:
        return None
    duty = b.get("duty_cycle", 1.0)
    conf = 0.55 if bw >= 8e6 else 0.35
    ev = [f"wide flat plateau ({bw / 1e6:.1f} MHz visible, flatness {sig['flatness_db']:.1f} dB)"]
    if truncated:
        ev.append("plateau is cut off by the edge of the captured span")
    if sig["flatness_db"] < 3:
        conf += 0.1
    if duty < 0.9:
        conf += 0.15
        ev.append(f"bursty traffic, duty cycle {duty:.0%}")
    med = b.get("burst_ms_median")
    if med is not None and 0.05 <= med <= 3:
        conf += 0.1
        ev.append(f"packet-length bursts (median {med:.2f} ms)")
    ch, c, _, _, _ = _wifi_channel(sig, facts)
    extra = {"channel": ch, "estimated_center_hz": round(c)}
    ev.append(f"matches 2.4 GHz channel {ch}" if ch else "does not sit on a standard channel center")
    return Verdict("wifi", min(conf, 0.95), ev, extra)


def _rule_ble(sig, facts, b):
    bw, c = sig["bandwidth_hz"], sig["center_hz"]
    if not 0.8e6 <= bw <= 2.6e6:
        return None
    duty = b.get("duty_cycle", 1.0)
    med = b.get("burst_ms_median")
    if duty > 0.3:
        return None
    ev = [f"~{bw / 1e6:.1f} MHz wide", f"sparse short bursts, duty cycle {duty:.1%}"]
    on_adv = min(abs(c - f) for f in BLE_ADV_HZ) <= 1e6
    if on_adv:
        ev.append(f"centered on BLE advertising channel {min(BLE_ADV_HZ, key=lambda f: abs(c - f)) / 1e6:.0f} MHz")
        conf = 0.7
        if med is not None and med <= 0.6:
            conf += 0.15
            ev.append(f"burst length {med:.2f} ms fits an advertising packet")
        return Verdict("ble_advertising", conf, ev)
    off = (c - 2402e6) % 2e6
    if min(off, 2e6 - off) <= 0.7e6:
        ev.append("on the 2 MHz BLE channel grid (a Zigbee channel is also possible)")
        return Verdict("ble_or_zigbee", 0.45, ev)
    return Verdict("narrowband_bursts", 0.3, ev)


def _rule_narrow(sig, facts, b):
    if sig["bandwidth_hz"] > 0.2e6:
        return None
    confirmed = sig.get("confirmed")
    duty = b.get("duty_cycle", 1.0)
    ev = [f"very narrow ({sig['bandwidth_hz'] / 1e3:.0f} kHz)"]
    if b.get("burst_count", 0) and duty < 0.05:
        ev.append(f"short pulses, duty cycle {duty:.1%}")
        label = "narrowband_pulses"
    else:
        label = "carrier"
    if confirmed is None:
        ev.append("not checked against a second capture; may be an SDR spur")
    conf = 0.5 if confirmed else 0.3
    return Verdict(label, conf, ev)


def _generic(sig, facts, b):
    bw = sig["bandwidth_hz"]
    if bw >= 3e6:
        label = "wideband"
    elif bw > 0.2e6:
        label = "narrowband_modulated"
    else:
        label = "carrier"
    return Verdict(label, 0.3, [f"{bw / 1e6:.2f} MHz wide, SNR {sig['snr_db']:.0f} dB"])


def classify_signal(sig: dict, facts: dict) -> Verdict:
    b = sig.get("burst") or {}
    if ISM_LO <= sig["center_hz"] <= ISM_HI:
        for rule in (_rule_microwave, _rule_wifi, _rule_ble, _rule_narrow):
            v = rule(sig, facts, b)
            if v:
                return v
        return Verdict("unknown", 0.2, [f"{sig['bandwidth_hz'] / 1e6:.2f} MHz wide, "
                                         f"SNR {sig['snr_db']:.0f} dB, no rule matched"])
    return _generic(sig, facts, b)


def interpret(facts: dict) -> dict:
    """Adds a label to every signal and a capture-level verdict. Does not modify `facts`."""
    signals = []
    for s in facts["signals"]:
        v = classify_signal(s, facts)
        signals.append({
            "center_hz": s["center_hz"], "bandwidth_hz": s["bandwidth_hz"], "snr_db": s["snr_db"],
            "label": v.label, "confidence": round(v.confidence, 2), "evidence": v.evidence, **v.extra,
        })
    span = facts["span_hz"] / 1e6
    where = f"{facts['center_hz'] / 1e6:.0f} MHz (+/-{span / 2:.0f} MHz)"
    if not signals:
        return {"verdict": "empty", "signals": [],
                "summary": f"No signals above the noise floor ({facts['noise_floor_db']} dB) around {where}."}
    labels = sorted({s["label"] for s in signals})
    parts = []
    for s in sorted(signals, key=lambda s: -s["confidence"]):
        name = s["label"] + (f" ch{s['channel']}" if s.get("channel") else "")
        parts.append(f"{name} at {s['center_hz'] / 1e6:.1f} MHz ({s['confidence']:.2f})")
    return {"verdict": labels[0] if len(labels) == 1 else "mixed", "signals": signals,
            "summary": f"Around {where}: " + "; ".join(parts) + "."}
