# signallm

`signallm` is a local LLM agent with hands on an SDR: it tunes, scans, decodes, and narrates the spectrum.

## Status

For now a tool currently support `BladeRF 2.0 xA9` SDR. The development status is below:

| Step | Description | State |
|------|-------------|-------|
| 0 | Hardware: BladeRF 2.0 xA9 + SoapySDR | done |
| 1 | Spectrum -> JSON facts | done |
| 2 | Level-0 heuristics (WiFi / BLE / microwave / empty) | done, validated on synthetic signals only |
| 3 | LLM agent (tool calling, interactive chat) | done, not yet tried with a real model |
| 4 | Decoder registry (rtl_433, ...), run over recorded files | planned |
| 5 | CNN classifier for unknown signals | planned |

## What works now

Given a capture (live or from a file), `signallm.facts` prints a JSON document with:

- **Noise floor**: running median of the lower half of PSD values, so it follows the tilt of the
  filter passband and stays below wide signals.
- **Signals**: center frequency, bandwidth, peak level, SNR, flatness (peak minus mean in band).
  Edges of the band (analog filter roll-off) and the DC/LO-leakage zone are excluded.
- **Spur rejection**: two captures with different centers. A real signal stays at the same
  absolute frequency; a spur stays at the same offset from the center and disappears.
  Each signal gets `confirmed: true | false | absent` (absent = outside the second capture's span).
- **Burst statistics** per signal from a spectrogram: duty cycle, burst count, median/max burst
  duration, burst period, envelope periodicity (e.g. 50 Hz for a microwave oven).
- Optional **spectrogram PNG**.

### Example output

```json
{
  "center_hz": 2437000000.0,
  "span_hz": 16000000.0,
  "noise_floor_db": -102.6,
  "dual_capture": true,
  "signals": [
    {
      "center_hz": 2439999820.0, "bandwidth_hz": 14648.4, "snr_db": 11.6,
      "confirmed": true,
      "burst": {"duty_cycle": 0.003, "burst_count": 38, "burst_ms_median": 0.051}
    }
  ],
  "spurs_rejected": [{"center_hz": 2444505401.9, "snr_db": 8.5, "confirmed": false}]
}
```

## Requirements

- Python 3.10+, `numpy`, `scipy`, `matplotlib` (PNG only)
- For live capture: `python3-soapysdr`, `soapysdr0.8-module-bladerf`

### Level-0 heuristics

`facts.analyze()` also returns an `interpretation` block: a label, confidence (0-1) and
human-readable evidence for every signal, plus a capture-level verdict and one-line summary.

| Label | Rule (2.4 GHz ISM band only) |
|-------|------------------------------|
| `empty` (verdict) | no signal above the noise floor |
| `wifi` | flat plateau >= 8 MHz (or cut by the span edge), bursty; channel 1-13 estimated from the visible edges |
| `ble_advertising` | 0.8-2.6 MHz wide, sparse short bursts, centered on 2402 / 2426 / 2480 MHz |
| `ble_or_zigbee` | same shape on the 2 MHz BLE grid but not an advertising channel |
| `microwave_oven` | wide, on/off envelope with a 14-22 ms period (50/60 Hz mains), duty 30-70% |
| `carrier` / `narrowband_pulses` | <= 200 kHz wide; low confidence, may be an SDR spur |
| `wideband` / `narrowband_modulated` | generic labels outside 2.4 GHz |

The noise floor is the lower of a frequency-median estimate and a per-bin low quantile over time.
The time estimate is what makes bursty signals that fill the whole span (WiFi, microwave oven)
visible: the gaps between bursts show the real noise.

Example summary: `Around 2437 MHz (+/-8 MHz): wifi ch6 at 2437.0 MHz (0.90).`

### LLM agent

`python3 -m signallm` opens an interactive chat, in the spirit of `bladeRF-cli`. The model calls
tools, the tools drive the SDR, and the model answers from the returned facts; it never sees IQ.

```
signallm> what is on 2.4 GHz?
  [tool] record(freq_mhz=2437)
Looks like WiFi channel 6 (confidence 0.9), otherwise quiet.
```

| Tool | What it does |
|------|--------------|
| `describe()` | tunable range, per-capture span, what was measured last |
| `record(freq_mhz, bandwidth_mhz=20, secs=0.5)` | tune, capture twice (spur check), return signals with labels |
| `scan(start_mhz, stop_mhz)` | sweep in ~14 MHz steps (max 60 steps), list all signals found |

Arguments are validated against the limits reported by the device; errors come back as plain
text so the model can fix its call and retry. Frequencies are in MHz at the tool boundary because
models make fewer unit mistakes that way. Slash commands `/device`, `/record`, `/scan` run the tools
directly without an LLM, `/reset` clears the conversation.

**LLM backend.** Any OpenAI-compatible endpoint with tool calling. Ollama
(`localhost:11434`) and llama.cpp `llama-server` (`localhost:8080`) are auto-detected; otherwise pass
`--base-url` and `--model`. Suggested: `ollama pull qwen3:8b` (or `qwen3:4b` on a weak machine).
Ollama's default context window (4096 tokens) is too small for scan results and is truncated
silently; start it with `OLLAMA_CONTEXT_LENGTH=8192`.

**Without hardware:** `python3 -m signallm --replay tests/cap.bin --replay-center 2437e6`
serves a recorded file as the radio (the two-capture spur check is meaningless there).

## Usage

Live capture (two captures: center and center + 0.2 * fs, spur rejection on):

```sh
python3 -m signallm.facts --center 2437e6 --gain 60
```

From an interleaved int16 (sc16) file, e.g. saved with `bladeRF-cli`:

```sh
python3 -m signallm.facts --file tests/cap.bin --center 2437e6 --fs 20e6
python3 -m signallm.facts --file a.bin --center 2437e6 --file-b b.bin --center-b 2441e6
```

Options: `--fs` (default 20e6), `--secs` (default 1.0), `--gain` (dB; omit for AGC),
`--single` (skip the second capture), `--png out.png` (spectrogram).

### Known issue: garbage IQ from SoapySDR

If `/usr/local/lib/libbladeRF.so.2` (a custom build) shadows the system one, the apt SoapySDR
bladerf module reads corrupted samples (8-bit-looking values, even/odd samples asymmetric,
gain has no effect). `SoapyRadio` works around this by preloading the system libbladeRF before
creating the device (override the path with `SIGNALLM_LIBBLADERF`). `bladeRF-cli` is not affected.

The workaround only applies inside signallm. Other Soapy clients (GNU Radio, `SoapySDRUtil`,
third-party scripts) will still hit the problem. To fix it system-wide, either remove
`/usr/local/lib/libbladeRF*` and run `ldconfig`, or rebuild `SoapyBladeRF` against the custom
libbladeRF.

## Tests

```sh
python3 -m unittest discover -s tests -t .
```

Tests need no hardware and no LLM: synthetic 2.4 GHz captures from `tests/synth.py` (WiFi plateau,
BLE advertising, microwave oven, CW, noise), a synthetic radio for the tools, a scripted fake LLM
for the agent loop, and a fake HTTP backend for the client. Takes ~3 minutes.

## Layout

```
signallm/
  capture.py      IQ sources: SoapyRadio (live, lazy import) and load_sc16 (file)
  spectrum.py     Welch PSD, noise floor, peak grouping into signals
  spurs.py        dual-capture spur rejection
  spectrogram.py  STFT, burst statistics, PNG output
  heuristics.py   level-0 rules: signal facts -> labels, confidence, evidence
  tools.py        agent tool schemas, argument validation, tool implementations
  llm.py          OpenAI-compatible chat client, Ollama / llama-server detection
  agent.py        system prompt and the tool-calling loop
  chat.py         interactive terminal chat (python3 -m signallm)
  facts.py        orchestrator and CLI (python3 -m signallm.facts)
```

## Design notes

- The LLM never sees raw IQ: it will drive the SDR through tool calls (`tune`, `scan`, `record`, `run_decoder`) and answer from JSON facts.
- One process owns the device (BladeRF is single-client). Later steps record IQ to a file and
  run decoders over that file instead of giving them the device.
- Code comments are in English.

## Limitations

- Heuristics are validated only on synthetic signals plus one real lightly occupied capture, which
  has no WiFi. Real WiFi/BLE/microwave captures are needed to tune the thresholds.
- A signal that is on continuously and fills the whole span (no gaps) raises the noise estimate and
  can hide itself; a single capture cannot tell it from noise.
- A narrow signal inside a WiFi plateau is merged into the plateau and not reported separately.
- 5 GHz WiFi, Zigbee channel plan and other bands are not covered by the rules yet.
- The agent has not been tried with a real model yet; the system prompt and the tool descriptions will
  need tuning per model. Small models may call tools wrongly or ignore results.
