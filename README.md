# signallm

`signallm` is a local LLM agent with hands on an SDR: it tunes, scans, decodes, and narrates the spectrum.

## Status

For now a tool currently support `BladeRF 2.0 xA9` SDR. The development status is below:

| Step | Description | State |
|------|-------------|-------|
| 0 | Hardware: BladeRF 2.0 xA9 + SoapySDR | done |
| 1 | Spectrum -> JSON facts | done |
| 2 | Level-0 heuristics (WiFi / BLE / empty) | planned |
| 3 | LLM agent (Ollama tool calling) | planned |
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

## Usage

Live capture (two captures: center and center + 0.2 * fs, spur rejection on):

```sh
python3 -m signallm.facts --center 2437e6 --gain 60
```

From an interleaved int16 (sc16) file, e.g. saved with `bladeRF-cli`:

```sh
python3 -m signallm.facts --file test/cap.bin --center 2437e6 --fs 20e6
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

## Layout

```
signallm/
  capture.py      IQ sources: SoapyRadio (live, lazy import) and load_sc16 (file)
  spectrum.py     Welch PSD, noise floor, peak grouping into signals
  spurs.py        dual-capture spur rejection
  spectrogram.py  STFT, burst statistics, PNG output
  facts.py        orchestrator and CLI (python3 -m signallm.facts)
```

## Design notes

- The LLM never sees raw IQ: it will drive the SDR through tool calls (`tune`, `scan`, `record`, `run_decoder`) and answer from JSON facts.
- One process owns the device (BladeRF is single-client). Later steps record IQ to a file and
  run decoders over that file instead of giving them the device.
- Code comments are in English.

## Limitations

- No automated tests yet; thresholds (6 dB over noise, 150 kHz merge gap) were tuned on a single
  lightly occupied 2.4 GHz capture.
- Signals are not classified yet: the output is raw facts (WiFi/BLE rules come in step 2).
