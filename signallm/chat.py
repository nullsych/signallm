"""Interactive terminal chat with the SDR agent (python3 -m signallm)."""
import argparse
import json
import sys
import threading
import time

from .agent import Agent
from .capture import ReplayRadio, SoapyRadio
from .llm import (DEFAULT_MODEL, LLMError, detect_backend, make_client, ollama_models,
                  pull_model)
from .tools import Toolbox

HELP = """\

Ask me anything about the air, e.g. "what is on 2.4 GHz?" or "scan 860-870 MHz",

Or use these commands:
  /device                     show what the SDR can do
  /record <MHz> [bw_MHz]      capture one frequency and print the facts (no LLM)
  /scan <start> <stop>        sweep a range and print the facts (no LLM)
  /reset                      forget the conversation
  /help, /quit"""


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class Spinner:
    """Animated 'something is happening' line with a running timer. No-op when output is not a terminal."""

    def __init__(self, stream=None, force: bool = False):
        self.stream = stream or sys.stdout
        self.enabled = force or self.stream.isatty()
        utf8 = "utf" in (getattr(self.stream, "encoding", "") or "").lower()
        self.frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if utf8 else "|/-\\"
        self._stop, self._thread, self._label, self._t0 = threading.Event(), None, "", 0.0

    def start(self, label: str):
        self.stop()
        if not self.enabled:
            return
        self._label, self._t0 = label, time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.wait(0.1):
            self.stream.write(f"\r  {self.frames[i % len(self.frames)]} {self._label} "
                              f"{time.monotonic() - self._t0:4.0f}s ")
            self.stream.flush()
            i += 1

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self.stream.write("\r" + " " * (len(self._label) + 16) + "\r")
            self.stream.flush()


def _llm_line(step: int, seconds: float, stats: dict, calls: list) -> str:
    parts = [f"llm #{step} {seconds:.1f}s"]
    if stats.get("prompt_tokens"):
        p = f"prompt {stats['prompt_tokens']} tok"
        if stats.get("prompt_s"):
            p += f"/{stats['prompt_s']:.1f}s"
        parts.append(p)
    if stats.get("gen_tokens"):
        g = f"gen {stats['gen_tokens']} tok"
        if stats.get("gen_s"):
            g += f"/{stats['gen_s']:.1f}s ({stats['gen_tokens'] / stats['gen_s']:.1f} tok/s)"
        parts.append(g)
    if stats.get("load_s", 0) > 0.5:
        parts.append(f"model load {stats['load_s']:.1f}s")
    if calls:
        parts.append("wants: " + ", ".join(calls))
    return " | ".join(parts)


def make_event_handler(spinner: Spinner, debug: bool = True):
    """Prints tool activity and (in debug mode) timestamps and timings; drives the spinner."""
    def show(kind: str, data: dict):
        if kind == "llm_start":
            spinner.start("thinking")
        elif kind == "llm_end":
            spinner.stop()
            if debug:
                print(f"  [{_ts()}] {_llm_line(data['step'], data['seconds'], data['stats'], data['tool_calls'])}",
                      flush=True)
        elif kind == "tool_call":
            args = data["arguments"]
            if isinstance(args, dict):
                args = ", ".join(f"{k}={v}" for k, v in args.items())
            note = "  (repeated call, not run again)" if data.get("repeat") else ""
            print(f"  [{_ts()}] tool {data['name']}({args}){note}", flush=True)
            if not data.get("repeat"):
                spinner.start(f"running {data['name']}")
        elif kind == "tool_result":
            spinner.stop()
            r = data["result"]
            if "error" in r:
                print(f"  [{_ts()}] tool error: {r['error']}", flush=True)
            elif debug:
                print(f"  [{_ts()}] tool {data['name']} done in {data['seconds']:.1f}s", flush=True)
    return show


def _direct(tools: Toolbox, line: str):
    """Slash commands that run a tool without the LLM."""
    parts = line.split()
    cmd, nums = parts[0], parts[1:]
    try:
        vals = [float(x) for x in nums]
    except ValueError:
        print("arguments must be numbers")
        return
    if cmd == "/device":
        out = tools.call("describe", {})
    elif cmd == "/record" and vals:
        out = tools.call("record", {"freq_mhz": vals[0], **({"bandwidth_mhz": vals[1]} if len(vals) > 1 else {})})
    elif cmd == "/scan" and len(vals) == 2:
        out = tools.call("scan", {"start_mhz": vals[0], "stop_mhz": vals[1]})
    else:
        print(HELP)
        return
    print(json.dumps(json.loads(out), ensure_ascii=False, indent=2))


def _has_model(installed: list[str], wanted: str) -> bool:
    return wanted in installed or f"{wanted}:latest" in installed


def _show_pull(status: str, done: int, total: int):
    pct = f" {done * 100 // total}%" if total else ""
    print(f"\r  {status[:40]:40s}{pct:5s}", end="", flush=True)


def _pick_backend(a):
    """Returns (kind, base_url, model) or None. Offers to download the model if it is missing."""
    if a.base_url:  # explicit endpoint: assume OpenAI-compatible
        return ("openai", a.base_url, a.model or "default")
    found = detect_backend()
    if not found:
        return None
    kind, base_url, model = found
    if kind != "ollama":
        return found
    wanted = a.model or model or DEFAULT_MODEL
    if _has_model(ollama_models(base_url), wanted):
        return kind, base_url, wanted
    try:
        ok = input(f"Model {wanted} is not installed. Download it now (a few GB)? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ok = "n"
    if ok not in ("", "y", "yes"):
        return None
    try:
        pull_model(base_url, wanted, _show_pull)
    except LLMError as e:
        print(f"\n{e}")
        return None
    print()
    return kind, base_url, wanted


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signallm", description="Chat with an SDR agent")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: auto-detect Ollama / llama-server)")
    ap.add_argument("--model", help="model name (default: auto-pick from Ollama)")
    ap.add_argument("--api-key")
    ap.add_argument("--no-debug", action="store_true", help="hide timestamps and timings")
    ap.add_argument("--think", action="store_true", help="let reasoning models think (much slower on a CPU)")
    ap.add_argument("--gain", type=float, help="RX gain in dB (default: AGC)")
    ap.add_argument("--replay", help="use a recorded sc16 file instead of the hardware")
    ap.add_argument("--replay-center", type=float, default=2437e6, help="center of the replay file, Hz")
    ap.add_argument("--replay-fs", type=float, default=20e6)
    a = ap.parse_args(argv)

    try:
        import readline  # noqa: F401  (line editing and history where available)
    except ImportError:
        pass

    radio = ReplayRadio(a.replay, a.replay_center, a.replay_fs) if a.replay else SoapyRadio()
    tools = Toolbox(radio, a.gain)
    lo, hi = tools.limits["freq_hz"]
    print(f"SDR: {radio.status()}. Range {lo / 1e6:.0f}-{hi / 1e6:.0f} MHz."
          + ("" if a.replay or radio.is_present() else "\n     (checked again before every measurement, "
                                                        "so you can plug it in later)"))

    agent = None
    spinner = Spinner()
    on_event = make_event_handler(spinner, debug=not a.no_debug)
    backend = _pick_backend(a)
    if backend:
        kind, base_url, model = backend
        agent = Agent(make_client(kind, base_url, model, a.api_key, think=a.think), tools, on_event)
        print(f"LLM: {model} via {kind} at {base_url}")
    else:
        print("No LLM available. Install Ollama (https://ollama.com) and restart, or start llama-server.\n"
              "Meanwhile /record, /scan and /device work without it.")
    print("Type /help for commands.\n")

    while True:
        try:
            line = input("signallm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/help":
            print(HELP)
        elif line == "/reset":
            agent and agent.reset()
            print("conversation cleared")
        elif line.split()[0] in ("/device", "/record", "/scan"):
            _direct(tools, line)
        elif line.startswith("/"):
            print(f"unknown command {line.split()[0]}; try /help")
        elif agent is None:
            print("no LLM available; use /record, /scan or /device")
        else:
            t0 = time.monotonic()
            if not a.no_debug:
                print(f"  [{_ts()}] question sent", flush=True)
            try:
                answer = agent.ask(line)
                spinner.stop()
                stamp = f"[{_ts()}] " if not a.no_debug else ""
                took = f"  ({time.monotonic() - t0:.1f}s total)" if not a.no_debug else ""
                print(f"\n{stamp}{answer}{took}\n")
            except LLMError as e:
                spinner.stop()
                print(f"[{_ts()}] LLM error: {e}\n")
            except KeyboardInterrupt:
                spinner.stop()
                agent.abort_turn()
                print(f"\n[{_ts()}] cancelled after {time.monotonic() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
