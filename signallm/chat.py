"""Interactive terminal chat with the SDR agent (python3 -m signallm)."""
import argparse
import json
import sys

from .agent import Agent
from .capture import ReplayRadio, SoapyRadio
from .llm import LLMClient, LLMError, detect_backend
from .tools import Toolbox

HELP = """\
Ask anything about the air, e.g. "what is on 2.4 GHz?" or "scan 860-870 MHz".
Commands:
  /device                     show what the SDR can do
  /record <MHz> [bw_MHz]      capture one frequency and print the facts (no LLM)
  /scan <start> <stop>        sweep a range and print the facts (no LLM)
  /reset                      forget the conversation
  /help, /quit"""


def _show_event(kind: str, data: dict):
    if kind == "tool_call":
        args = data["arguments"]
        if isinstance(args, dict):
            args = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"  [tool] {data['name']}({args})", flush=True)
    elif kind == "tool_result" and "error" in data["result"]:
        print(f"  [tool error] {data['result']['error']}", flush=True)


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signallm", description="Chat with an SDR agent")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: auto-detect Ollama / llama-server)")
    ap.add_argument("--model", help="model name (default: auto-pick from Ollama)")
    ap.add_argument("--api-key")
    ap.add_argument("--gain", type=float, help="RX gain in dB (default: AGC)")
    ap.add_argument("--replay", help="use a recorded sc16 file instead of the hardware")
    ap.add_argument("--replay-center", type=float, default=2437e6, help="center of the replay file, Hz")
    ap.add_argument("--replay-fs", type=float, default=20e6)
    a = ap.parse_args(argv)

    try:
        import readline  # noqa: F401  (line editing and history where available)
    except ImportError:
        pass

    try:
        radio = ReplayRadio(a.replay, a.replay_center, a.replay_fs) if a.replay else SoapyRadio()
    except Exception as e:  # driver missing, no device, busy device
        sys.exit(f"cannot open the SDR: {e}\nUse --replay FILE to run without hardware.")
    tools = Toolbox(radio, a.gain)
    lo, hi = tools.limits["freq_hz"]
    print(f"SDR ready, {lo / 1e6:.0f}-{hi / 1e6:.0f} MHz." + ("  (replay mode)" if a.replay else ""))

    agent = None
    base_url, model = a.base_url, a.model
    if not base_url:
        found = detect_backend()
        if found:
            base_url, model = found[0], model or found[1]
    if base_url and model:
        agent = Agent(LLMClient(base_url, model, a.api_key), tools, _show_event)
        print(f"LLM: {model} at {base_url}")
    else:
        print("No LLM backend found. Install Ollama (https://ollama.com), run `ollama pull qwen3:8b`,\n"
              "or start llama-server. Meanwhile /record, /scan and /device work without it.")
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
        elif line.startswith("/"):
            _direct(tools, line)
        elif agent is None:
            print("no LLM available; use /record, /scan or /device")
        else:
            try:
                print(agent.ask(line) + "\n")
            except LLMError as e:
                print(f"LLM error: {e}\n")


if __name__ == "__main__":
    main()
