"""The agent loop: user question -> model -> tool calls -> SDR -> facts -> model -> answer."""
import json
import re
import uuid
from typing import Callable

from .tools import TOOLS, Toolbox

SYSTEM_PROMPT = """You are signallm, an assistant that tells people what is happening on the radio \
spectrum, using a BladeRF software-defined radio through tools.

Rules:
- You cannot see the air yourself. For any question about what is on a frequency or band, call a \
tool first (record for one frequency, scan for a range) and answer only from its result. Never \
guess or answer from memory about what is on the air right now.
- Frequencies are in MHz (2.4 GHz = 2400 MHz, 868 MHz = 868).
- If a tool returns an error, read it, fix the arguments and call again.
- Report what the tool found in plain words: what kind of signal, where, how strong, and how sure \
(confidence). If the verdict is empty, say the air is empty (only noise). Mention it when \
confidence is low, and remember that narrow low-confidence carriers can be the SDR's own spurs.
- Signal labels come from simple rules, not decoding. Say "looks like WiFi", not "this is a \
network named X". You cannot read packet contents or identify devices.
- Be concise. Answer in the language the user writes in."""

MAX_OLD_TOOL_CHARS = 300
_THINK = re.compile(r"<think>.*?</think>", re.S)


class Agent:
    def __init__(self, llm, toolbox: Toolbox, on_event: Callable[[str, dict], None] | None = None,
                 max_steps: int = 8):
        self.llm, self.tools = llm, toolbox
        self.on_event = on_event or (lambda kind, data: None)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _compact_history(self):
        """Old tool results were already turned into an answer; shrink them to save context."""
        for m in self.messages:
            if m["role"] == "tool" and len(m["content"]) > MAX_OLD_TOOL_CHARS:
                m["content"] = m["content"][:MAX_OLD_TOOL_CHARS] + " ...(truncated)"

    def ask(self, text: str) -> str:
        self._compact_history()
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_steps):
            msg = self.llm.chat(self.messages, TOOLS)
            calls = msg.get("tool_calls") or []
            entry = {"role": "assistant", "content": msg.get("content") or ""}
            if calls:
                for c in calls:
                    c.setdefault("id", "call_" + uuid.uuid4().hex[:8])
                entry["tool_calls"] = calls
            self.messages.append(entry)
            if not calls:
                return _THINK.sub("", entry["content"]).strip()
            for c in calls:
                fn = c.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {})
                self.on_event("tool_call", {"name": name, "arguments": args})
                result = self.tools.call(name, args)
                self.on_event("tool_result", {"name": name, "result": json.loads(result)})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        return ("I could not finish: too many tool calls in a row without a final answer. "
                "Try a more specific question.")
