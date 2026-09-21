"""The agent loop: user question -> model -> tool calls -> SDR -> facts -> model -> answer."""
import json
import re
import time
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
- If a tool returns an error about wrong arguments, read it, fix them and call again. If the error \
says the SDR is disconnected or the capture failed, tell the user plainly and do not retry.
- Report what the tool found in plain words: what kind of signal, where, how strong, and how sure \
(confidence). If the verdict is empty, say the air is empty (only noise). Mention it when \
confidence is low, and remember that narrow low-confidence carriers can be the SDR's own spurs.
- Signal labels come from simple rules, not decoding. Say "looks like WiFi", not "this is a \
network named X". You cannot read packet contents or identify devices.
- Be brief: at most three short sentences, plain text, no lists or headings, unless the user \
asks for details. Lead with the answer (what is there and where), then one useful caveat if any. \
Answer in the language the user writes in."""

MAX_OLD_TOOL_CHARS = 300
_THINK = re.compile(r"<think>.*?</think>", re.S)
_THINK_TAIL = re.compile(r"^.*?</think>", re.S)  # chat templates may open the block inside the prompt


def strip_reasoning(text: str) -> str:
    """Remove a reasoning block from model output, including one whose opening tag is missing."""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = _THINK_TAIL.sub("", text)
    return text.strip()


class Agent:
    def __init__(self, llm, toolbox: Toolbox, on_event: Callable[[str, dict], None] | None = None,
                 max_steps: int = 8):
        self.llm, self.tools = llm, toolbox
        self.on_event = on_event or (lambda kind, data: None)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def abort_turn(self):
        """Drop the unfinished turn (after Ctrl+C) so the history stays valid for the next question."""
        last_user = max(i for i, m in enumerate(self.messages) if m["role"] == "user")
        del self.messages[last_user:]

    def _compact_history(self):
        """Old tool results were already turned into an answer; shrink them to save context."""
        for m in self.messages:
            if m["role"] == "tool" and len(m["content"]) > MAX_OLD_TOOL_CHARS:
                m["content"] = m["content"][:MAX_OLD_TOOL_CHARS] + " ...(truncated)"

    def ask(self, text: str) -> str:
        self._compact_history()
        self.messages.append({"role": "user", "content": text})
        done_calls = set()  # (tool, arguments) already run for this question
        for step in range(1, self.max_steps + 1):
            self.on_event("llm_start", {"step": step})
            t0 = time.monotonic()
            msg = self.llm.chat(self.messages, TOOLS)
            calls = msg.get("tool_calls") or []
            self.on_event("llm_end", {"step": step, "seconds": time.monotonic() - t0,
                                      "stats": dict(getattr(self.llm, "last_stats", None) or {}),
                                      "tool_calls": [c.get("function", {}).get("name", "") for c in calls]})
            entry = {"role": "assistant", "content": strip_reasoning(msg.get("content") or "")}
            if calls:
                for c in calls:
                    c.setdefault("id", "call_" + uuid.uuid4().hex[:8])
                entry["tool_calls"] = calls
            self.messages.append(entry)
            if not calls:
                return entry["content"]
            for c in calls:
                fn = c.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {})
                key = (name, json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args))
                repeat = key in done_calls
                done_calls.add(key)
                self.on_event("tool_call", {"name": name, "arguments": args, "repeat": repeat})
                t0 = time.monotonic()
                if repeat:
                    result = json.dumps({"note": "You already called this tool with the same arguments "
                                                 "for this question. Use the earlier result and answer now."})
                else:
                    result = self.tools.call(name, args)
                self.on_event("tool_result", {"name": name, "result": json.loads(result),
                                              "seconds": time.monotonic() - t0})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        return ("I could not finish: too many tool calls in a row without a final answer. "
                "Try a more specific question.")
