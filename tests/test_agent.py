"""Tools and the agent loop, without hardware and without a real LLM."""
import json
import unittest

from signallm.agent import Agent, strip_reasoning
from signallm.tools import Toolbox
from tests import synth


def wifi_ch6_scene():
    return synth.SynthRadio([(2437e6, lambda off: synth.wifi(offset_hz=off))])


class FakeLLM:
    """Replays scripted assistant messages and records what it was sent."""

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.script.pop(0)


def tool_call(name, args, raw=False):
    return {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args) if not raw else args}}]}


class Validation(unittest.TestCase):
    def setUp(self):
        self.tb = Toolbox(synth.SynthRadio())

    def err(self, name, args):
        out = json.loads(self.tb.call(name, args))
        self.assertIn("error", out, out)
        return out["error"]

    def test_frequency_out_of_range_says_the_range(self):
        self.assertIn("70-6000 MHz", self.err("record", {"freq_mhz": 99999}))
        self.assertIn("70-6000 MHz", self.err("record", {"freq_mhz": 10}))

    def test_unit_mistake_hz_instead_of_mhz(self):
        self.assertIn("outside", self.err("record", {"freq_mhz": 2.437e9}))

    def test_bad_bandwidth_and_secs(self):
        self.assertIn("bandwidth_mhz", self.err("record", {"freq_mhz": 2437, "bandwidth_mhz": 200}))
        self.assertIn("secs", self.err("record", {"freq_mhz": 2437, "secs": 60}))

    def test_missing_and_non_numeric_arguments(self):
        self.assertIn("freq_mhz", self.err("record", {}))
        self.assertIn("number", self.err("record", {"freq_mhz": "wifi"}))
        self.assertIn("finite", self.err("record", {"freq_mhz": float("nan")}))

    def test_bad_json_unknown_tool(self):
        self.assertIn("valid JSON", self.err("record", "{freq"))
        self.assertIn("JSON object", self.err("record", "[1]"))
        self.assertIn("unknown tool", self.err("tune", {}))

    def test_scan_ranges(self):
        self.assertIn("greater", self.err("scan", {"start_mhz": 2500, "stop_mhz": 2400}))
        self.assertIn("Narrow it", self.err("scan", {"start_mhz": 100, "stop_mhz": 5000}))

    def test_no_capture_on_bad_arguments(self):
        self.err("record", {"freq_mhz": 99999})
        self.assertEqual(self.tb.radio.calls, [])

    def test_string_numbers_are_accepted(self):
        out = json.loads(self.tb.call("record", {"freq_mhz": "2437", "secs": "0.2"}))
        self.assertNotIn("error", out)

    def test_describe(self):
        out = json.loads(self.tb.call("describe", {}))
        self.assertEqual(out["freq_range_mhz"], [70.0, 6000.0])


class Tools(unittest.TestCase):
    def test_record_reports_wifi_and_spur_check(self):
        tb = Toolbox(wifi_ch6_scene())
        out = json.loads(tb.call("record", {"freq_mhz": 2437, "secs": 0.2}))
        self.assertEqual(out["verdict"], "wifi", out["summary"])
        self.assertEqual(out["signals"][0]["wifi_channel"], 6)
        self.assertTrue(out["spur_checked"])
        self.assertEqual(len(tb.radio.calls), 2)  # two captures with different centers

    def test_record_empty_air(self):
        out = json.loads(Toolbox(synth.SynthRadio()).call("record", {"freq_mhz": 433.9, "secs": 0.2}))
        self.assertEqual(out["verdict"], "empty")
        self.assertEqual(out["signals"], [])

    def test_scan_finds_wifi_and_ble(self):
        radio = synth.SynthRadio([(2437e6, lambda off: synth.wifi(offset_hz=off)),
                                  (2402e6, lambda off: synth.ble_adv(off))])
        out = json.loads(Toolbox(radio).call("scan", {"start_mhz": 2400, "stop_mhz": 2440}))
        self.assertNotIn("error", out, out)
        labels = {s["label"] for s in out["signals"]}
        self.assertIn("wifi", labels)
        self.assertIn("ble_advertising", labels)
        self.assertEqual(out["steps"], 3)

    def test_capture_failure_becomes_a_tool_error(self):
        class Broken(synth.SynthRadio):
            def capture(self, *a, **k):
                raise RuntimeError("device busy")
        out = json.loads(Toolbox(Broken()).call("record", {"freq_mhz": 2437, "secs": 0.2}))
        self.assertIn("device busy", out["error"])

    def test_results_stay_small_for_the_model(self):
        out = wifi_ch6_scene()
        res = Toolbox(out).call("record", {"freq_mhz": 2437, "secs": 0.2})
        self.assertLess(len(res), 2500)


class Loop(unittest.TestCase):
    def agent(self, script, radio=None):
        events = []
        a = Agent(FakeLLM(script), Toolbox(radio or wifi_ch6_scene()), lambda k, d: events.append((k, d)))
        return a, events

    def test_tool_call_then_answer(self):
        a, ev = self.agent([tool_call("record", {"freq_mhz": 2437, "secs": 0.2}, raw=False),
                            {"content": "Looks like WiFi channel 6."}])
        self.assertEqual(a.ask("what is on 2.4 GHz?"), "Looks like WiFi channel 6.")
        kinds = [k for k, _ in ev]
        self.assertEqual(kinds, ["llm_start", "llm_end", "tool_call", "tool_result", "llm_start", "llm_end"])
        self.assertGreaterEqual(ev[1][1]["seconds"], 0)
        self.assertEqual(ev[1][1]["tool_calls"], ["record"])
        self.assertIn("seconds", ev[3][1])
        # the model saw the tool result on its second turn
        second = a.llm.seen[1]
        self.assertEqual(second[-1]["role"], "tool")
        self.assertIn("wifi", second[-1]["content"])
        self.assertEqual(second[-1]["tool_call_id"], "c1")

    def test_arguments_as_ollama_native_dict_and_as_string(self):
        for raw in (False, True):
            args = '{"freq_mhz": 2437, "secs": 0.2}' if raw else {"freq_mhz": 2437, "secs": 0.2}
            a, ev = self.agent([tool_call("record", args, raw=raw), {"content": "ok"}])
            a.ask("q")
            result = [d["result"] for k, d in ev if k == "tool_result"][0]
            self.assertNotIn("error", result, raw)

    def test_model_corrects_itself_after_an_error(self):
        a, ev = self.agent([tool_call("record", {"freq_mhz": 2.437e9}),
                            tool_call("record", {"freq_mhz": 2437, "secs": 0.2}),
                            {"content": "WiFi."}])
        self.assertEqual(a.ask("q"), "WiFi.")
        results = [d["result"] for k, d in ev if k == "tool_result"]
        self.assertIn("error", results[0])
        self.assertNotIn("error", results[1])
        # the error text reached the model
        self.assertIn("outside", a.llm.seen[1][-1]["content"])

    def test_repeated_identical_call_is_not_run_again(self):
        a, ev = self.agent([tool_call("describe", {}), tool_call("describe", {}), {"content": "done"}])
        self.assertEqual(a.ask("q"), "done")
        calls = [d for k, d in ev if k == "tool_call"]
        self.assertEqual([c["repeat"] for c in calls], [False, True])
        self.assertIn("already called", a.llm.seen[2][-1]["content"])

    def test_same_tool_with_other_arguments_is_allowed(self):
        a, ev = self.agent([tool_call("record", {"freq_mhz": 2437, "secs": 0.2}),
                            tool_call("record", {"freq_mhz": 2426, "secs": 0.2}), {"content": "ok"}])
        a.ask("q")
        self.assertEqual([d["repeat"] for k, d in ev if k == "tool_call"], [False, False])

    def test_abort_turn_restores_valid_history(self):
        a, _ = self.agent([tool_call("describe", {})])
        class Boom(FakeLLM):
            def chat(self, messages, tools):
                if len(self.seen) == 1:
                    raise KeyboardInterrupt
                return super().chat(messages, tools)
        a.llm = Boom([tool_call("describe", {}), {"content": "x"}])
        with self.assertRaises(KeyboardInterrupt):
            a.ask("first")
        a.abort_turn()
        self.assertEqual([m["role"] for m in a.messages], ["system"])

    def test_gives_up_after_max_steps(self):
        a, _ = self.agent([tool_call("describe", {"n": i}) for i in range(20)])
        a.max_steps = 3
        self.assertIn("too many tool calls", a.ask("q"))
        self.assertEqual(len(a.llm.seen), 3)

    def test_think_blocks_are_stripped(self):
        a, _ = self.agent([{"content": "<think>hmm</think>The air is empty."}])
        self.assertEqual(a.ask("q"), "The air is empty.")

    def test_reasoning_without_opening_tag_is_stripped(self):
        # some chat templates open the <think> block inside the prompt, so only </think> shows up
        a, _ = self.agent([{"content": "Okay, the user asks... let me think.\n</think>\n\nThe air is empty."}])
        self.assertEqual(a.ask("q"), "The air is empty.")
        self.assertNotIn("think", a.messages[-1]["content"])  # history stays clean too

    def test_strip_reasoning_cases(self):
        self.assertEqual(strip_reasoning("<think>a</think>b"), "b")
        self.assertEqual(strip_reasoning("plain"), "plain")
        self.assertEqual(strip_reasoning("x</think>y"), "y")
        self.assertEqual(strip_reasoning(""), "")

    def test_old_tool_results_are_compacted_on_the_next_question(self):
        a, _ = self.agent([tool_call("scan", {"start_mhz": 2400, "stop_mhz": 2440}), {"content": "a"},
                           {"content": "b"}], radio=synth.SynthRadio([(2437e6, lambda o: synth.wifi(offset_hz=o))]))
        a.ask("first")
        a.ask("second")
        tool_msgs = [m for m in a.llm.seen[-1] if m["role"] == "tool"]
        self.assertTrue(all(len(m["content"]) < 400 for m in tool_msgs))

    def test_reset_clears_history(self):
        a, _ = self.agent([{"content": "x"}])
        a.ask("q")
        a.reset()
        self.assertEqual(len(a.messages), 1)


if __name__ == "__main__":
    unittest.main()
