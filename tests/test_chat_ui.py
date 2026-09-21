"""Debug output and spinner of the terminal chat."""
import io
import re
import time
import unittest

from signallm.chat import Spinner, _llm_line, make_event_handler


class LlmLine(unittest.TestCase):
    def test_shows_timings_and_speed(self):
        line = _llm_line(2, 33.24, {"prompt_tokens": 1180, "prompt_s": 12.1, "gen_tokens": 41,
                                    "gen_s": 9.8, "load_s": 0.0}, ["record"])
        self.assertIn("llm #2 33.2s", line)
        self.assertIn("prompt 1180 tok/12.1s", line)
        self.assertIn("gen 41 tok/9.8s (4.2 tok/s)", line)
        self.assertIn("wants: record", line)
        self.assertNotIn("load", line)

    def test_model_load_only_shown_when_significant(self):
        self.assertIn("model load 8.0s", _llm_line(1, 9, {"load_s": 8.0}, []))

    def test_missing_stats_do_not_crash(self):
        self.assertEqual(_llm_line(1, 1.0, {}, []), "llm #1 1.0s")


class SpinnerTest(unittest.TestCase):
    def test_animates_and_cleans_up(self):
        out = io.StringIO()
        sp = Spinner(out, force=True)
        sp.start("thinking")
        time.sleep(0.35)
        sp.stop()
        text = out.getvalue()
        self.assertIn("thinking", text)
        self.assertGreaterEqual(len(re.findall(r"thinking", text)), 2)  # several frames were drawn
        self.assertTrue(text.endswith("\r"))  # line is cleared at the end

    def test_disabled_when_not_a_terminal(self):
        out = io.StringIO()
        sp = Spinner(out)  # StringIO is not a tty
        sp.start("thinking")
        sp.stop()
        self.assertEqual(out.getvalue(), "")

    def test_stop_without_start_is_safe(self):
        Spinner(io.StringIO(), force=True).stop()


class Handler(unittest.TestCase):
    def test_debug_lines_have_timestamps_and_spinner_toggles(self):
        out = io.StringIO()
        sp = Spinner(out)
        show = make_event_handler(sp, debug=True)
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            show("llm_start", {"step": 1})
            show("llm_end", {"step": 1, "seconds": 2.0, "stats": {}, "tool_calls": ["describe"]})
            show("tool_call", {"name": "describe", "arguments": {}})
            show("tool_result", {"name": "describe", "result": {}, "seconds": 0.1})
            show("tool_call", {"name": "describe", "arguments": {}, "repeat": True})
        lines = printed.getvalue().splitlines()
        self.assertTrue(all(re.match(r"  \[\d\d:\d\d:\d\d\] ", l) for l in lines), lines)
        self.assertIn("repeated call", lines[-1])

    def test_quiet_mode_hides_timings(self):
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            show = make_event_handler(Spinner(io.StringIO()), debug=False)
            show("llm_end", {"step": 1, "seconds": 2.0, "stats": {}, "tool_calls": []})
            show("tool_result", {"name": "x", "result": {}, "seconds": 0.1})
        self.assertEqual(printed.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
