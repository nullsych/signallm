"""Device presence handling, using a fake SoapySDR module."""
import contextlib
import io
import json
import sys
import types
import unittest
from unittest import mock

from signallm import chat
from signallm.capture import (BINDINGS_HELP, DEFAULT_LIMITS, NOT_FOUND_HELP, DeviceError, SoapyRadio)
from signallm.tools import Toolbox
from tests import synth


class FakeSoapy:
    """Just enough of SoapySDR for SoapyRadio: enumerate() and Device()."""
    SOAPY_SDR_RX = 1
    SOAPY_SDR_CF32 = "CF32"
    SOAPY_SDR_OVERFLOW = -4

    def __init__(self, present=True, open_error=None, fail_ops=None, unplug_on_fail=False):
        self.present, self.open_error, self.fail_ops = present, open_error, fail_ops
        self.unplug_on_fail = unplug_on_fail
        self.opened = 0
        outer = self

        class Device:
            @staticmethod
            def enumerate(args=""):
                return [{"driver": "bladerf"}] if outer.present else []

            def __init__(self, args=""):
                if outer.open_error:
                    raise RuntimeError(outer.open_error)
                outer.opened += 1

            def setSampleRate(self, *a):
                if outer.fail_ops:
                    if outer.unplug_on_fail:  # the cable comes out while we are talking to the device
                        outer.present = False
                    raise RuntimeError(outer.fail_ops)

            def readStream(self, st, bufs, n, timeoutUs=0):
                return types.SimpleNamespace(ret=n)  # pretend n samples arrived

            def __getattr__(self, name):  # every other call is a no-op
                return lambda *a, **k: None

        self.Device = Device


def try_capture(radio):
    return radio.capture(2437e6, 20e6, 0.2)


class Lazy(unittest.TestCase):
    def test_creating_the_radio_never_touches_the_device(self):
        fake = FakeSoapy(present=False)
        radio = SoapyRadio(soapy=fake)  # no exception although nothing is plugged in
        self.assertEqual(fake.opened, 0)
        self.assertEqual(radio.probe(), DEFAULT_LIMITS)
        self.assertEqual(radio.status(), "BladeRF not connected")

    def test_probe_does_not_open_a_present_device_either(self):
        fake = FakeSoapy()
        radio = SoapyRadio(soapy=fake)
        radio.probe()
        self.assertEqual(fake.opened, 0)

    def test_measurement_without_device_gives_the_not_connected_message(self):
        with self.assertRaises(DeviceError) as cm:
            try_capture(SoapyRadio(soapy=FakeSoapy(present=False)))
        self.assertEqual(str(cm.exception), NOT_FOUND_HELP)

    def test_device_is_checked_before_every_measurement(self):
        fake = FakeSoapy()
        radio = SoapyRadio(soapy=fake)
        try_capture(radio)              # works, device gets opened
        fake.present = False
        with self.assertRaises(DeviceError):
            try_capture(radio)          # unplugged between two questions
        fake.present = True
        try_capture(radio)              # plugged back in: works without restarting
        self.assertEqual(fake.opened, 2)  # reopened, the old handle was dropped

    def test_plugging_in_after_start_works(self):
        fake = FakeSoapy(present=False)
        radio = SoapyRadio(soapy=fake)
        with self.assertRaises(DeviceError):
            try_capture(radio)
        fake.present = True
        try_capture(radio)

    def test_found_but_busy(self):
        with self.assertRaises(DeviceError) as cm:
            try_capture(SoapyRadio(soapy=FakeSoapy(open_error="Resource busy")))
        self.assertIn("Resource busy", str(cm.exception))
        self.assertIn("Another program", str(cm.exception))

    def test_missing_python_bindings_are_reported_on_use_not_at_start(self):
        with mock.patch.dict(sys.modules, {"SoapySDR": None}):  # makes `import SoapySDR` fail
            radio = SoapyRadio()
            self.assertIn("bindings", radio.status())
            with self.assertRaises(DeviceError) as cm:
                try_capture(radio)
        self.assertEqual(str(cm.exception), BINDINGS_HELP)

    def test_unplugged_during_capture_is_reported_as_disconnected(self):
        fake = FakeSoapy()
        radio = SoapyRadio(soapy=fake)
        try_capture(radio)
        fake.fail_ops, fake.unplug_on_fail = "LIBUSB_ERROR_NO_DEVICE", True
        with self.assertRaisesRegex(DeviceError, "disconnected"):
            try_capture(radio)
        self.assertIsNone(radio.dev)  # the dead handle is dropped

    def test_other_failures_stay_plain_runtime_errors(self):
        radio = SoapyRadio(soapy=FakeSoapy(fail_ops="bad sample rate"))
        with self.assertRaises(RuntimeError) as cm:
            try_capture(radio)
        self.assertNotIsInstance(cm.exception, DeviceError)
        self.assertIn("bad sample rate", str(cm.exception))


class WithTools(unittest.TestCase):
    def test_tool_error_carries_the_message_to_the_model(self):
        tb = Toolbox(SoapyRadio(soapy=FakeSoapy(present=False)))
        for name, args in (("record", {"freq_mhz": 2437, "secs": 0.2}),
                           ("scan", {"start_mhz": 2400, "stop_mhz": 2440})):
            out = json.loads(tb.call(name, args))
            self.assertIn("not connected", out["error"].lower().replace("target connected", "not connected"), out)

    def test_argument_validation_still_works_without_a_device(self):
        tb = Toolbox(SoapyRadio(soapy=FakeSoapy(present=False)))
        self.assertIn("outside", json.loads(tb.call("record", {"freq_mhz": 99999}))["error"])

    def test_describe_reports_connection_state(self):
        fake = FakeSoapy(present=False)
        tb = Toolbox(SoapyRadio(soapy=fake))
        self.assertFalse(json.loads(tb.call("describe", {}))["connected"])
        fake.present = True
        self.assertTrue(json.loads(tb.call("describe", {}))["connected"])

    def test_synthetic_radio_counts_as_connected(self):
        self.assertTrue(json.loads(Toolbox(synth.SynthRadio()).call("describe", {}))["connected"])


class Cli(unittest.TestCase):
    def run_chat(self, radio, lines):
        out = io.StringIO()
        with mock.patch.object(chat, "SoapyRadio", return_value=radio), \
                mock.patch.object(chat, "detect_backend", return_value=None), \
                mock.patch("builtins.input", side_effect=lines + [EOFError]), \
                contextlib.redirect_stdout(out):
            chat.main([])
        return out.getvalue()

    def test_chat_starts_without_a_device_and_says_so(self):
        text = self.run_chat(SoapyRadio(soapy=FakeSoapy(present=False)), ["/quit"])
        self.assertIn("BladeRF not connected", text)
        self.assertIn("plug it in later", text)

    def test_hardware_command_reports_the_problem_but_chat_keeps_running(self):
        text = self.run_chat(SoapyRadio(soapy=FakeSoapy(present=False)), ["/record 2437", "/help", "/quit"])
        self.assertIn("error", text)
        self.assertIn(NOT_FOUND_HELP[:30], text)
        self.assertIn("/reset", text)  # /help ran after the failed /record

    def test_connected_device_is_reported(self):
        text = self.run_chat(SoapyRadio(soapy=FakeSoapy()), ["/quit"])
        self.assertIn("BladeRF connected", text)
        self.assertNotIn("plug it in later", text)

    def test_system_prompt_says_do_not_retry_hardware_errors(self):
        from signallm.agent import SYSTEM_PROMPT
        self.assertIn("disconnected", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
