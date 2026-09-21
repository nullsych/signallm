"""Level-0 heuristics on synthetic captures. Run: python3 -m unittest discover -s tests -t ."""
import unittest

from signallm.facts import analyze
from tests import synth


def run(*components, center=synth.CENTER, seed=0, **kw):
    return analyze(synth.make(*components, center=center, seed=seed, **kw))["interpretation"]


def labels(interp):
    return [s["label"] for s in interp["signals"]]


class Empty(unittest.TestCase):
    def test_noise_only_is_empty_for_many_seeds(self):
        for seed in range(6):
            r = run(seed=seed)
            self.assertEqual(r["verdict"], "empty", f"seed {seed}: {r['summary']}")

    def test_tilted_and_flat_noise(self):
        for tilt in (0.0, 3.0, 6.0):
            self.assertEqual(run(tilt_db=tilt)["verdict"], "empty")


class WiFi(unittest.TestCase):
    def test_channel_6_light_and_busy(self):
        for duty in (0.05, 0.15, 0.6):
            for seed in range(3):
                r = run(synth.wifi(duty=duty), seed=seed)
                self.assertEqual(labels(r), ["wifi"], f"duty {duty} seed {seed}: {r['summary']}")
                self.assertEqual(r["signals"][0]["channel"], 6)
                self.assertGreaterEqual(r["signals"][0]["confidence"], 0.7)

    def test_partial_plateau_at_span_edge_is_still_wifi(self):
        r = run(synth.wifi(offset_hz=-6e6))
        self.assertEqual(labels(r), ["wifi"], r["summary"])

    def test_channel_1_seen_from_its_own_center(self):
        r = run(synth.wifi(), center=2412e6)
        self.assertEqual(r["signals"][0]["channel"], 1)

    def test_not_wifi_outside_ism_band(self):
        r = run(synth.wifi(), center=868e6)
        self.assertNotIn("wifi", labels(r))
        self.assertEqual(labels(r), ["wideband"])


class BLE(unittest.TestCase):
    def test_advertising_channels(self):
        for center in (2402e6, 2426e6, 2480e6):
            for seed in range(3):
                r = run(synth.ble_adv(0), center=center, seed=seed)
                self.assertEqual(labels(r), ["ble_advertising"], f"{center / 1e6} seed {seed}: {r['summary']}")

    def test_data_channel_is_only_ble_or_zigbee(self):
        r = run(synth.ble_adv(3e6))  # 2440 MHz
        self.assertEqual(labels(r), ["ble_or_zigbee"], r["summary"])

    def test_ble_is_not_wifi(self):
        self.assertNotIn("wifi", labels(run(synth.ble_adv(0), center=2426e6)))


class Microwave(unittest.TestCase):
    def test_50hz(self):
        for seed in range(3):
            r = run(synth.microwave(), seed=seed)
            self.assertEqual(labels(r), ["microwave_oven"], r["summary"])

    def test_60hz(self):
        r = run(synth.microwave(period_s=1 / 60))
        self.assertEqual(labels(r), ["microwave_oven"], r["summary"])

    def test_wifi_is_not_a_microwave(self):
        self.assertNotIn("microwave_oven", labels(run(synth.wifi(duty=0.5))))


class Narrow(unittest.TestCase):
    def test_cw_is_carrier_with_low_confidence(self):
        r = run(synth.carrier(2e6))
        self.assertEqual(labels(r), ["carrier"])
        self.assertLessEqual(r["signals"][0]["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
