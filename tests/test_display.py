from __future__ import annotations

import unittest
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from inky_bird_frame.display import detect_inky_display, show_on_inky, wait_for_panel_idle
from inky_bird_frame.images import HARDWARE_SIZE, PAPER_COLOR


class FakeDisplay:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image: Image.Image | None = None
        self.show_count = 0

    def set_image(self, image: object) -> None:
        if not isinstance(image, Image.Image):
            raise TypeError("expected a Pillow image")
        self.image = image

    def show(self) -> None:
        self.show_count += 1


class DisplayTests(unittest.TestCase):
    def test_auto_detection_rejects_unknown_geometry(self) -> None:
        module = SimpleNamespace(auto=lambda: FakeDisplay(600, 400))
        with (
            patch("inky_bird_frame.display.import_module", return_value=module),
            self.assertRaisesRegex(ValueError, "unsupported Inky display size"),
        ):
            detect_inky_display()

    def test_canonical_image_is_unchanged_for_13_inch_display(self) -> None:
        display = FakeDisplay(*HARDWARE_SIZE)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "display.png"
            Image.new("RGB", HARDWARE_SIZE, "navy").save(path)

            size = show_on_inky(path, display=display)

        self.assertEqual(size, HARDWARE_SIZE)
        self.assertIsNotNone(display.image)
        assert display.image is not None
        self.assertEqual(display.image.size, HARDWARE_SIZE)
        self.assertEqual(display.image.getpixel((0, 0)), (0, 0, 128))
        self.assertEqual(display.show_count, 1)

    def test_canonical_image_is_contained_without_cropping_for_7_inch_display(self) -> None:
        display = FakeDisplay(800, 480)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "display.png"
            Image.new("RGB", HARDWARE_SIZE, "black").save(path)

            size = show_on_inky(path, display=display)

        self.assertEqual(size, (800, 480))
        self.assertIsNotNone(display.image)
        assert display.image is not None
        self.assertEqual(display.image.size, (800, 480))
        self.assertEqual(display.image.getpixel((79, 240)), PAPER_COLOR)
        self.assertEqual(display.image.getpixel((80, 240)), (0, 0, 0))
        self.assertEqual(display.image.getpixel((719, 240)), (0, 0, 0))
        self.assertEqual(display.image.getpixel((720, 240)), PAPER_COLOR)
        self.assertEqual(display.show_count, 1)

    def test_rejects_noncanonical_image_for_7_inch_display(self) -> None:
        display = FakeDisplay(800, 480)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.png"
            Image.new("RGB", (640, 480), "black").save(path)

            with self.assertRaisesRegex(ValueError, "canonical size"):
                show_on_inky(path, display=display)

        self.assertEqual(display.show_count, 0)


if __name__ == "__main__":
    unittest.main()


class PanelSettleTests(unittest.TestCase):
    def _fake_time(self) -> tuple[list[float], Callable[[float], None], Callable[[], float]]:
        now = [0.0]
        return now, (lambda s: now.__setitem__(0, now[0] + s)), (lambda: now[0])

    def test_waits_until_the_busy_line_clears(self) -> None:
        states = iter([True, True, True, False])
        now, sleep, clock = self._fake_time()
        waited = wait_for_panel_idle(
            object(), timeout=60, is_busy=lambda: next(states), sleep=sleep, clock=clock
        )
        # Three busy readings: the first is consumed while waiting for BUSY to rise
        # (no sleep), then two polls of 0.5 s before it clears.
        self.assertAlmostEqual(waited, 1.0)

    def test_waits_for_busy_to_rise_before_trusting_idle(self) -> None:
        # Idle for the first second, busy for the next three, then clear.
        states = iter([False, False, True, True, True, False])
        now, sleep, clock = self._fake_time()
        waited = wait_for_panel_idle(
            object(), timeout=60, is_busy=lambda: next(states), sleep=sleep, clock=clock
        )
        # Two idle polls (1.0 s) until BUSY rises, then two busy polls (1.0 s) until it clears.
        self.assertAlmostEqual(waited, 2.0)

    def test_holds_a_full_refresh_when_busy_never_rises(self) -> None:
        from inky_bird_frame.display import PANEL_MINIMUM_HOLD_SECONDS

        now, sleep, clock = self._fake_time()
        waited = wait_for_panel_idle(
            object(), timeout=60, is_busy=lambda: False, sleep=sleep, clock=clock
        )
        self.assertGreaterEqual(waited, PANEL_MINIMUM_HOLD_SECONDS)

    def test_warns_and_stops_at_the_cap_when_busy_never_clears(self) -> None:
        now, sleep, clock = self._fake_time()
        with self.assertWarnsRegex(UserWarning, "still busy after 3s"):
            waited = wait_for_panel_idle(
                object(), timeout=3, is_busy=lambda: True, sleep=sleep, clock=clock
            )
        self.assertGreaterEqual(waited, 3)

    def test_returns_immediately_without_a_busy_line(self) -> None:
        self.assertEqual(wait_for_panel_idle(FakeDisplay(1600, 1200)), 0.0)
