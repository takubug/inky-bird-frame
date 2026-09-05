"""Pimoroni Inky display adapter."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from .errors import MissingDependencyError
from .images import HARDWARE_SIZE, PAPER_COLOR, SUPPORTED_HARDWARE_SIZES

logger = logging.getLogger(__name__)

# inky's show() waits at most 40 s on the BUSY line and then returns with a
# warning as if the refresh had finished. A 13.3-inch Spectra 6 refresh already
# takes ~35 s warm and stretches well past 40 s in the cold, so on a battery
# display node that powers off shortly after show() returns, the cut lands
# mid-refresh and the previous plate stays on the glass. Keep polling BUSY
# until the panel is genuinely idle before reporting the update as sent.
PANEL_SETTLE_TIMEOUT_SECONDS = 300.0
PANEL_SETTLE_POLL_SECONDS = 0.5


class InkyDisplay(Protocol):
    width: int
    height: int

    def set_image(self, image: object) -> None: ...

    def show(self) -> None: ...


def detect_inky_display() -> InkyDisplay:
    """Auto-detect and validate a supported Pimoroni Inky panel."""

    try:
        module = import_module("inky.auto")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Pimoroni Inky is required for display output") from exc

    auto = cast(Callable[[], InkyDisplay], module.auto)
    display = auto()
    size = (display.width, display.height)
    if size not in SUPPORTED_HARDWARE_SIZES:
        supported = ", ".join(
            f"{width}x{height}" for width, height in sorted(SUPPORTED_HARDWARE_SIZES)
        )
        raise ValueError(f"unsupported Inky display size {size}; supported sizes: {supported}")
    return display


def _fit_canonical_image(image: object, display_size: tuple[int, int]) -> object:
    from PIL import Image, ImageOps

    if not isinstance(image, Image.Image):
        raise TypeError("display image must be a Pillow image")
    if image.size == display_size:
        return image
    if image.size != HARDWARE_SIZE:
        raise ValueError(
            f"image size {image.size} does not match canonical size {HARDWARE_SIZE} "
            f"or display size {display_size}"
        )

    fitted = ImageOps.contain(image, display_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", display_size, PAPER_COLOR)
    canvas.paste(
        fitted,
        (
            (display_size[0] - fitted.width) // 2,
            (display_size[1] - fitted.height) // 2,
        ),
    )
    return canvas


def show_on_inky(
    image_path: Path,
    *,
    display: InkyDisplay | None = None,
) -> tuple[int, int]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Pillow is required to load display images") from exc
    image = Image.open(image_path).convert("RGB")
    active_display = display if display is not None else detect_inky_display()
    expected_size = (active_display.width, active_display.height)
    if expected_size not in SUPPORTED_HARDWARE_SIZES:
        raise ValueError(f"unsupported Inky display size {expected_size}")
    active_display.set_image(_fit_canonical_image(image, expected_size))
    active_display.show()
    settled = wait_for_panel_idle(active_display)
    if settled:
        logger.info("panel settled %.1fs after show() returned", settled)
    return expected_size


def _panel_busy_probe(display: object) -> Callable[[], bool] | None:
    """Build a BUSY-line probe from the inky driver's own GPIO request, if present."""
    gpio = getattr(display, "_gpio", None)
    pin = getattr(display, "busy_pin", None)
    if gpio is None or pin is None:
        return None
    try:
        value_module = import_module("gpiod.line")
    except ModuleNotFoundError:
        return None
    active = value_module.Value.ACTIVE

    def is_busy() -> bool:
        return bool(gpio.get_value(pin) == active)

    return is_busy


def wait_for_panel_idle(
    display: object,
    *,
    timeout: float = PANEL_SETTLE_TIMEOUT_SECONDS,
    is_busy: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Block until the panel's BUSY line clears; return the seconds waited.

    Uses the same line and polarity as the driver's own busy wait, but keeps
    polling instead of giving up at 40 s. Returns 0.0 immediately when the
    display exposes no BUSY line (test doubles, unsupported drivers).
    """
    probe = is_busy if is_busy is not None else _panel_busy_probe(display)
    if probe is None:
        return 0.0
    start = clock()
    try:
        while probe():
            if clock() - start >= timeout:
                warnings.warn(
                    f"panel still busy after {timeout:.0f}s; powering off now may interrupt "
                    "the refresh",
                    stacklevel=2,
                )
                break
            sleep(PANEL_SETTLE_POLL_SECONDS)
    except Exception as exc:  # a GPIO read failure must not fail a completed render
        logger.warning("could not read the panel BUSY line: %s", exc)
    return clock() - start
