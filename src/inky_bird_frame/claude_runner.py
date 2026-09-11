"""Claude-backed research and review with Gemini-drawn plate illustrations.

Drop-in alternative to :class:`~inky_bird_frame.codex_runner.CodexRunner` with the
same three-method interface. Responsibilities are split by capability:

- ``create_profile`` and ``review_plate`` call the Anthropic API (vision, web
  search restricted to the configured domains, structured JSON output).
- ``generate_plate`` asks Gemini's image model for a text-free illustration on
  plain old paper and then composites the factual labels deterministically with
  Pillow in one fixed font, so every plate carries the same hand and the lettering
  can never be misspelled or drift in style.

Credentials come from the environment, matching the scheduler's env-file model:
``ANTHROPIC_API_KEY`` for research and review, ``GEMINI_API_KEY`` for the
illustration. Both SDKs are optional dependencies (the ``claude`` extra) and are
imported lazily like the other optional integrations in this package.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Final

from .birds import BirdSpecies, TaxonContext
from .codex_runner import (
    PROFILE_SCHEMA,
    REVIEW_SCHEMA,
    _parse_review,
    parse_species_profile,
)
from .errors import GenerationError, MissingDependencyError
from .images import PORTRAIT_SIZE
from .models import QualityReview, ReferencePhoto, SpeciesProfileData
from .prompts import profile_prompt, reference_list

ANTHROPIC_MODEL: Final = "claude-sonnet-5"
# Nano Banana Pro. The base gemini-2.5-flash-image model tops out around 1K,
# which would be upscaled ~1.4x to the 1200x1600 canonical plate and soften the
# fine linework the e-paper panel depends on; the Pro model's 2K output covers
# the canonical size natively.
GEMINI_IMAGE_MODEL: Final = "gemini-3-pro-image-preview"
GEMINI_IMAGE_SIZE: Final = "2K"
GEMINI_ASPECT_RATIO: Final = "3:4"
GENERATOR_LABEL: Final = (
    f"Claude API ({ANTHROPIC_MODEL}) research and review / "
    f"Gemini ({GEMINI_IMAGE_MODEL}) illustration with composited labels"
)
MAX_OUTPUT_TOKENS: Final = 16000
# Fail fast on the image call. The google-genai default retries rate limits (429)
# with unbounded exponential backoff, which can wedge a single generation for
# 15+ minutes; the controller would rather see a prompt error and defer the
# species. A healthy 2K generation returns in well under a minute.
GEMINI_REQUEST_TIMEOUT_MS: Final = 150_000
GEMINI_RETRY_ATTEMPTS: Final = 2
GEMINI_RETRY_STATUS_CODES: Final = (429, 500, 502, 503, 504)
# Web-search content accumulating in context is the dominant per-plate cost, so
# cap each phase tightly. Research verifies a handful of facts, not exhaustive
# browsing; the review leans on the already-verified profile plus vision, so it
# needs only a light independent spot-check.
PROFILE_MAX_WEB_SEARCHES: Final = 4
REVIEW_MAX_WEB_SEARCHES: Final = 2
MAX_SERVER_TOOL_CONTINUATIONS: Final = 4
REFERENCE_MAX_EDGE: Final = 2048
MAX_PNG_ENCODED_BYTES: Final = 3_500_000

# The 13.3-inch Inky Impression is an E Ink Spectra 6 panel: six real pigments
# (black, white, red, yellow, green, blue); every other color is dithered.
SPECTRA6_PALETTE: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
)
# Pure black is a native panel pigment and renders crisply; a warm brown ink
# would dither into black/red/yellow speckle around letterforms.
INK_COLOR: Final = (0, 0, 0)
# One fixed font on every plate: Special Elite (Apache-2.0), an old typewriter
# face chosen against handwriting and monospace candidates on a real plate. The
# bundled default can be overridden with the INKY_BIRD_LABEL_FONT environment
# variable (an absolute .ttf path) so faces can be compared without a code change.
LABEL_FONT_ENV: Final = "INKY_BIRD_LABEL_FONT"
BUNDLED_LABEL_FONT: Final = (
    Path(__file__).resolve().parent / "assets" / "fonts" / "SpecialElite-Regular.ttf"
)
_FALLBACK_FONTS: Final[tuple[str, ...]] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
)


def label_font_path() -> Path | None:
    """Resolve the label font: env override, then the bundled face, then fallbacks."""
    override = os.environ.get(LABEL_FONT_ENV)
    if override and Path(override).is_file():
        return Path(override)
    if BUNDLED_LABEL_FONT.is_file():
        return BUNDLED_LABEL_FONT
    for candidate in _FALLBACK_FONTS:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def illustration_prompt(
    species: BirdSpecies,
    profile: SpeciesProfileData,
    references: list[ReferencePhoto],
    correction_findings: tuple[str, ...] = (),
) -> str:
    measurements = profile["measurements"]
    field_marks = "\n".join(f"  - {mark}" for mark in profile["field_marks"])
    palette = ", ".join(profile["palette"])
    correction = ""
    if correction_findings:
        issues = "\n".join(f"- {finding}" for finding in correction_findings)
        correction = f"""
Correction required after an independent review of the previous attempt:
{issues}

Create a new illustration that corrects every visual issue above. Lettering is handled outside
this illustration; never add text to fix anything. Do not copy or lightly edit the previous attempt.
"""
    return f"""Illustrate one scientific field-journal plate for the species below.

Species identity:
- Common name: "{species.common_name}"
- Scientific name: "{species.scientific_name}"
- Family: "{profile["family"]}"

Species-specific field notes:
- Length: "{measurements["length"]}"
- Wingspan: "{measurements["wingspan"]}"
- Weight: "{measurements["weight"]}"
- Habitat: {profile["habitat"]}
- Behavior: {profile["behavior"]}
- Field marks:
{field_marks}
- Plumage palette: {palette}

Reference images, in attachment order:
{reference_list(references)}

Treat every attached image as a species-accuracy reference. Synthesize the consistent anatomy,
proportions, posture, plumage pattern, and colors across them. Do not reproduce any photograph's
background, pose, crop, or composition.
{correction}
Style and composition:
- Portrait 3:4 page. The background is one sheet of old paper: a warm, slightly yellowed cream
  with a faint, even pebbled or stippled grain, filling the image edge to edge. It is the only
  background. Render it perfectly flat and straight-on: no marks, stains, foxing, or creases; no
  book or notebook; no binding, spiral, spine, or gutter; no page edges, curl, or shadow; no desk
  or surface behind it; no border, inset panel, or pasted card anywhere; and no ruled lines,
  column rules, folds, or creases drawn across the page. Keep the ageing subtle so the paper
  never competes with the drawing.
- Fine graphite and confident ink linework with restrained transparent watercolor.
- Bold, crisp, high-contrast lines and flat watercolor washes that survive a six-color e-paper
  panel; avoid soft gradients, airbrushed shading, and low-contrast detail.
- One full-body bird, large, filling the middle and right of the page in a natural perched
  posture; it may reach down toward the bottom band but must not enter the left third above it.
- Along the bottom quarter of the page, as one row from left to right spanning the full width:
  one wing-pattern study, one bill/head study, and one row of unlabeled color swatches. Exactly
  those three elements, kept inside the bottom quarter; no other figures, studies, or flight
  poses anywhere. They sit directly on the bare paper: no boxes, cells, frames, table lines,
  divider lines, or underlines around, between, or beneath them.
- Fill the page: no large empty regions apart from the reserved label area described below.
- No ruler, scale bar, tick marks, or measurement marks of any kind, anywhere on the page.
- It should look like a carefully scanned scientific field-journal page, not Audubon, not a
  decorative poster, not a collage, and not photorealistic.
- Quiet margins. No scenery, map, location, coordinates, date, logo, or watermark.
- Exactly one bird, one head, one beak, two wings, two legs, and one tail. Feet must be plausible.

Typography is composited separately by software. Do not render any letters, numerals, words,
labels, captions, or handwriting anywhere on the page. Keep the left third of the page above
the bottom quarter, and the top margin band (roughly the top eighth), as quiet, blank paper: no
bird, no studies, no swatches, no wash, and no stray marks there, so the labels composited
afterward sit on bare paper and never touch the artwork.
"""


def _label_font(size: int) -> Any:
    from PIL import ImageFont

    path = label_font_path()
    if path is not None:
        return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _wrapped_lines(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def composite_plate_labels(image: Any, profile: SpeciesProfileData) -> None:
    """Draw the factual labels onto a text-free illustration, in place.

    The illustration prompt reserves the left third and top margin for these
    labels. Rendering them from the validated profile in one fixed font keeps
    every plate consistent and every name and measurement spelled exactly.
    """
    try:
        from PIL import ImageDraw
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Pillow is required to composite plate labels") from exc

    draw = ImageDraw.Draw(image)
    width, height = image.size
    margin = max(width // 20, 24)
    column_width = width * 3 // 10
    title_size = max(width // 18, 16)
    subtitle_size = max(width // 28, 12)
    title_font = _label_font(title_size)
    subtitle_font = _label_font(subtitle_size)
    # The illustration keeps its studies in the lower-right, so the label block
    # must end above the bottom 30% of the page. Shrink the body face until the
    # measurements and field marks fit; never spill into the studies' band.
    label_floor = height * 7 // 10
    body_size = max(width // 44, 10)
    minimum_body = max(width // 70, 8)
    # Plates must read as one set, so the measurement line limits may only pull
    # the face down a little (qualifiers are trimmed first); the bottom-band rule
    # may go further, since a wall of field marks is rarer than a long qualifier.
    consistent_body = max(int(body_size * 0.9), minimum_body)
    while True:
        body_font = _label_font(body_size)
        block_top = (
            margin + int(title_size * 1.25) + int(subtitle_size * 1.3) + int(body_size * 2.4)
        )
        lines = _label_lines(draw, profile, body_font, body_size, column_width)
        block_bottom = block_top + len(lines) * int(body_size * 1.5)
        fits_floor = block_bottom <= label_floor
        fits_lines = _measurements_fit(draw, profile, body_font, column_width)
        if fits_floor and (fits_lines or body_size <= consistent_body):
            break
        if body_size <= minimum_body:
            break
        body_size -= 1
    # At the minimum face, drop trailing lines rather than cross into the band.
    line_height = int(body_size * 1.5)
    room = max((label_floor - block_top) // line_height, 0)
    lines = lines[:room]

    y = margin
    draw.text((margin, y), profile["common_name"], font=title_font, fill=INK_COLOR)
    y += int(title_size * 1.25)
    draw.text((margin, y), profile["scientific_name"], font=subtitle_font, fill=INK_COLOR)
    y += int(subtitle_size * 1.3)
    draw.text((margin, y), f"Family {profile['family']}", font=body_font, fill=INK_COLOR)
    y += int(body_size * 2.4)

    for line in lines:
        if line:
            draw.text((margin, y), line, font=body_font, fill=INK_COLOR)
        y += int(body_size * 1.5)


# Measurement qualifiers are abbreviated so they read as tidy field notes rather
# than wrapping into orphaned fragments ("...in)" alone on a line).
_MEASUREMENT_ABBREVIATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("standard references", "standard refs"),
    ("references", "refs"),
    ("approximately", "approx."),
    ("approximate", "approx."),
)


def _measurement_specs(profile: SpeciesProfileData) -> list[tuple[str, str, int]]:
    """(label, value, maximum lines): length and weight sit on one line, wingspan may take two."""
    measurements = profile["measurements"]
    return [
        ("Length", measurements["length"], 1),
        ("Wingspan", measurements["wingspan"], 2),
        ("Weight", measurements["weight"], 1),
    ]


_TRAILING_FLUFF = re.compile(
    r"[,;]?\s*\(?(approx\.?|approximate(ly)?|est\.?|estimated)\)?\s*$", re.IGNORECASE
)


def _measurement_text(label: str, value: str) -> str:
    for long, short in _MEASUREMENT_ABBREVIATIONS:
        value = value.replace(long, short)
    # A trailing "approx." adds nothing and tends to dangle alone; drop it.
    value = _TRAILING_FLUFF.sub("", value).strip()
    return f"{label}: {value}"


def _dangles(wrapped: list[str]) -> bool:
    """True when a multi-line wrap ends in a lone fragment ("in)", "g", "refs")."""
    if len(wrapped) < 2:
        return False
    last = wrapped[-1].strip()
    words = last.split()
    return len(words) == 1 and (len(words[0]) < 5 or not words[0][0].isalpha())


def _trim_candidates(value: str) -> list[str]:
    """Progressively shorter readings of a measurement value, leading clause first."""
    candidates = [value]
    for separator in ("; ", " (", ", ", " - ", " \u2013 "):
        head = value.split(separator, 1)[0].strip()
        if head and head != value and head not in candidates:
            candidates.append(head)
    words = candidates[-1].split()
    while len(words) > 2:
        words = words[:-1]
        candidate = " ".join(words)
        if candidate.count("(") == candidate.count(")"):  # never cut inside a parenthetical
            candidates.append(candidate)
    return [c for c in candidates if c.count("(") == c.count(")")]


def _fitted_measurement(
    draw: Any, label: str, value: str, limit: int, body_font: Any, column_width: int
) -> list[str]:
    """Wrap a measurement within its line limit without a dangling fragment.

    Among the readings of the value (full, then progressively trimmed), take the
    longest one that fits the limit and does not end in a lone fragment such as
    "in)" or "g". Trailing "approx." is dropped before any of this.
    """
    best: list[str] | None = None
    best_length = -1
    for candidate in _trim_candidates(value):
        text = _measurement_text(label, candidate)
        plain = _wrapped_lines(draw, text, body_font, column_width)
        options = [plain]
        # If the plain wrap would cut a parenthetical, fall back to breaking before
        # it, so "(28-35 in)" stays whole on the second line.
        if (len(plain) > limit or _dangles(plain)) and " (" in text and limit >= 2:
            head, paren = text.split(" (", 1)
            paren = "(" + paren
            if (
                draw.textlength(head, font=body_font) <= column_width
                and draw.textlength(paren, font=body_font) <= column_width
            ):
                options.append([head, paren])
        for wrapped in options:
            if len(wrapped) > limit or _dangles(wrapped):
                continue
            if len(candidate) > best_length:
                best, best_length = wrapped, len(candidate)
            break
    if best is not None:
        return best
    return _wrapped_lines(draw, _measurement_text(label, value), body_font, column_width)


def _measurement_lines(
    draw: Any, profile: SpeciesProfileData, body_font: Any, column_width: int
) -> list[list[str]]:
    return [
        _fitted_measurement(draw, label, value, limit, body_font, column_width)
        for label, value, limit in _measurement_specs(profile)
    ]


def _measurements_fit(
    draw: Any, profile: SpeciesProfileData, body_font: Any, column_width: int
) -> bool:
    """True when every measurement respects its line limit at this face size."""
    return all(
        len(wrapped) <= limit
        for wrapped, (_, _, limit) in zip(
            _measurement_lines(draw, profile, body_font, column_width),
            _measurement_specs(profile),
            strict=True,
        )
    )


def _label_lines(
    draw: Any, profile: SpeciesProfileData, body_font: Any, body_size: int, column_width: int
) -> list[str]:
    """Wrap the measurements and field marks into the label column."""
    lines: list[str] = []
    for wrapped in _measurement_lines(draw, profile, body_font, column_width):
        lines.append(wrapped[0] if wrapped else "")
        lines.extend(f"   {extra}" for extra in wrapped[1:])
    lines.append("")
    for mark in profile["field_marks"]:
        wrapped = _wrapped_lines(draw, mark, body_font, column_width - body_size)
        if wrapped:
            lines.append(f"\u2022 {wrapped[0]}")
            lines.extend(f"   {extra}" for extra in wrapped[1:])
    return lines


def spectra_panel_preview(source_path: Path, destination_path: Path) -> Path:
    """Quantize a plate to the Spectra 6 palette with Floyd-Steinberg dithering.

    The review pass otherwise only sees the full-color PNG, which can pass while
    looking washed out after the panel's six-color quantization.
    """
    try:
        from PIL import Image as PILImage
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Pillow is required to build panel previews") from exc
    flat = [channel for color in SPECTRA6_PALETTE for channel in color]
    palette_image = PILImage.new("P", (1, 1))
    palette_image.putpalette(flat + [0] * (768 - len(flat)))
    with PILImage.open(source_path) as source:
        preview = (
            source.convert("RGB")
            .quantize(palette=palette_image, dither=PILImage.Dither.FLOYDSTEINBERG)
            .convert("RGB")
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination_path)
    return destination_path


def review_prompt_with_preview(
    species: BirdSpecies,
    profile: SpeciesProfileData,
    references: list[ReferencePhoto],
    allowed_domains: tuple[str, ...],
) -> str:
    return f"""Review Image 1 as a candidate scientific field-journal plate for
{species.common_name} ({species.scientific_name}). Image 2 is the same candidate quantized to the
six-color e-paper palette (black, white, red, yellow, green, blue) used by the framed display;
judge contrast, wash rendering, and label legibility as displayed on Image 2, and fold that
e-paper legibility into composition_quality. Images 3 onward are licensed field-reference photos
of the same species.

Facts proposed by the research pass:
{json.dumps(profile, indent=2, sort_keys=True)}

Independently verify the species identity, measurements, and field marks against live source pages
and the attached references. Do not assume the proposed facts are correct. Restrict browsing to
these domains: {", ".join(allowed_domains)}. Do not rely on search snippets. Inspect the candidate
for correct plumage, proportions, bill, eye, wings, tail, legs, feet, and species field marks
against the attached field-reference photos. Compare every visible factual claim to the
independently verified facts. Confirm that no place name, ZIP code, coordinates, map, or
local-observation detail appears. Record every concrete issue and return at least two direct HTTPS
source URLs from distinct configured domains used for verification.

All lettering on the plate was composited by software from the verified profile in a fixed font,
so its spelling is exact by construction: do not lower text_accuracy for the typeface. Check only
that every label is legible on Image 2 and that no label overlaps the bird, the studies, or the
swatches.

House style is part of composition_quality. The plate must be one flat sheet of plain old paper
carrying only the illustration and the composited labels. Score composition_quality 3 or lower
and set passed=false if any of these appear: any letters, numerals, words, or handwriting painted
by the image model (anything not in the composited label column); any ruler, scale bar, tick
marks, or measurement marks; a visible notebook binding, spiral, spine, or gutter; a torn, curled,
or aged page edge; stains, foxing, or heavy ageing marks on the paper; a drawn vertical or
horizontal line, fold, or crease running through the label area; any box, cell, frame, table
line, or divider drawn around, between, or beneath the studies or swatches; or the bird or any
study placed inside a drawn border, inset panel, or pasted card. Also lower composition_quality for a
poorly filled page: a large empty region (for example a blank lower-left corner beneath a short
label column) or a bird drawn small in a sea of paper reads as unfinished.

Set passed=true only when all four scores are at least 4, location_free is true, the bird has
exactly one head, one beak, two wings, two legs, and one tail, and there are no material species or
text errors. Return only the requested JSON.

Reference provenance:
{reference_list(references)}
"""


def _reference_image(path: Path) -> Any:
    try:
        from PIL import Image as PILImage
        from PIL import ImageOps
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Pillow is required to load reference images") from exc
    if not path.is_file():
        raise GenerationError(f"Reference image not found: {path}")
    with PILImage.open(path) as source:
        image = ImageOps.exif_transpose(source.convert("RGB"))
    image.thumbnail((REFERENCE_MAX_EDGE, REFERENCE_MAX_EDGE))
    return image


def _encoded_reference(path: Path) -> dict[str, Any]:
    image = _reference_image(path)
    buffer = io.BytesIO()
    media_type = "image/jpeg"
    if path.suffix.lower() == ".png":
        # Plates and panel previews are PNG; JPEG re-encoding would smear the
        # linework and dither patterns the review is asked to judge.
        image.save(buffer, format="PNG")
        media_type = "image/png"
        if buffer.tell() > MAX_PNG_ENCODED_BYTES:
            buffer = io.BytesIO()
            media_type = "image/jpeg"
            image.save(buffer, format="JPEG", quality=90)
    else:
        image.save(buffer, format="JPEG", quality=90)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii"),
        },
    }


def _final_text(response: Any) -> str:
    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text" and block.text.strip():
            text = block.text
    return text


def _response_summary(response: Any) -> str:
    return (
        f"model={getattr(response, 'model', '?')} "
        f"stop_reason={getattr(response, 'stop_reason', '?')} "
        f"usage={getattr(response, 'usage', '?')}"
    )


def _inline_image_bytes(response: Any) -> bytes:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            data = getattr(getattr(part, "inline_data", None), "data", None)
            if isinstance(data, bytes) and data:
                return data
    raise GenerationError("Gemini returned no image data")


def _write_log(log_path: Path, prompt: str, result: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"PROMPT:\n{prompt}\n\nRESULT:\n{result}")


# Anthropic structured outputs reject JSON-Schema numeric, length, and pattern
# constraints (a 400). The shared schemas carry e.g. minimum/maximum on the
# review scores; the parse_* validators re-check those ranges, so stripping the
# schema-level bound before sending loses no enforcement.
_UNSUPPORTED_SCHEMA_KEYS: Final = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    }
)


def supported_schema(node: Any) -> Any:
    """Return a copy of a JSON schema without constraints Anthropic rejects."""
    if isinstance(node, dict):
        return {
            key: supported_schema(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [supported_schema(item) for item in node]
    return node


class ClaudeRunner:
    generator_label: str = GENERATOR_LABEL

    def __init__(self, workspace: Path, timeout_seconds: int = 1200) -> None:
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds
        self._anthropic_client: Any = None
        self._genai_client: Any = None

    def _anthropic(self) -> Any:
        if self._anthropic_client is None:
            try:
                import anthropic
            except ModuleNotFoundError as exc:
                raise MissingDependencyError(
                    "The anthropic package is required for Claude research and review; "
                    "install the claude extra"
                ) from exc
            self._anthropic_client = anthropic.Anthropic(timeout=float(self.timeout_seconds))
        return self._anthropic_client

    def _genai(self) -> Any:
        if self._genai_client is None:
            try:
                from google import genai
                from google.genai import types as genai_types
            except ModuleNotFoundError as exc:
                raise MissingDependencyError(
                    "The google-genai package is required for plate illustration; "
                    "install the claude extra"
                ) from exc
            self._genai_client = genai.Client(
                http_options=genai_types.HttpOptions(
                    timeout=GEMINI_REQUEST_TIMEOUT_MS,
                    retry_options=genai_types.HttpRetryOptions(
                        attempts=GEMINI_RETRY_ATTEMPTS,
                        max_delay=15.0,
                        http_status_codes=list(GEMINI_RETRY_STATUS_CODES),
                    ),
                )
            )
        return self._genai_client

    def _structured(
        self,
        prompt: str,
        schema: dict[str, object],
        image_paths: list[Path],
        output_path: Path,
        log_path: Path,
        *,
        allowed_domains: tuple[str, ...],
        max_searches: int,
    ) -> object:
        # Two phases: web search + vision as free-form research, then a
        # no-tools call that structures those findings against the schema.
        # Forcing output_config.format on the same request as server-side
        # web_search is unstable -- the model cannot always reconcile "call
        # tools" with "emit only JSON" and spirals until it hits max_tokens.
        research = self._research(
            prompt,
            image_paths,
            log_path,
            allowed_domains=allowed_domains,
            max_searches=max_searches,
        )
        return self._structure(prompt, research, schema, output_path, log_path)

    def _research(
        self,
        prompt: str,
        image_paths: list[Path],
        log_path: Path,
        *,
        allowed_domains: tuple[str, ...],
        max_searches: int,
    ) -> str:
        client = self._anthropic()
        import anthropic  # cached module; _anthropic() guarded the import

        instruction = (
            f"{prompt}\n\nWork through this now. Use web_search as needed and study the attached "
            "images. Write your findings as thorough plain-text notes that cover every fact the "
            "downstream record requires. Do not emit JSON yet; a later step converts your notes."
        )
        content: list[Any] = [_encoded_reference(path) for path in image_paths]
        content.append({"type": "text", "text": instruction})
        params: dict[str, Any] = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "thinking": {"type": "adaptive"},
            "tools": [
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": max_searches,
                    "allowed_domains": list(allowed_domains),
                }
            ],
        }
        messages: list[Any] = [{"role": "user", "content": content}]
        research_log = log_path.with_name(f"{log_path.stem}-research{log_path.suffix}")
        try:
            response = client.messages.create(**params, messages=messages)
            for _ in range(MAX_SERVER_TOOL_CONTINUATIONS):
                if response.stop_reason != "pause_turn":
                    break
                messages = [messages[0], {"role": "assistant", "content": response.content}]
                response = client.messages.create(**params, messages=messages)
        except (anthropic.APIError, anthropic.APIConnectionError) as exc:
            _write_log(research_log, instruction, f"ERROR: {exc}")
            raise GenerationError(
                f"Claude research request failed: {exc}; see {research_log}"
            ) from (exc)
        text = _final_text(response)
        _write_log(research_log, instruction, f"{_response_summary(response)}\n\n{text}")
        if response.stop_reason == "refusal":
            raise GenerationError(f"Claude declined the research request; see {research_log}")
        if response.stop_reason == "pause_turn":
            raise GenerationError(f"Claude search did not finish; see {research_log}")
        if not text.strip():
            raise GenerationError(f"Claude research produced no notes; see {research_log}")
        return text

    def _structure(
        self,
        prompt: str,
        research: str,
        schema: dict[str, object],
        output_path: Path,
        log_path: Path,
    ) -> object:
        client = self._anthropic()
        import anthropic  # cached module; _anthropic() guarded the import

        # The original prompt carries the authoritative identity (taxon id,
        # names) and the field requirements; the notes carry the researched
        # facts. Phase 2 needs both -- notes alone omit anything the prompt
        # told the model to "keep as supplied."
        instruction = (
            f"{prompt}\n\nResearch notes gathered for this task:\n{research}\n\n"
            "Now return the single JSON object described above, matching the required schema. "
            "Populate every field from the identity given above and the researched facts; do not "
            "leave a field blank and do not invent facts absent from both."
        )
        params: dict[str, Any] = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            # Structuring is deterministic extraction from the notes; disable
            # thinking explicitly so it stays cheap on models (e.g. Sonnet 5)
            # that would otherwise run adaptive thinking when it is omitted.
            "thinking": {"type": "disabled"},
            "output_config": {
                "format": {"type": "json_schema", "schema": supported_schema(schema)}
            },
        }
        try:
            response = client.messages.create(
                **params, messages=[{"role": "user", "content": instruction}]
            )
        except (anthropic.APIError, anthropic.APIConnectionError) as exc:
            _write_log(log_path, instruction, f"ERROR: {exc}")
            raise GenerationError(
                f"Claude structuring request failed: {exc}; see {log_path}"
            ) from (exc)
        text = _final_text(response)
        _write_log(log_path, instruction, f"{_response_summary(response)}\n\n{text}")
        if response.stop_reason == "refusal":
            raise GenerationError(f"Claude declined to structure the record; see {log_path}")
        if response.stop_reason == "max_tokens":
            raise GenerationError(f"Claude structured output was truncated; see {log_path}")
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"Claude did not return valid structured output; see {log_path}"
            ) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(parsed, indent=2, sort_keys=True))
        return parsed

    def create_profile(
        self,
        species: BirdSpecies,
        context: TaxonContext,
        references: list[ReferencePhoto],
        reference_paths: list[Path],
        output_path: Path,
        log_path: Path,
        *,
        allowed_domains: tuple[str, ...],
    ) -> SpeciesProfileData:
        raw = self._structured(
            profile_prompt(species, context, references, allowed_domains),
            PROFILE_SCHEMA,
            reference_paths,
            output_path,
            log_path,
            allowed_domains=allowed_domains,
            max_searches=PROFILE_MAX_WEB_SEARCHES,
        )
        profile = parse_species_profile(raw, allowed_domains)
        if (
            profile["taxon_id"] != species.taxon_id
            or profile["common_name"] != species.common_name
            or profile["scientific_name"] != species.scientific_name
        ):
            raise GenerationError("Claude profile identity does not match the discovered taxon")
        return profile

    def generate_plate(
        self,
        species: BirdSpecies,
        profile: SpeciesProfileData,
        references: list[ReferencePhoto],
        reference_paths: list[Path],
        output_path: Path,
        log_path: Path,
        correction_findings: tuple[str, ...] = (),
    ) -> Path:
        try:
            from PIL import Image as PILImage
            from PIL import ImageOps
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Pillow is required to prepare generated plates") from exc
        client = self._genai()
        from google.genai import errors as genai_errors  # cached; _genai() guarded the import
        from google.genai import types as genai_types

        prompt = illustration_prompt(species, profile, references, correction_findings)
        contents: list[Any] = [_reference_image(path) for path in reference_paths]
        contents.append(prompt)
        try:
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=genai_types.ImageConfig(
                        aspect_ratio=GEMINI_ASPECT_RATIO,
                        image_size=GEMINI_IMAGE_SIZE,
                    ),
                ),
            )
        except genai_errors.APIError as exc:
            _write_log(log_path, prompt, f"ERROR: {exc}")
            raise GenerationError(f"Gemini request failed: {exc}; see {log_path}") from exc
        image_bytes = _inline_image_bytes(response)
        _write_log(
            log_path,
            prompt,
            f"model={GEMINI_IMAGE_MODEL} produced {len(image_bytes)} bytes",
        )
        with PILImage.open(io.BytesIO(image_bytes)) as source:
            # LANCZOS keeps the 2K linework crisp through the downscale; softened
            # lines dither into fuzz on the panel.
            plate = ImageOps.fit(source.convert("RGB"), PORTRAIT_SIZE, PILImage.Resampling.LANCZOS)
        # Retain the text-free illustration with the run logs (never in the
        # candidate tree, which the catalog publisher allowlists) so a font or
        # layout change can be re-composited without paying for a new render.
        raw_path = log_path.with_name(f"{log_path.stem}-raw.png")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        plate.save(raw_path, format="PNG")
        composite_plate_labels(plate, profile)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plate.save(output_path, format="PNG")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise GenerationError(f"Gemini did not create the requested plate: {output_path}")
        return output_path

    def review_plate(
        self,
        species: BirdSpecies,
        profile: SpeciesProfileData,
        references: list[ReferencePhoto],
        plate_path: Path,
        reference_paths: list[Path],
        output_path: Path,
        log_path: Path,
        *,
        allowed_domains: tuple[str, ...],
    ) -> QualityReview:
        # The preview lives beside the review log in the retained runs directory,
        # never in the candidate directory: approve_candidate copies that tree
        # into the catalog, whose publisher allowlists species files.
        preview_path = spectra_panel_preview(
            plate_path, log_path.with_name(f"{log_path.stem}-panel-preview.png")
        )
        raw = self._structured(
            review_prompt_with_preview(species, profile, references, allowed_domains),
            REVIEW_SCHEMA,
            [plate_path, preview_path, *reference_paths],
            output_path,
            log_path,
            allowed_domains=allowed_domains,
            max_searches=REVIEW_MAX_WEB_SEARCHES,
        )
        return _parse_review(raw, allowed_domains)
