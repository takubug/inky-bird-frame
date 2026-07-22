"""Claude-backed research and review with Gemini-drawn plate illustrations.

Drop-in alternative to :class:`~inky_bird_frame.codex_runner.CodexRunner` with the
same three-method interface. Responsibilities are split by capability:

- ``create_profile`` and ``review_plate`` call the Anthropic API (vision, web
  search restricted to the configured domains, structured JSON output).
- ``generate_plate`` asks Gemini's image model for a text-free illustration and
  then composites the factual labels deterministically with Pillow, so lettering
  accuracy never depends on a diffusion model.

Credentials come from the environment, matching the scheduler's env-file model:
``ANTHROPIC_API_KEY`` for research and review, ``GEMINI_API_KEY`` for the
illustration. Both SDKs are optional dependencies (the ``claude`` extra) and are
imported lazily like the other optional integrations in this package.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

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

if TYPE_CHECKING:
    from PIL import Image

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
MAX_WEB_SEARCHES: Final = 8
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
_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
)


def illustration_prompt(
    species: BirdSpecies,
    profile: SpeciesProfileData,
    references: list[ReferencePhoto],
    correction_findings: tuple[str, ...] = (),
) -> str:
    field_marks = "\n".join(f"  - {mark}" for mark in profile["field_marks"])
    palette = ", ".join(profile["palette"])
    correction = ""
    if correction_findings:
        issues = "\n".join(f"- {finding}" for finding in correction_findings)
        correction = f"""
Correction required after an independent review of the previous attempt:
{issues}

Create a new illustration that corrects every visual issue above. Lettering issues are handled
outside this illustration; never add text to fix them.
"""
    return f"""Illustrate one scientific field-journal plate for the species below.

Species identity:
- Common name: "{species.common_name}"
- Scientific name: "{species.scientific_name}"
- Family: "{profile["family"]}"

Species-specific guidance for accuracy:
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
- Portrait 3:4 page on warm aged cream naturalist-notebook paper.
- Fine graphite and confident ink linework with restrained transparent watercolor.
- Bold, crisp, high-contrast lines and flat watercolor washes that survive a six-color e-paper
  panel; avoid soft gradients, airbrushed shading, and low-contrast detail.
- One full-body bird, large and centered-right, in a natural perched posture.
- Bottom margin contains a small wing-pattern study, a bill/head study, and unlabeled color
  swatches.
- It should look like a carefully scanned scientific field-journal page, not Audubon, not a
  decorative poster, not a collage, and not photorealistic.
- Exactly one bird, one head, one beak, two wings, two legs, and one tail. Feet must be plausible.
- No scenery, map, logo, or watermark.

Typography is composited separately by software. Do not render any letters, numerals, words,
labels, captions, rulers, or handwriting anywhere on the page. Keep the left third of the page and
the top margin as quiet, empty paper so the labels can be placed there afterward.
"""


def _label_font(size: int) -> Any:
    from PIL import ImageFont

    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
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


def composite_plate_labels(image: Image.Image, profile: SpeciesProfileData) -> None:
    """Draw the factual labels onto an illustration, in place.

    The illustration prompt reserves the left third and top margin for these
    labels, and rendering them from the validated profile keeps every name and
    measurement spelled exactly as reviewed.
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
    body_size = max(width // 44, 10)
    title_font = _label_font(title_size)
    subtitle_font = _label_font(subtitle_size)
    body_font = _label_font(body_size)

    y = margin
    draw.text((margin, y), profile["common_name"], font=title_font, fill=INK_COLOR)
    y += int(title_size * 1.25)
    draw.text((margin, y), profile["scientific_name"], font=subtitle_font, fill=INK_COLOR)
    y += int(subtitle_size * 1.3)
    draw.text((margin, y), f"Family {profile['family']}", font=body_font, fill=INK_COLOR)
    y += int(body_size * 2.4)

    measurements = profile["measurements"]
    lines = [
        f"Length: {measurements['length']}",
        f"Wingspan: {measurements['wingspan']}",
        f"Weight: {measurements['weight']}",
        "",
    ]
    for mark in profile["field_marks"]:
        wrapped = _wrapped_lines(draw, mark, body_font, column_width - body_size)
        if wrapped:
            lines.append(f"• {wrapped[0]}")
            lines.extend(f"   {extra}" for extra in wrapped[1:])
    limit = height - 2 * margin
    for line in lines:
        if y > limit:
            break
        if line:
            draw.text((margin, y), line, font=body_font, fill=INK_COLOR)
        y += int(body_size * 1.5)


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
            except ModuleNotFoundError as exc:
                raise MissingDependencyError(
                    "The google-genai package is required for plate illustration; "
                    "install the claude extra"
                ) from exc
            self._genai_client = genai.Client()
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
    ) -> object:
        # Two phases: web search + vision as free-form research, then a
        # no-tools call that structures those findings against the schema.
        # Forcing output_config.format on the same request as server-side
        # web_search is unstable -- the model cannot always reconcile "call
        # tools" with "emit only JSON" and spirals until it hits max_tokens.
        research = self._research(prompt, image_paths, log_path, allowed_domains=allowed_domains)
        return self._structure(prompt, research, schema, output_path, log_path)

    def _research(
        self,
        prompt: str,
        image_paths: list[Path],
        log_path: Path,
        *,
        allowed_domains: tuple[str, ...],
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
                    "max_uses": MAX_WEB_SEARCHES,
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
        )
        return _parse_review(raw, allowed_domains)
