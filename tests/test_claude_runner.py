from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from inky_bird_frame.birds import BirdSpecies
from inky_bird_frame.claude_runner import (
    GENERATOR_LABEL,
    SPECTRA6_PALETTE,
    _final_text,
    _inline_image_bytes,
    composite_plate_labels,
    illustration_prompt,
    label_font_path,
    review_prompt_with_preview,
    spectra_panel_preview,
    supported_schema,
)
from inky_bird_frame.codex_runner import PROFILE_SCHEMA, REVIEW_SCHEMA
from inky_bird_frame.errors import GenerationError
from inky_bird_frame.images import PAPER_COLOR, PORTRAIT_SIZE
from inky_bird_frame.models import ReferencePhoto, SpeciesProfileData


def _species() -> BirdSpecies:
    return BirdSpecies(1, "Test Bird", "Avis test", 1, "test")


def _profile() -> SpeciesProfileData:
    return SpeciesProfileData(
        taxon_id=1,
        common_name="Test Bird",
        scientific_name="Avis test",
        family="Testidae",
        measurements={"length": "1 in", "wingspan": "2 in", "weight": "3 oz"},
        field_marks=["one", "two", "three", "a longer field mark that should wrap onto more lines"],
        habitat="Woods",
        behavior="Perches",
        palette=["red", "green", "blue"],
        sources=[
            {"title": "One", "url": "https://birds.example/one"},
            {"title": "Two", "url": "https://field.example/two"},
        ],
    )


def _reference() -> ReferencePhoto:
    return ReferencePhoto(
        photo_id=1,
        observation_id=2,
        observer="Observer",
        attribution="(c) Observer",
        license_code="cc-by-nc",
        source_url="https://birds.example/photos/1",
        image_url="https://birds.example/photos/1.jpg",
        width=1000,
        height=800,
        filename="1.jpg",
        sha256="0" * 64,
    )


class _Block:
    def __init__(self, block_type: str, text: str = "") -> None:
        self.type = block_type
        self.text = text


class _Response:
    def __init__(self, blocks: list[_Block]) -> None:
        self.content = blocks


class IllustrationPromptTests(unittest.TestCase):
    def test_prompt_carries_identity_and_forbids_lettering_and_rulers(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn('"Test Bird"', prompt)
        self.assertIn('"Avis test"', prompt)
        self.assertIn('"Testidae"', prompt)
        self.assertIn("(c) Observer", prompt)
        # Labels are composited by software in one fixed font; the model draws neither
        # text nor a ruler, both of which drifted when it was allowed to.
        self.assertIn("Do not render any letters", prompt)
        self.assertIn("No ruler, scale bar, tick marks", prompt)
        self.assertNotIn("letter for letter", prompt)
        self.assertNotIn("Correction required", prompt)

    def test_prompt_includes_correction_findings(self) -> None:
        prompt = illustration_prompt(
            _species(),
            _profile(),
            [_reference()],
            ("The eye-ring is too wide.",),
        )
        self.assertIn("Correction required", prompt)
        self.assertIn("The eye-ring is too wide.", prompt)

    def test_prompt_constrains_style_for_the_epaper_panel(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("six-color e-paper", prompt)
        self.assertIn("avoid soft gradients", prompt)

    def test_prompt_specifies_plain_old_paper(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("slightly yellowed cream", prompt)
        self.assertIn("pebbled or stippled grain", prompt)
        self.assertIn("no marks, stains, foxing, or creases", prompt)
        self.assertIn("no binding, spiral, spine, or gutter", prompt)
        self.assertIn("Keep the ageing\n  subtle", prompt)

    def test_prompt_reserves_the_label_zones(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("left third of the page", prompt)
        self.assertIn("never touch the artwork", prompt)

    def test_generator_label_names_both_models(self) -> None:
        self.assertIn("claude-sonnet-5", GENERATOR_LABEL)
        self.assertIn("gemini-3-pro-image", GENERATOR_LABEL)


class ReviewPromptTests(unittest.TestCase):
    def test_preview_prompt_shifts_image_indices_for_the_panel_preview(self) -> None:
        prompt = review_prompt_with_preview(
            _species(),
            _profile(),
            [_reference()],
            ("birds.example", "field.example"),
        )
        self.assertIn("Image 2 is the same candidate quantized", prompt)
        self.assertIn("Images 3 onward", prompt)
        self.assertIn("black, white, red, yellow, green, blue", prompt)
        self.assertIn("birds.example, field.example", prompt)

    def test_preview_prompt_treats_composited_lettering_as_exact(self) -> None:
        prompt = review_prompt_with_preview(_species(), _profile(), [_reference()], ("a.example",))
        self.assertIn("composited by software", prompt)
        self.assertIn("do not lower text_accuracy for the typeface", prompt)

    def test_preview_prompt_fails_painted_text_rulers_and_page_objects(self) -> None:
        prompt = review_prompt_with_preview(_species(), _profile(), [_reference()], ("a.example",))
        self.assertIn("House style is part of composition_quality", prompt)
        self.assertIn("painted\nby the image model", prompt)
        self.assertIn("any ruler, scale bar, tick", prompt)
        self.assertIn("spiral, spine, or gutter", prompt)
        self.assertIn("inset panel", prompt)


class FontAndCompositeTests(unittest.TestCase):
    def test_bundled_font_is_the_default(self) -> None:
        path = label_font_path()
        assert path is not None
        self.assertEqual(path.name, "SpecialElite-Regular.ttf")
        self.assertTrue(path.is_file())

    def test_env_override_wins_when_it_exists(self) -> None:
        from inky_bird_frame.claude_runner import BUNDLED_LABEL_FONT, LABEL_FONT_ENV

        with patch.dict("os.environ", {LABEL_FONT_ENV: str(BUNDLED_LABEL_FONT)}):
            self.assertEqual(label_font_path(), BUNDLED_LABEL_FONT)
        with patch.dict("os.environ", {LABEL_FONT_ENV: "/nonexistent/font.ttf"}):
            self.assertEqual(label_font_path(), BUNDLED_LABEL_FONT)

    def test_long_measurement_lines_stay_inside_the_label_column(self) -> None:
        from PIL import Image

        profile = _profile()
        profile["measurements"]["wingspan"] = (
            "not well documented in standard references or field guides anywhere"
        )
        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        composite_plate_labels(image, profile)
        width, height = image.size
        # Nothing may be inked right of the label column in the upper (label) band.
        column_edge = max(width // 20, 24) + width * 3 // 10 + width // 40
        band = image.crop((column_edge, 0, width, height * 3 // 10))
        self.assertIsNone(
            Image.eval(band.convert("L"), lambda v: 255 if v < 128 else 0).getbbox(),
            "measurement text overran the label column",
        )

    def test_label_block_never_enters_the_bottom_band(self) -> None:
        from PIL import Image

        profile = _profile()
        profile["field_marks"] = [
            "a very long field mark that wraps onto several lines of the label column " * 2
        ] * 9
        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        composite_plate_labels(image, profile)
        width, height = image.size
        band = image.crop((0, height * 7 // 10, width, height))
        self.assertIsNone(
            Image.eval(band.convert("L"), lambda v: 255 if v < 128 else 0).getbbox(),
            "label text entered the bottom 30% reserved for the studies",
        )

    def test_prompt_confines_studies_to_the_lower_right(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("lower-right region only, exactly three", prompt)
        self.assertIn("nothing at all in the lower-left", prompt)
        self.assertIn("from the very top to the very bottom", prompt)

    def test_labels_are_drawn_in_pure_black_without_resizing(self) -> None:
        from PIL import Image, ImageChops

        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        composite_plate_labels(image, _profile())
        self.assertEqual(image.size, PORTRAIT_SIZE)
        blank = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        self.assertIsNotNone(ImageChops.difference(image, blank).getbbox())
        color_counts = image.getcolors(1_000_000)
        assert color_counts is not None
        self.assertIn((0, 0, 0), {color for _, color in color_counts})


class SpectraPreviewTests(unittest.TestCase):
    def test_preview_quantizes_to_the_panel_palette(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "plate.png"
            gradient = Image.new("RGB", (64, 64))
            gradient.putdata([(x * 4, y * 4, 128) for y in range(64) for x in range(64)])
            gradient.save(source)
            destination = root / "logs" / "preview.png"

            spectra_panel_preview(source, destination)

            with Image.open(destination) as preview:
                color_counts = preview.convert("RGB").getcolors(64 * 64)

        assert color_counts is not None
        colors = {color for _, color in color_counts}
        self.assertTrue(colors.issubset(set(SPECTRA6_PALETTE)))
        self.assertGreater(len(colors), 1)


class SupportedSchemaTests(unittest.TestCase):
    def test_strips_numeric_bounds_anthropic_rejects(self) -> None:
        cleaned = supported_schema(REVIEW_SCHEMA)
        score = cleaned["properties"]["species_accuracy"]
        self.assertEqual(score, {"type": "integer"})
        self.assertNotIn("minimum", str(cleaned))
        self.assertNotIn("maximum", str(cleaned))

    def test_preserves_structure_and_required(self) -> None:
        cleaned = supported_schema(REVIEW_SCHEMA)
        self.assertEqual(cleaned["type"], "object")
        self.assertEqual(cleaned["required"], REVIEW_SCHEMA["required"])
        self.assertEqual(cleaned["additionalProperties"], False)

    def test_leaves_constraint_free_schema_unchanged(self) -> None:
        self.assertEqual(supported_schema(PROFILE_SCHEMA), PROFILE_SCHEMA)


class GeminiClientTests(unittest.TestCase):
    def test_client_caps_timeout_and_retries(self) -> None:
        from inky_bird_frame.claude_runner import (
            GEMINI_REQUEST_TIMEOUT_MS,
            GEMINI_RETRY_ATTEMPTS,
            ClaudeRunner,
        )

        captured: dict[str, Any] = {}

        def fake_client(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        with patch("google.genai.Client", fake_client):
            ClaudeRunner(Path("."))._genai()

        http_options = captured["http_options"]
        self.assertEqual(http_options.timeout, GEMINI_REQUEST_TIMEOUT_MS)
        self.assertEqual(http_options.retry_options.attempts, GEMINI_RETRY_ATTEMPTS)
        self.assertIn(429, http_options.retry_options.http_status_codes)


class ResponseParsingTests(unittest.TestCase):
    def test_final_text_returns_last_non_empty_text_block(self) -> None:
        response = _Response(
            [
                _Block("text", "interim narration"),
                _Block("web_search_tool_result"),
                _Block("text", '{"passed": true}'),
            ]
        )
        self.assertEqual(_final_text(response), '{"passed": true}')

    def test_inline_image_bytes_raises_without_image(self) -> None:
        class _Empty:
            candidates: list[object] = []

        with self.assertRaisesRegex(GenerationError, "no image data"):
            _inline_image_bytes(_Empty())


if __name__ == "__main__":
    unittest.main()
