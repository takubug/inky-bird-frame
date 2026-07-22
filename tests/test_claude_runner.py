from __future__ import annotations

import unittest

from inky_bird_frame.birds import BirdSpecies
from inky_bird_frame.claude_runner import (
    GENERATOR_LABEL,
    _final_text,
    _inline_image_bytes,
    composite_plate_labels,
    illustration_prompt,
)
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
    def test_prompt_carries_identity_and_forbids_lettering(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn('"Test Bird"', prompt)
        self.assertIn('"Avis test"', prompt)
        self.assertIn('"Testidae"', prompt)
        self.assertIn("(c) Observer", prompt)
        self.assertIn("Do not render any letters", prompt)
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

    def test_generator_label_names_both_models(self) -> None:
        self.assertIn("claude-opus-4-8", GENERATOR_LABEL)
        self.assertIn("gemini-2.5-flash-image", GENERATOR_LABEL)


class CompositeLabelTests(unittest.TestCase):
    def test_labels_are_drawn_without_resizing(self) -> None:
        from PIL import Image, ImageChops

        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        composite_plate_labels(image, _profile())
        self.assertEqual(image.size, PORTRAIT_SIZE)
        blank = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        self.assertIsNotNone(ImageChops.difference(image, blank).getbbox())


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
