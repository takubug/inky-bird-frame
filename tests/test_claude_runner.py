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

    def test_prompt_leaves_the_reserved_label_area_unmarked(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("That reserved area has no visible\nboundary of any kind", prompt)
        self.assertIn("do not draw a header rule beneath the top band", prompt)

    def test_prompt_specifies_plain_old_paper(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("slightly yellowed cream", prompt)
        self.assertIn("pebbled or stippled grain", prompt)
        self.assertIn("no marks, stains, foxing, or creases", prompt)
        self.assertIn("no binding, spiral, spine, or gutter", prompt)
        self.assertIn("Keep the ageing subtle", prompt)

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

    def test_a_trailing_qualifier_is_dropped_rather_than_dangling(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import _dangles, _label_font, _measurement_lines

        profile = _profile()
        profile["measurements"]["wingspan"] = "70\u201390 cm (28\u201335 in), approximate"
        draw = ImageDraw.Draw(Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR))
        width = PORTRAIT_SIZE[0]
        for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
            wingspan = _measurement_lines(
                draw, profile, _label_font(size), width * 3 // 10, keep_imperial=True
            )[1]
            joined = " ".join(wingspan)
            self.assertNotIn("approx", joined)
            self.assertLessEqual(len(wingspan), 2)
            # A second line may be a whole parenthetical like "(28-35 in)", never "in)".
            self.assertFalse(_dangles(wingspan), wingspan)
        # At a face where the inches fit on one line they are kept.
        small = _measurement_lines(
            draw, profile, _label_font(18), width * 3 // 10, keep_imperial=True
        )[1]
        self.assertEqual(small, ["Wingspan: 70\u201390 cm (28\u201335 in)"])
        # At the plate face the parenthetical moves whole to the second line.
        plate = _measurement_lines(
            draw, profile, _label_font(24), width * 3 // 10, keep_imperial=True
        )[1]
        self.assertEqual(plate, ["Wingspan: 70\u201390 cm", "(28\u201335 in)"])
        # And no reading is ever cut inside a parenthesis.
        for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
            for line in _measurement_lines(
                draw, profile, _label_font(size), width * 3 // 10, keep_imperial=True
            )[1]:
                self.assertEqual(line.count("("), line.count(")"), line)

    def test_measurements_without_a_figure_are_left_off_the_plate(self) -> None:
        from inky_bird_frame.claude_runner import _measurement_specs

        profile = _profile()
        profile["measurements"]["wingspan"] = "not well documented"
        self.assertEqual(
            [label for label, _, _ in _measurement_specs(profile)], ["Length", "Weight"]
        )

    def test_drawn_page_lines_flags_thin_rules_but_not_clean_paper_or_broad_shapes(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import drawn_page_lines

        clean = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        self.assertEqual(drawn_page_lines(clean), ())

        ruled = clean.copy()
        draw = ImageDraw.Draw(ruled)
        draw.line([(300, 0), (300, 1100)], fill=(150, 140, 120), width=2)  # column rule, 69% tall
        draw.line([(300, 320), (1180, 320)], fill=(150, 140, 120), width=1)  # header rule, 73% wide
        findings = drawn_page_lines(ruled)
        self.assertEqual(len(findings), 2, findings)
        self.assertIn("vertical line runs across 69% of the page at 25% of the width", findings[0])
        self.assertIn(
            "horizontal line runs across 73% of the page at 20% of the height", findings[1]
        )
        self.assertIn("Draw no line, rule, fold, crease, border, or panel edge", findings[0])

        broad = clean.copy()
        ImageDraw.Draw(broad).rectangle([(500, 0), (560, 1599)], fill=(40, 40, 40))  # a dark bar
        self.assertEqual(drawn_page_lines(broad), ())

        edge = clean.copy()
        ImageDraw.Draw(edge).line([(2, 0), (2, 1599)], fill=(120, 110, 90), width=3)  # bezel shadow
        self.assertEqual(drawn_page_lines(edge), ())

    def test_leading_approx_is_dropped_and_a_unit_is_never_trimmed_away(self) -> None:
        from inky_bird_frame.claude_runner import _measurement_text, _trim_candidates

        self.assertEqual(
            _measurement_text(
                "Wingspan", "approx. 100\u2013120 cm (est., not precisely documented)"
            ),
            "Wingspan: 100\u2013120 cm (est., not precisely documented)",
        )
        self.assertEqual(
            _measurement_text("Weight", "About 650\u2013800 g"), "Weight: 650\u2013800 g"
        )
        for candidate in _trim_candidates("100\u2013120 cm (est., not precisely documented)"):
            self.assertFalse(candidate.split()[-1][-1].isdigit(), candidate)

    def test_erase_page_lines_removes_a_rule_but_keeps_strokes_crossing_it(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import drawn_page_lines, erase_page_lines

        page = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        draw = ImageDraw.Draw(page)
        draw.line([(300, 0), (300, 1200)], fill=(150, 140, 120), width=2)  # column rule
        draw.line([(280, 600), (330, 600)], fill=(0, 0, 0), width=3)  # a stroke across it
        erased = erase_page_lines(page)
        self.assertEqual(len(erased), 1, erased)
        self.assertIn("vertical line", erased[0])
        self.assertEqual(drawn_page_lines(page), ())
        self.assertEqual(page.getpixel((300, 200)), PAPER_COLOR)  # rule gone
        self.assertEqual(page.getpixel((300, 600)), (0, 0, 0))  # crossing stroke kept
        self.assertEqual(erase_page_lines(page), ())  # idempotent

    def test_a_dotted_column_is_not_a_line_but_a_short_underline_is_erased(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import drawn_page_lines, erase_page_lines

        page = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        draw = ImageDraw.Draw(page)
        for y in range(0, 1600, 12):  # a dotted paper grain aligned in one column
            draw.point((400, y), fill=(120, 110, 90))
        draw.line([(600, 1450), (1000, 1450)], fill=(150, 140, 120), width=1)  # 33% underline
        self.assertEqual(drawn_page_lines(page), ())  # neither spans 40% as a real line
        erased = erase_page_lines(page)
        self.assertEqual(len(erased), 1, erased)
        self.assertRegex(erased[0], r"horizontal line runs across 3[0-9]%")
        self.assertEqual(page.getpixel((800, 1450)), PAPER_COLOR)
        self.assertEqual(page.getpixel((400, 600)), (120, 110, 90))  # grain left alone

    def test_flowed_field_marks_continue_past_an_intrusion_instead_of_stopping(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import _flow_lines, _label_font

        profile = _profile()
        draw = ImageDraw.Draw(Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR))
        width, height = PORTRAIT_SIZE
        column_width = width * 3 // 10
        free = [column_width] * height
        for y in range(500, 700):  # a tail sweeping deep into the column
            free[y] = column_width * 2 // 5
        body_size = max(width // 44, 10)
        placed, complete = _flow_lines(
            draw, profile, _label_font(body_size), body_size, free, column_width, 300, height
        )
        self.assertTrue(complete)
        text = " ".join(line.strip("\u2022 ") for _, line in placed)
        for mark in profile["field_marks"]:
            self.assertIn(mark, text)  # every word of every mark is on the page, in order
        narrow = [line for y, line in placed if 500 <= y < 700]
        self.assertTrue(narrow)
        for line in narrow:
            self.assertLessEqual(
                draw.textlength(line, font=_label_font(body_size)), column_width * 2 // 5
            )

    def test_imperial_readings_are_shown_on_every_measurement_or_on_none(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import (
            _has_imperial,
            _label_font,
            _measurement_lines,
            _measurements_fit,
        )

        draw = ImageDraw.Draw(Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR))
        width = PORTRAIT_SIZE[0]
        font = _label_font(max(width // 44, 10))
        column = width * 3 // 10

        # One measurement without a conversion drops it from all of them.
        mixed = _profile()
        mixed["measurements"] = {
            "length": "46\u201353 cm (18\u201321 in)",
            "wingspan": "100\u2013120 cm",
            "weight": "650\u2013800 g",
        }
        self.assertEqual(
            _measurement_lines(draw, mixed, font, column),
            [["Length: 46\u201353 cm"], ["Wingspan: 100\u2013120 cm"], ["Weight: 650\u2013800 g"]],
        )

        # When every measurement offers one, the face shrinks to keep them, and at
        # any face the plate shows three conversions or none -- never one or two.
        whole = _profile()
        whole["measurements"] = {
            "length": "14\u201315 cm (5.5\u20136 in)",
            "wingspan": "28\u201332 cm (11\u201312.5 in)",
            "weight": "10\u201315 g (0.35\u20130.53 oz)",
        }
        self.assertTrue(_measurements_fit(draw, whole, _label_font(max(width // 60, 8)), column))
        self.assertEqual(
            _measurement_lines(draw, whole, _label_font(max(width // 60, 8)), column),
            [
                ["Length: 14\u201315 cm (5.5\u20136 in)"],
                ["Wingspan: 28\u201332 cm (11\u201312.5 in)"],
                ["Weight: 10\u201315 g (0.35\u20130.53 oz)"],
            ],
        )
        for candidate in (whole, mixed):
            for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
                shown = [
                    lines
                    for lines in _measurement_lines(draw, candidate, _label_font(size), column)
                    if _has_imperial(" ".join(lines))
                ]
                self.assertIn(len(shown), (0, 3), (candidate["measurements"], size, shown))

        # A non-imperial parenthetical is a note, not a conversion, and is left alone.
        noted = _profile()
        noted["measurements"] = {
            "length": "16\u201320 cm",
            "wingspan": "24\u201330 cm (wild birds)",
            "weight": "15\u201324 g",
        }
        wingspan = _measurement_lines(draw, noted, font, column)[1]
        self.assertEqual(" ".join(wingspan), "Wingspan: 24\u201330 cm (wild birds)")

    def test_a_parenthetical_is_never_split_across_lines(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import _label_font, _measurement_lines

        profile = _profile()
        profile["measurements"]["wingspan"] = "17\u201321 cm (6.7\u20138.3 in) folded wing chord"
        draw = ImageDraw.Draw(Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR))
        width = PORTRAIT_SIZE[0]
        for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
            wingspan = _measurement_lines(
                draw, profile, _label_font(size), width * 3 // 10, keep_imperial=True
            )[1]
            for line in wingspan:
                self.assertEqual(line.count("("), line.count(")"), wingspan)
        plate = _measurement_lines(
            draw, profile, _label_font(24), width * 3 // 10, keep_imperial=True
        )[1]
        self.assertEqual(plate, ["Wingspan: 17\u201321 cm", "(6.7\u20138.3 in)"])

    def test_long_qualifiers_are_trimmed_instead_of_shrinking_the_face(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import _label_font, _measurement_lines

        profile = _profile()
        profile["measurements"]["wingspan"] = (
            "60\u201370 cm; proportionate to a small-to-medium honeyeater build"
        )
        draw = ImageDraw.Draw(Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR))
        width = PORTRAIT_SIZE[0]
        default_face = _label_font(max(width // 44, 10))
        wrapped = _measurement_lines(draw, profile, default_face, width * 3 // 10)
        # The wingspan is trimmed to its leading clause; a lone whole word such as
        # "documented" may end the second line, a fragment may not.
        self.assertLessEqual(len(wrapped[1]), 2)
        self.assertIn("60\u201370 cm", " ".join(wrapped[1]))
        self.assertNotIn("proportionate", " ".join(wrapped[1]))

    def test_length_and_weight_never_wrap_and_wingspan_takes_at_most_two_lines(self) -> None:
        from PIL import Image, ImageDraw

        from inky_bird_frame.claude_runner import _label_font, _measurement_lines

        profile = _profile()
        profile["measurements"] = {
            "length": "33\u201337 cm (13\u201314.5 in)",
            "wingspan": "45\u201355 cm (17.7\u201321.7 in) in standard references",
            "weight": "approx. 80\u2013120 g",
        }
        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        composite_plate_labels(image, profile)
        # Re-derive the face the compositor settled on by checking the invariant at
        # every size down from the default: the largest size that satisfies it is
        # what gets drawn, so at least one size must satisfy it.
        draw = ImageDraw.Draw(image)
        width = PORTRAIT_SIZE[0]
        column_width = width * 3 // 10
        satisfied = []
        for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
            wrapped = _measurement_lines(draw, profile, _label_font(size), column_width)
            satisfied.append([len(w) for w in wrapped])
        self.assertTrue(any(counts == [1, 2, 1] or counts == [1, 1, 1] for counts in satisfied))
        # No dangling single word on a second line, at any face.
        for size in range(max(width // 44, 10), max(width // 70, 8) - 1, -1):
            wingspan = _measurement_lines(draw, profile, _label_font(size), column_width)[1]
            if len(wingspan) == 2:
                self.assertGreaterEqual(len(wingspan[1].split()), 2, wingspan)
        # The abbreviation still applies whenever the full qualifier is kept.
        from inky_bird_frame.claude_runner import _measurement_text

        self.assertIn("refs", _measurement_text("Wingspan", "in standard references"))

    def test_paper_grain_does_not_narrow_the_column(self) -> None:
        import random

        from PIL import Image

        from inky_bird_frame.claude_runner import _ink_profile

        random.seed(7)
        width, height = PORTRAIT_SIZE
        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        px = image.load()
        assert px is not None
        base = PAPER_COLOR
        for y in range(0, height, 2):  # a faint, even stipple across the whole page
            for x in range(0, width, 2):
                d = random.randint(-28, 28)
                px[x, y] = (base[0] + d, base[1] + d, base[2] + d)
        margin = max(width // 20, 24)
        column_width = width * 3 // 10
        free = _ink_profile(image, margin, column_width)
        self.assertGreaterEqual(min(free), column_width * 9 // 10)

    def test_labels_flow_around_ink_that_intrudes_into_the_column(self) -> None:
        from PIL import Image, ImageDraw

        width, height = PORTRAIT_SIZE
        margin = max(width // 20, 24)
        column_width = width * 3 // 10
        image = Image.new("RGB", PORTRAIT_SIZE, PAPER_COLOR)
        # A dark "tail" sweeping into the right half of the label column mid-page.
        blob = (
            margin + column_width // 2,
            height * 30 // 100,
            margin + column_width,
            height * 45 // 100,
        )
        ImageDraw.Draw(image).rectangle(blob, fill=(70, 60, 50))
        profile = _profile()
        profile["field_marks"] = ["a fairly long field mark that will need several lines"] * 8
        composite_plate_labels(image, profile)
        # No label ink (pure black) may land inside the intrusion.
        region = image.crop(blob)
        colours = region.getcolors(region.size[0] * region.size[1]) or []
        self.assertNotIn((0, 0, 0), {c for _, c in colours})
        # And labels were still drawn beside it (something black exists left of the blob).
        beside = image.crop((margin, blob[1], blob[0] - 2, blob[3]))
        colours_beside = beside.getcolors(beside.size[0] * beside.size[1]) or []
        self.assertIn((0, 0, 0), {c for _, c in colours_beside})

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

    def test_prompt_puts_studies_in_a_full_width_bottom_band(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("bottom quarter of the page", prompt)
        self.assertIn("spanning the full width", prompt)
        self.assertIn("Fill the page", prompt)
        self.assertIn("left third of the page above\nthe bottom quarter", prompt)

    def test_review_penalises_empty_regions(self) -> None:
        prompt = review_prompt_with_preview(_species(), _profile(), [_reference()], ("a.example",))
        self.assertIn("poorly filled page", prompt)

    def test_prompt_and_review_forbid_creases_through_the_labels(self) -> None:
        prompt = illustration_prompt(_species(), _profile(), [_reference()])
        self.assertIn("no ruled lines,\n  column rules, folds, or creases", prompt)
        review = review_prompt_with_preview(_species(), _profile(), [_reference()], ("a.example",))
        self.assertIn("crease running through the label area", review)
        self.assertIn("no boxes, cells, frames, table lines", prompt)
        self.assertIn("divider drawn around, between, or beneath the studies", review)

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
