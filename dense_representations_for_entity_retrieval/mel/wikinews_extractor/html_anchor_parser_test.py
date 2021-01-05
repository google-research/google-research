# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `html_anchor_parser`."""

from absl.testing import absltest
import random
import re

from dense_representations_for_entity_retrieval.mel.wikinews_extractor import html_anchor_parser

_SECTION_TEXT = "Section:::Здравствуй, мир!\nMore text."
_LINKED_TEXT = (
    """Der Komponist <a href="%3Aw%3AWolfgang%20Amadeus%20Mozart">"""
    """Wolfgang Amadeus Mozart</a> wurde am 27. Januar 1756 geboren.""")
_IGNORED_TAG_TEXT = "A new line <br> and ending."
_IGNORED_TAG_PAIR_TEXT = "Here is <nowiki> another word</nowiki>." ""
_THUMBNAIL_ANCHOR_TEXT = "Link <a>tiger.jpg|thumb|200px|</a> skip."
_NESTED_TAG_TEXT = """One <a href="blank">two <a> three </a> </a>"""
_EXPECTED_CLEAN_TEXT = ("Der Komponist Wolfgang Amadeus Mozart wurde am 27. "
                        "Januar 1756 geboren.")
_EXPECTED_MENTION_INFO = {
    "mention": "Wolfgang Amadeus Mozart",
    "position": 14,
    "length": 23,
    "target": ":w:Wolfgang Amadeus Mozart"
}


class WikiExtractorHTMLParserTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._parser = html_anchor_parser.WikiExtractorHTMLParser()

  def _inject_newline(self, text):
    return text.replace("Amadeus Mozart", "Amadeus \nMozart")

  def _extract_mention_by_position(self, mention_info, text):
    pos = mention_info["position"]
    length = mention_info["length"]
    return text[pos:pos + length]

  def test_anchor(self):
    self._parser.feed(_LINKED_TEXT)
    output_text = self._parser.output

    self.assertEqual(_EXPECTED_CLEAN_TEXT, output_text)
    self.assertCountEqual((_EXPECTED_MENTION_INFO,), self._parser.mentions)

    for mention_info in self._parser.mentions:
      self.assertEqual(
          _EXPECTED_MENTION_INFO["mention"],
          self._extract_mention_by_position(mention_info, output_text))
      break  # single mention in input

  def test_anchor_twice(self):
    self._parser.feed(_LINKED_TEXT)
    self._parser.feed(_LINKED_TEXT)
    output_text = self._parser.output

    self.assertEqual(_EXPECTED_CLEAN_TEXT + _EXPECTED_CLEAN_TEXT, output_text)
    self.assertLen(self._parser.mentions, 2)
    for mention_info in self._parser.mentions:
      self.assertEqual(
          _EXPECTED_MENTION_INFO["mention"],
          self._extract_mention_by_position(mention_info, output_text))

    # Verify that they are different mentions.
    positions = {
        mention_info["position"] for mention_info in self._parser.mentions
    }
    self.assertLen(positions, 2)

  def test_anchor_containing_newline(self):
    self._parser.feed(self._inject_newline(_LINKED_TEXT))
    output_text = self._parser.output

    # Verify text.
    self.assertCountEqual(
        [s.strip() for s in _EXPECTED_CLEAN_TEXT.splitlines()],
        [s.strip() for s in output_text.splitlines()])

    # Verify mention extracted.
    EXP_MEN = {
        "mention": "Wolfgang Amadeus Mozart",  # newline replaced by space
        "position": 14,
        "length": 23,
        "target": ":w:Wolfgang Amadeus Mozart"
    }
    self.assertCountEqual((EXP_MEN,), self._parser.mentions)

  def test_ignore_other_tags(self):
    self._parser.feed(_IGNORED_TAG_TEXT)
    output_text = self._parser.output
    self.assertEqual(re.sub("<[^>]+?>", "", _IGNORED_TAG_TEXT), output_text)
    self.assertEmpty(self._parser.mentions)

  def test_ignore_other_tag_pair(self):
    self._parser.feed(_IGNORED_TAG_PAIR_TEXT)
    output_text = self._parser.output
    self.assertEqual(
        re.sub("<[^>]+?>", "", _IGNORED_TAG_PAIR_TEXT), output_text)
    self.assertEmpty(self._parser.mentions)

  def test_ignore_thumb(self):
    self._parser.feed(_EXPECTED_CLEAN_TEXT)
    self._parser.feed(_THUMBNAIL_ANCHOR_TEXT)
    output_text = self._parser.output
    self.assertNotIn(_THUMBNAIL_ANCHOR_TEXT, output_text)
    self.assertIn(_EXPECTED_CLEAN_TEXT, output_text)
    self.assertEmpty(self._parser.mentions)

  def test_nested_anchors_not_supported(self):
    with self.assertRaises(ValueError):
      self._parser.feed(_NESTED_TAG_TEXT)

  def test_unmatched_a_tag_raises(self):
    with self.assertRaises(ValueError):
      self._parser.feed("Unmatched closing tag </a>.")

  def test_passthrough_section_markup_and_nonascii_ok(self):
    self._parser.feed(_SECTION_TEXT)
    output_text = self._parser.output
    self.assertEqual(_SECTION_TEXT, output_text)

  def test_composite_input(self):
    self._parser.feed(_IGNORED_TAG_TEXT)
    self._parser.feed(_LINKED_TEXT)
    self._parser.feed(_SECTION_TEXT)
    self._parser.feed(_IGNORED_TAG_PAIR_TEXT)

    output_text = self._parser.output
    self.assertIn(_EXPECTED_CLEAN_TEXT, output_text)
    self.assertIn(_SECTION_TEXT, output_text)
    self.assertIn(re.sub("<[^>]+?>", "", _IGNORED_TAG_TEXT), output_text)
    self.assertIn(re.sub("<[^>]+?>", "", _IGNORED_TAG_PAIR_TEXT), output_text)

    self.assertLen(self._parser.mentions, 1)
    for mention_info in self._parser.mentions:
      self.assertEqual(
          _EXPECTED_MENTION_INFO["mention"],
          self._extract_mention_by_position(mention_info, output_text))

  def test_whitespace_only(self):
    self._parser.feed("")
    self._parser.feed("  \n")
    self.assertEmpty(self._parser.output)
    self.assertEmpty(self._parser.mentions)


class LinkPatternTest(absltest.TestCase):

  def test_exclude_link_by_prefix(self):
    self.assertTrue(
        html_anchor_parser.exclude_link_by_prefix("category:horses"))

    self.assertFalse(
        html_anchor_parser.exclude_link_by_prefix("One: The number"))

  def test_title_parse(self):
    # Missing w/wikipedia identifier.
    self.assertTupleEqual(
        (None, None),
        html_anchor_parser.parse_title_if_wikipedia(":woosh:nomatch"))

    # Empty title.
    self.assertTupleEqual((None, None),
                          html_anchor_parser.parse_title_if_wikipedia("w:"))

    self.assertTupleEqual(
        (None, "APage"), html_anchor_parser.parse_title_if_wikipedia("w:APage"))
    self.assertTupleEqual(
        (None, "APage"),
        html_anchor_parser.parse_title_if_wikipedia("wikiPEDIA:APage"))
    self.assertTupleEqual(
        (None, "APage"),
        html_anchor_parser.parse_title_if_wikipedia(":wikipedia:APage"))

    # Non-existent language identifier.
    self.assertTupleEqual(
        (None, "xxxx:APage"),
        html_anchor_parser.parse_title_if_wikipedia("w:xxxx:APage"))

    # Valid language identifier.
    self.assertTupleEqual(
        ("es", "APage"),
        html_anchor_parser.parse_title_if_wikipedia("w:es:APage"))

    # Implicitly just a title even though it matches language.
    self.assertTupleEqual((None, "es:"),
                          html_anchor_parser.parse_title_if_wikipedia("w:es:"))


if __name__ == "__main__":
  absltest.main()
