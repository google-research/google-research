# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Basic HTML parser for anchor links only.

Intended for use on a specific style of output produced by the third-party
`wikiextractor`.
"""

import collections
from html import parser as html_parser
import re
import urllib

from absl import logging

from dense_representations_for_entity_retrieval.mel.wikinews_extractor import constants

_ANCHOR_EXCLUDE_REGEX = re.compile("|".join(constants._ANCHOR_EXCLUDE_PATTERNS),
                                   re.UNICODE | re.IGNORECASE)

# Regex to detect and parse wiki link targets that point to Wikipedia.
# Breakdown of the expression:
#     Optional initial colon.
#     Required "w".
#     Optionally write out "wikipedia".
#     Required colon.
#     Optional language spec of the form "LANG:", capturing the LANG part.
#     Required non-zero length title.
# Example strings (square brackets just for display here):
#   [[w:title] / [[:w:title]] / [[:w:de:title]] / [[:wikipedia:de:title]]
# TODO(jabot): Consider also exploiting the d: prefixes that point to WikiData.
_WIKIPEDIA_TITLE_REGEX = re.compile(
    r":?w(?:ikipedia)?:(?:(%s):)?(.+)" % "|".join(constants.WIKI_LANG_CODES),
    re.IGNORECASE)


def parse_title_if_wikipedia(title):
  """Require wikipedia prefix and parse title into language and title."""

  m = _WIKIPEDIA_TITLE_REGEX.match(title)
  if m:
    # Two capturing groups even if the first one was empty; hence language will
    # be None here if the language spec was absent.
    language, title = m.groups()
    if language:
      language = language.lower()
    return language, title
  else:
    return None, None


# Regex to detect blocked terms in link prefixes. The terms appear in a context
# like "[[term:foobar]]".
_LINK_PREFIX_BLOCKLIST_REGEX = re.compile(
    "(?:%s):.+" % "|".join(constants.LINK_PREFIX_BLOCKLIST),
    re.UNICODE | re.IGNORECASE)


def exclude_link_by_prefix(unqouted_url):
  """Returns whether this URL is in the exclusion list."""
  return _LINK_PREFIX_BLOCKLIST_REGEX.match(unqouted_url.lstrip())


class WikiExtractorHTMLParser(html_parser.HTMLParser):
  """Light-weight parser to extract text and linked mentions.

  Targets the wiki-extractor's output, which should only have <a> tags. Parsing
  will throw an exception when those tags are nested.

  The class is stateful, so use a new instance for each next document.

  It only handles unnested <a> tags. It drops any other tags.

  Usage:

    parser = WikiExtractorHTMLParser()

    try:
      parser.feed(wiki_doc)
      output = parser.output
      mentions = parser.mentions
    except ValueError:
      pass # parse failure
  """

  def __init__(self):
    super(WikiExtractorHTMLParser, self).__init__()

    # Cleaned output text.
    self._output = ""

    # List of mention dicts extracted.
    self._mentions = []

    # State information.
    # This indicates whether we are inside an <a> tag.
    self._in_a_tag = False

    # The href attribute for the <a> tag we are inside of.
    self._href = ""

    # The anchor text for the <a> tag we are inside of.
    self._anchor_text = ""

    # The character position in output where the <a> tag started.
    self._last_pos = 0

    # Counter of observed tags; for diagnostics.
    self._tags = collections.Counter()

    self._found_end = False

  @property
  def mentions(self):
    """Returns list of mentions extracted so far, as a `dict` per mention."""
    return self._mentions

  def handle_starttag(self, tag, attrs):
    logging.debug("Encountered a start tag %s (%s):", tag,
                  ",".join(["{}={}".format(k, v) for (k, v) in attrs]))
    if tag == "a":
      # Fail on nested a-tags for simplicity.
      if self._in_a_tag:
        raise ValueError("Handling of nested <a>-tags is not implemented.")
      self._in_a_tag = True
      self._reset()

      # Extract hyperlink target.
      for name, value in attrs:
        if name == "href":
          self._href = urllib.parse.unquote(value)

  def handle_data(self, data):
    logging.debug("Encountered data: '%s'", data)

    # Normalize whitespace by eliminating trailing whitespace and more than
    # two consecutive newlines.
    self._output = re.sub(r" *\n", r"\n", self._output)
    self._output = re.sub(r"\n\n\n*", r"\n\n", self._output)

    # Remove preceding whitespace when starting a new line.
    if self._output and self._output[-1] == "\n":
      data = data.lstrip()

    # Capture anchor text if inside an a-tag and track position.
    if self._in_a_tag and self._href:
      # Completely drop certain anchor texts based on a blocklist.
      if _ANCHOR_EXCLUDE_REGEX.search(data):
        return

      _, title = parse_title_if_wikipedia(self._href)
      self._anchor_text = ""
      if title:
        # If this is a wikipedia link, keep the anchor text and extract
        # mention.
        mention_text = data.replace(" \n", " ").replace("\n", " ")
        self._anchor_text = mention_text
        self._last_pos = len(self._output)
        self._output += mention_text
        logging.debug("Keep link, keep contents: %s", self._href)
      elif exclude_link_by_prefix(self._href):
        # If link matches exclusion list, do not append it's anchor text.
        logging.debug("Drop link, drop contents: %s", self._href)
        pass
      else:
        # For all other links, ignore the hyperlink but keep the anchor text.
        logging.debug("Drop link, keep contents: %s", self._href)
        self._output += data
    else:
      # Just append output.
      self._output += data

  def _reset(self):
    self._anchor_text = ""  # reset
    self._href = ""

  def handle_endtag(self, tag):
    logging.debug("Encountered an end tag: '%s'", tag)
    self._tags.update((tag,))

    if tag == "a":
      if not self._in_a_tag:
        raise ValueError("Spurious end tag.")
      self._in_a_tag = False

      # Extract mention if we got a well-formed link with anchor text.
      if self._href and self._anchor_text:

        self._mentions.append({
            "mention": self._anchor_text,
            "position": self._last_pos,
            "length": len(self._anchor_text),
            "target": self._href,
        })

      self._reset()

  @property
  def output(self):
    """Returns output text extracted so far."""
    self._output = self._output.rstrip()
    return self._output
