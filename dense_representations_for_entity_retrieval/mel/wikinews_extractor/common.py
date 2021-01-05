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

"""Common utilities and classes for WikiNews-i18n."""

import collections
import glob
import hashlib
import html
import json
import os
import re
import urllib

from absl import logging
import bz2file
import pandas as pd

from dense_representations_for_entity_retrieval.mel.wikinews_extractor import constants
from dense_representations_for_entity_retrieval.mel.wikinews_extractor import html_anchor_parser

_SECTION_REGEX = re.compile(r"Section::::(.+)")


def wiki_encode(url):
  """URLEncode a URL (or URL component) to the format used by Wikipedia.

  Args:
    url: The URL (or URL component) to encode.

  Returns:
    The URL with illegal characters %-encoded and spaces turned to underscores.
  """
  # PY3 changed the handling of '~', but wikimedia keeps it unescaped.
  return urllib.parse.quote(url.replace(" ", "_"),
                            ";@$!*(),/:'").replace("%7E", "~")


def title_to_url(title, language):
  """Converts title to Wikipedia link in given language."""
  # Strip preceding underscores, as per Wikimedia convention.
  title = title.lstrip("_")

  # Convert first letter to upper case and encode.
  encoded_title = wiki_encode(title[0].upper() + title[1:])

  url = "http://{language}.wikipedia.org/wiki/{title}".format(
      language=language,
      title=encoded_title,
  )
  return url


class WikiNewsArchiveParser(object):
  """A class for extracting docs from preprocessed WikiNews archive.

  This targets the jsonl output of github.com/attardi/wikiextractor.
  """

  def __init__(self, wikinews_archive, language):
    """Constructor.

    Args:
      wikinews_archive: Glob pattern matching one or more bzipped files
        containing jsonl records.
      language: Wikipedia-style language code used to produce document URLs.
    """
    self._wikinews_archive = wikinews_archive
    self._language = language
    self._filtered_no_text = []
    self._counter = collections.Counter()

  def documents(self):
    """Iterates over pages in the wikinews archive.

    Yields:
      docid, title, url, curid, revid, wiki_doc
    """
    logging.info("Extracting docs from [%s]", self._wikinews_archive)
    file_list = glob.glob(self._wikinews_archive)
    assert file_list, self._wikinews_archive
    for archive in file_list:
      with bz2file.open(archive, "rt", encoding="utf-8", errors="strict") as xf:
        # One line of json as produced by wikiextractor.
        for line in xf:
          record = json.loads(line)

          # Extract page title.
          title = html.unescape(record["title"])
          curid = record["id"]  # pageid, e.g. 73052
          revid = record["revid"]  # revision, e.g. 730271
          url = record["url"]  # e.g. https://de.wikinews.org/wiki?curid=73052
          logging.debug("Got title: %s", title)

          # Skip pages that don't have text.
          wiki_doc = record.get("text")
          if not wiki_doc:
            self._counter["no text"] += 1
            self._filtered_no_text.append(url)
            logging.debug("Skip: no text element")
            continue

          # Create internal document identifier.
          docid = f"{self._language}-{curid}"
          logging.debug("Found (%s): %s", docid, title)
          self._counter["found"] += 1

          yield docid, title, url, curid, revid, wiki_doc


class WikiNewsArchiveConsumer(object):
  """Extract pages through a WikiNewsArchiveParser."""

  def __init__(
      self,
      archive_parser,
      wiki_output_dir,
      max_docs=None,
      total_docs=None,
  ):
    """Constructor.

    Args:
      archive_parser: An object providing a documents() function that yields
        (docid, title, url, wiki_doc) tuples, such as WikiNewsArchiveParser.
      wiki_output_dir: Directory to output wiki_doc to, one file per document.
      max_docs: Stop after processing this many documents. Process everything if
        this is None.
      total_docs: Controls the stride when max_docs is given, otherwise ignored.
    """
    self._archive_parser = archive_parser
    self._wiki_dir = wiki_output_dir
    self._max_docs = max_docs
    self._total_docs = total_docs

  def extract_docs(self, doc_index_df=None):
    """Process documents using the underlying archive_parser.

    This outputs each wiki_doc to a file under wiki_output_dir, if specified.

    Args:
      doc_index_df: optional document index as DataFrame. If provided, only
        extract docs in this index.

    Returns:
      document index as dataframe

    Raises:
      ValueError when doc_index_df is provided and there is a mismatch in what
      documents can be extracted.
    """

    # Stride.
    incr = 1
    if self._total_docs and self._max_docs:
      incr = max([1, int(self._total_docs / self._max_docs)])
    logging.debug("Incr: %d", incr)

    logging.info("Creating wiki dir: [%s]", self._wiki_dir)
    os.makedirs(self._wiki_dir, exist_ok=True)

    doc_index_list = []
    for i, item in enumerate(self._archive_parser.documents()):
      if i % incr != 0:
        continue
      (docid, title, url, curid, revid, wiki_doc) = item

      # Limit to pre-existing index if one was provided.
      if doc_index_df is None or docid in doc_index_df.index:
        logging.debug("Take: %s", title)

        doc_index_list.append({
            "docid": docid,
            "curid": curid,
            "revid": revid,
            "title": title,
            "url": url,
        })

        if self._wiki_dir:
          # Write the extracted wiki doc to disk for parsing and analysis.
          with open(os.path.join(self._wiki_dir, docid), "wb") as f:
            f.write(wiki_doc.encode())

        if self._max_docs and self._max_docs > 0 and len(
            doc_index_list) >= self._max_docs:
          logging.info("Stop after %d inputs.", len(doc_index_list))
          break

    if not doc_index_list:
      raise ValueError("No documents founds. "
                       "Too restrictive doc-index, or wrong input archive?")
    new_doc_index_df = pd.DataFrame(doc_index_list).set_index(
        "docid", verify_integrity=True)

    return new_doc_index_df


###############################################################################
# WikiNewsDocParser
###############################################################################

# Set of normalized headings to match against when identifying the end of
# article content.
_END_OF_CONTENT_HEADINGS_FOR_MATCH = frozenset(
    [s.casefold() for s in constants.END_OF_CONTENT_HEADINGS])


class WikiNewsDocParser(object):
  """Parse mentions from a wikinews doc as preprocessed by wiki-extractor."""

  def __init__(self, language):
    """Constructor.

    Args:
      language: Wikipedia-style language code of the doc. Used as fallback for
        mention hyperlink targets when a language is not encoded.
    """
    self._language = language

  def _truncate_end(self, wiki_doc):
    """Truncates end of document based on predefined headings."""
    # WikiExtractor outputs headings with special syntax, e.g.
    #   Section::::Quellen.
    # Truncate everything after the first heading that matches the predefined
    # list.
    end_pos = len(wiki_doc)
    for m in _SECTION_REGEX.finditer(wiki_doc):
      heading = m.group(1)
      if heading.rstrip(". ").casefold() in _END_OF_CONTENT_HEADINGS_FOR_MATCH:
        logging.debug("Stop @ %s", heading)
        end_pos = m.start()
        break
    truncated_doc = wiki_doc[:end_pos]

    # Replace remaining section heading markup with the section title and an
    # extra new line.
    return _SECTION_REGEX.sub(r"\1\n", truncated_doc)

  def parse_doc(self, wiki_doc, docid):
    """Parse wiki_doc to produce a text document and a set of mention spans."""
    output_text = ""
    logging.debug("Parsing doc [%s]", docid)
    wiki_doc = self._truncate_end(wiki_doc)

    markup_parser = html_anchor_parser.WikiExtractorHTMLParser()
    raw_mentions = []
    try:
      markup_parser.feed(wiki_doc)
      output_text = markup_parser.output
      raw_mentions = markup_parser.mentions
    except ValueError:
      # Ignore all mentions from these pages.
      logging.warning("Ignore %d mentions due to parse fail [%s]",
                      len(raw_mentions), docid)
      raw_mentions = []

    # Filter raw mentions to ones that land on Wikipedia pages and add
    # page-level metadata.
    final_mentions = []
    for mention_dict in raw_mentions:
      link_target = mention_dict.pop("target")

      # Limit mentions to links starting with wikipedia prefix.
      lang, title = html_anchor_parser.parse_title_if_wikipedia(link_target)
      if not title:  # need at least title
        logging.debug("Skip anchor with target: %s", link_target)
        continue

      # Default to the wiki's language when the link target itself does not
      # provide a language code.
      lang = lang or self._language

      mention_dict["docid"] = docid
      mention_dict["url"] = title_to_url(title, lang)  # entity target
      final_mentions.append(mention_dict)

    return output_text, pd.DataFrame(final_mentions), markup_parser._tags


def load_doc_index(doc_index_path):
  """Load the doc index dataframe from a TSV file."""
  logging.info("Reading doc index from: [%s]", doc_index_path)
  return pd.read_csv(
      doc_index_path, sep="\t", encoding="utf-8", index_col="docid")


def write_doc_index(doc_index, doc_index_path, overwrite=False):
  """Write the doc index dataframe to a TSV file."""
  if not overwrite:
    assert not os.path.exists(doc_index_path)

  logging.info("Writing doc index to: [%s]", doc_index_path)
  # docid is already the DataFrame Index.
  column_order = ["title", "curid", "revid", "url", "text_md5"]
  doc_index.to_csv(
      doc_index_path, sep="\t", encoding="utf-8", columns=column_order)
