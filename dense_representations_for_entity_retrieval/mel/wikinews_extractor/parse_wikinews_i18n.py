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

r"""Universal WikiNews linked mention extractor.

Extract Wikipedia entity mentions and the document text they appear in, given a
WikiNews snapshot.

The main input (wikinews_archive) is the jsonl archives produced by running
wiki-extractor (with namespace filtering deactivated). It has one record per
document. The records include lightly marked up text that retains <a> tags and
Section::: markings. See html_anchor_parser.WikiExtractorHTMLParser and
common.WikiNewsDocParser for details.

----------------------------------------------------
'dataset' mode: **intended for public consumption**
----------------------------------------------------
A dataset is pre-defined by the secondary inputs: docs.tsv, mentions.tsv

In this mode, the program extracts the text needed as context for the provided
mentions, and verifies that all of them have been found.

Outputs:
  -Clean text is output to --output_dir_text.
  -Intermediate mark-up documents are output to --output_dir_wiki.

----------------------------------------------------
'raw' mode: intended for corpus development
----------------------------------------------------
Runs in unconstrained mode to extract as many mentions and documents from the
input archive as possible.

Outputs:
 - docs.tsv
 - mentions.tsv
 - Clean text is output to --output_dir_text.
 - Intermediate mark-up documents are output to --output_dir_wiki.
"""

import collections
import hashlib
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from dense_representations_for_entity_retrieval.mel.wikinews_extractor import common

flags.DEFINE_enum("language", None,
                  "ar de en es fa ja sr ta tr ca cs pl ro sv uk".split(),
                  "Language code. Must be one of the supported languages.")

flags.DEFINE_string(
    "wikinews_archive", None,
    "Glob pattern matching .bz2 files containing jsonl as output by "
    "wikiextractor.")
flags.DEFINE_string("output_dir_wiki", None,
                    "Output directory for wikidoc files.")
flags.DEFINE_string("output_dir_text", None,
                    "Output directory for clean text output files.")
flags.DEFINE_string(
    "doc_index_path", None,
    "Path to docs.tsv. 'dataset' mode loads the document index from here; "
    "'raw' mode outputs it here.")
flags.DEFINE_string(
    "mention_index_path", None,
    "Path to mentions.tsv. 'dataset' mode loads the mention index from here; "
    "'raw' mode outputs it here")
flags.DEFINE_enum(
    "mode", "dataset", ["dataset", "raw"],
    "Execution mode: 'raw'-mode extracts as many documents and mentions "
    "from wikinews_archive as possible. 'dataset'-mode only extracts what "
    "is covered by the provided {docs,mentions}.tsv indices.")
flags.DEFINE_integer("max_docs", None, "Maximum documents to extract.")

FLAGS = flags.FLAGS


def extract_text_and_mentions(doc_index, wiki_doc_parser, wiki_dir, text_dir):
  """Parse wikidoc files from wiki_dir to populate text_dir."""
  logging.info("Creating text dir: [%s]", text_dir)
  os.makedirs(text_dir, exist_ok=True)
  logging.info("Parsing docs from [%s]", wiki_dir)

  # List of per-document mention DataFrames created while parsing the wikitext.
  mention_index_dfs = []

  docid_to_text_hash = {}

  # Read wiki_doc and parse into clean text and mentions.
  docids = set(doc_index.index)
  for docid in docids:
    with open(os.path.join(wiki_dir, docid), "rb") as f:
      wiki_doc = f.read().decode("utf-8")

    text_doc, doc_mention_index, _ = wiki_doc_parser.parse_doc(wiki_doc, docid)

    text_doc_bytes = text_doc.encode()
    docid_to_text_hash[docid] = hashlib.md5(text_doc_bytes).hexdigest()
    mention_index_dfs.append(doc_mention_index)

    # Output clean text to file.
    with open(os.path.join(text_dir, docid), "wb") as f:
      f.write(text_doc_bytes)

  # Build single mention dataframe.
  mention_index_df = pd.concat(
      mention_index_dfs,
      ignore_index=True,
  ).sort_values(
      by=["docid", "position"],
      ignore_index=True,
  )
  return mention_index_df, docid_to_text_hash


def verify_mention_index(original, reconstructed):
  """Verify that reconstruction succeeded relative to original mention_index."""

  def df_to_strings(df):
    """Returns unique IDs for the mentions in df."""
    return df.apply(
        lambda r: "{}:{}-{}".format(r["docid"], r["mention"], r["position"]),
        axis=1).values

  expected_records = set(df_to_strings(original))
  got_records = set(df_to_strings(reconstructed))
  missing = expected_records.difference(got_records)
  for missed in missing:
    logging.warning("Missing mention: %s", missed)
  return not missing


def verify_doc_index(original, reconstructed):
  """Verifies document index and final extracted text."""
  # Check if all documents were found.
  missing_docs = set(original.index).difference(reconstructed.index)
  for missed in missing_docs:
    logging.warning("Missing doc: %s", missed)

  # Check if the extracted text match the original text.
  merged = pd.merge(
      original,
      reconstructed,
      how="left",
      on="docid",
      suffixes=("_expected", "_got"))
  text_diff = (merged["text_md5_expected"] != merged["text_md5_got"])
  for missed in merged[text_diff].itertuples():
    logging.warning("Text hash mismatch: %s - %s", missed.Index,
                    missed.text_md5_got)

  return not missing_docs and not any(text_diff)


def verify_context(mention_df, text_dir):
  """Verifies that the mentions can be located in the text."""
  for docid, group in mention_df.groupby(by="docid"):
    with open(os.path.join(text_dir, docid), "rb") as f:
      text = f.read().decode("utf-8")
    for _, row in group.iterrows():
      start = int(row["position"])
      end = start + int(row["length"])
      mention = text[start:end]
      assert row["mention"] == mention, (docid, row["mention"], mention)
  return True


def load_mention_index(mention_index_path):
  """Load the mention index dataframe from a TSV file."""
  logging.info("Reading mention index from: [%s]", mention_index_path)
  return pd.read_csv(mention_index_path, sep="\t", dtype=str, encoding="utf-8")


def write_mention_index(mention_index, mention_index_path, overwrite=False):
  """Write the mention index dataframe to a TSV file."""
  if not overwrite:
    assert not os.path.exists(mention_index_path), mention_index_path

  logging.info("Writing mention index to: [%s]", mention_index_path)
  mention_index.to_csv(
      mention_index_path,
      sep="\t",
      encoding="utf-8",
      columns=["docid", "position", "length", "mention", "url"],
      index=False)


def add_text_hash_column(docid_to_text_hash, doc_index):
  doc_index["text_md5"] = doc_index.index.to_series().map(
      docid_to_text_hash.get)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  language = FLAGS.language
  archive_parser = common.WikiNewsArchiveParser(FLAGS.wikinews_archive,
                                                language)
  archive_consumer = common.WikiNewsArchiveConsumer(archive_parser,
                                                    FLAGS.output_dir_wiki,
                                                    FLAGS.max_docs,
                                                    FLAGS.max_docs)
  wiki_doc_parser = common.WikiNewsDocParser(language)

  if FLAGS.mode == "raw":
    # Create document index and extract wiki_docs from all documents available
    # in archive.
    doc_index = archive_consumer.extract_docs()
    logging.info("Document index contains %d records.", doc_index.shape[0])
    logging.info(doc_index.head())

    # Extract mentions by parsing wiki_docs.
    mention_index, docid_to_text_hash = extract_text_and_mentions(
        doc_index, wiki_doc_parser, FLAGS.output_dir_wiki,
        FLAGS.output_dir_text)
    add_text_hash_column(docid_to_text_hash, doc_index)
    logging.info("Mention index contains %d records.", mention_index.shape[0])

    # Output.
    common.write_doc_index(doc_index, FLAGS.doc_index_path)
    write_mention_index(mention_index, FLAGS.mention_index_path)
  elif FLAGS.mode == "dataset":
    # Load provided document index.
    doc_index = common.load_doc_index(FLAGS.doc_index_path)
    logging.info("Document index contains %d records.", doc_index.shape[0])
    logging.info(doc_index.head())

    # Extract only the relevant wiki_docs from archive.
    fresh_doc_index = archive_consumer.extract_docs(doc_index)

    # Load provided mention index
    mention_index = load_mention_index(FLAGS.mention_index_path)

    # Recover clean text for the relevant by parsing the relevant wiki_docs.
    # The documents are constrained to the provided doc_index. The mentions
    # found may be a superset of the provided mention_index.
    fresh_mention_index, docid_to_text_hash = extract_text_and_mentions(
        doc_index, wiki_doc_parser, FLAGS.output_dir_wiki,
        FLAGS.output_dir_text)
    add_text_hash_column(docid_to_text_hash, fresh_doc_index)

    logging.info("Extracted potential contexts for %d mentions.",
                 fresh_mention_index.shape[0])

    if (verify_doc_index(doc_index, fresh_doc_index) and
        verify_mention_index(mention_index, fresh_mention_index) and
        verify_context(mention_index, FLAGS.output_dir_text)):
      print(
          "Success! Found all {count} mentions and contexts across {doc_count} "
          "documents. Make use of the following:\n"
          "\t{m_idx}\n\t{d_idx}\n\t{txt}".format(
              count=mention_index.shape[0],
              doc_count=doc_index.shape[0],
              m_idx=FLAGS.mention_index_path,
              d_idx=FLAGS.doc_index_path,
              txt=os.path.join(FLAGS.output_dir_text, "*")))
    else:
      print("Fail! Extracted mentions and contexts do not match all "
            "expectations. (Run with --logtostderr for details.)")

  else:
    raise ValueError("Invalid mode %s", FLAGS.mode)


if __name__ == "__main__":

  flags.mark_flag_as_required("language")
  flags.mark_flag_as_required("wikinews_archive")
  flags.mark_flag_as_required("output_dir_wiki")
  flags.mark_flag_as_required("output_dir_text")
  flags.mark_flag_as_required("doc_index_path")
  flags.mark_flag_as_required("mention_index_path")
  app.run(main)
