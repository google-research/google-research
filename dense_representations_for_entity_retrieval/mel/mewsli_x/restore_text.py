# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Restore WikiNews article text for Mewsli-X dataset.

Run this tool to insert WikiNews article text back into the released Mewsli-X
data files wikinews_mentions_no_text-*.jsonl.

The tool assumes you have already extracted and parsed the article texts from
public WikiNews dumps using the other scripts in this directory.
"""

import pathlib
import typing
from typing import Dict, List, Sequence

from absl import app
from absl import flags
from absl import logging

from dense_representations_for_entity_retrieval.mel.mewsli_x import io_util
from dense_representations_for_entity_retrieval.mel.mewsli_x import schema

ContextualMentions = schema.ContextualMentions
JsonDict = schema.JsonDict

INDEX_DIR = flags.DEFINE_string(
    "index_dir",
    None,
    "Path to base directory, such that the WikiNews article text for a given "
    "document ID is the text-file {index_dir}/{language}/text/{docid}.",
    required=True)

INPUT = flags.DEFINE_string(
    "input",
    None,
    "Path to Mewsli-X WikiNews .jsonl, where each line contains a serialized "
    "`schema.ContextualMentions` with empty `context.text` attributes.",
    required=True)

OUTPUT = flags.DEFINE_string(
    "output",
    None, "Output path. Each line will contain a serialized "
    "`schema.ContextualMentions` with `context.text` containing the restored "
    "WikiNews article text.",
    required=True)


def _load_text(
    input_path  # pylint: disable=g-bare-generic
):
  logging.log(logging.DEBUG, "Read %s", input_path)
  return schema.load_text(input_path)


def _restore_text(
    data_dicts,
    base_input_dir  # pylint: disable=g-bare-generic
):
  """Creates ContextualMentions by loading article text from files.

  Args:
    data_dicts: Each item is a nested dictionary structured like
      `ContextualMentions`, covering one document each, and
      item["context"]["text"] must be an empty string.
    base_input_dir: Path to base directory where the article text is available
      as text files with the naming scheme {base_input_dir}/{language}/{docid}.

  Returns:
    A list of validated `ContextualMentions`, with `context.text` repopulated.
  """
  outputs = []
  for document in data_dicts:
    docid = document["context"]["document_id"]
    logging.log(logging.DEBUG, "Process doc %s", docid)
    if "text" in document["context"] and document["context"]["text"]:
      raise ValueError("Unexpected text encountered in data dictionary for "
                       f"document id '{docid}': {document['context']['text']}")

    language = document["context"]["language"]
    text_to_restore = _load_text(base_input_dir / language / "text" / docid)

    # Add the recovered text to the structured dictionary. (Needs explicit cast,
    # since the type-checker cannot infer the correct type inside the Union.)
    context = typing.cast(Dict[str, str], document["context"])
    context["text"] = text_to_restore

    # Span information is automatically validated against the article text when
    # constructing the ContextualMentions instance.
    try:
      outputs.append(ContextualMentions.from_json(document))
    except ValueError:
      print(f"Validation error for document id '{docid}'; record: {document}")
      raise

  return outputs


def restore_text(
    input_path,  # pylint: disable=g-bare-generic
    index_dir,  # pylint: disable=g-bare-generic
    output_path  # pylint: disable=g-bare-generic
):
  """Restores the text."""
  # Load the data without parsing it to ContextualMentions at first, since
  # the automatic span validation requires the text to be present.
  print(f"Restoring text for {input_path}")
  data: List[JsonDict] = schema.load_jsonl_as_dicts(input_path)

  restored_contextual_mentions: List[ContextualMentions] = _restore_text(
      data, index_dir)

  print(f"Writing to {output_path}")
  schema.write_jsonl(output_path, restored_contextual_mentions)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  output_path = pathlib.Path(OUTPUT.value)
  io_util.make_dirs(output_path.parent)
  restore_text(
      input_path=pathlib.Path(INPUT.value),
      index_dir=pathlib.Path(INDEX_DIR.value),
      output_path=output_path)


if __name__ == "__main__":
  app.run(main)
