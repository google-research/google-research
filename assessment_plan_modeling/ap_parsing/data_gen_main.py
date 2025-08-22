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

r"""Beam pipeline to generate tf.train.Examples for AP Parsing.

Run with:
DATADIR="path/to/data" && \
python assessment_plan_modeling/ap_parsing/data_gen_main.py \
    --input_note_events="${DATADIR}/note_events.csv" \
    --output_path="${DATADIR}/ap_parsing_tf_examples/$(date +%Y%m%d)" \
    --input_ratings="${DATADIR}/all_model_ratings.csv" \
    --vocab_file="${DATADIR}/word_vocab_25K.txt" \
    --section_markers=""
    --n_downsample=100 \
    --max_seq_length=2048

"""

from typing import Any, Dict, List, Sequence, Tuple

from absl import app
from absl import flags
import apache_beam as beam

from assessment_plan_modeling.ap_parsing import augmentation_lib as aug_lib
from assessment_plan_modeling.ap_parsing import data_lib
from assessment_plan_modeling.ap_parsing.configs import augmentation_config
from assessment_plan_modeling.note_sectioning import note_section_lib

_VOCAB_PATH = flags.DEFINE_string(
    "vocab_file",
    None, ("Text file with a single vocab token per line. "
           "Expected to have the same number of rows as the vocabulary size"),
    required=True)
_SECTION_MARKERS_PATH = flags.DEFINE_string(
    "section_markers", None, "JSON of sectioning markers.", required=True)
_INPUT_NOTE_EVENTS = flags.DEFINE_string(
    "input_note_events",
    None,
    "CSV containing mimic3 note events.",
    required=True)
_INPUT_RATINGS = flags.DEFINE_string(
    "input_ratings", None, "CSV containing ratings.", required=True)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None, ("Output directory path, "
           "where train, val and test sets will be stored "
           "(as sharded TFRecords of tf.Examples)."),
    required=True)
_RANDOM_SEED = flags.DEFINE_integer("random_seed", 0, "")
_MAX_SEQ_LENGTH = flags.DEFINE_integer(
    "max_seq_length", 1024, ("Max sequence length for ap sections (in tokens), "
                             "Larger sections are dropped, "
                             "smaller are post-padded to size."))
_N_DOWNSAMPLE = flags.DEFINE_integer("n_downsample", 25_000,
                                     ("Number of non-rated notes to process. "
                                      "Downsampled from all suitable notes."))
_DEBUG_OUTPUT = flags.DEFINE_bool(
    "debug_output", False,
    ("Whether to add additional debug output shuch as the text, "
     "tokens etc. in the tf.Examples."))


def run_pipeline(root, input_note_events,
                 input_ratings, output_path, vocab,
                 section_markers,
                 cur_augmentation_config):
  """Create beam pipeline to generate TF examples.

  Args:
    root: beam.Pipeline root.
    input_note_events: Path to csv of notes.
    input_ratings: Path to csv of ratings.
    output_path: Directory path to write output to.
    vocab: List of tokens in the vocabulary.
    section_markers: Dict of markers as accepted by note sectioning.
    cur_augmentation_config: AugmentationConfig dataclass instance, defines the
      kinds of augmentations to apply.
  """

  # Load and process ratings:
  raw_ratings = data_lib.read_raw_ratings(root, input_ratings)

  ratings = (
      raw_ratings
      | "GetLabels" >> beam.Map(data_lib.convert_ratings)
      | "GroupRatingsByNoteId" >> beam.GroupByKey()
      | "UnpackRatings" >> beam.Map(lambda x: (x[0], list(x[1]))))

  # Load and process notes:
  notes = data_lib.read_filter_notes(root, input_note_events)

  note_partitions = (
      raw_ratings
      | "PartitionMap" >>
      (beam.Map(lambda x: (str(x.note_id), x.partition))).with_output_types(
          Tuple[str, str])
      | "DedupPartitionMap" >> beam.Distinct())

  # Join.
  non_rated_notes, rated_notes = (
      ({
          "ratings": ratings,
          "notes": notes,
          "note_partition": note_partitions
      })
      | "Join" >> beam.CoGroupByKey().with_output_types(Tuple[str, Dict[str,
                                                                        Any]])
      | "SplitRated" >> beam.Partition(
          lambda x, n_part: int(bool(x[1]["ratings"])), 2))

  # Downsample non-rated.
  non_rated_notes = data_lib.downsample(non_rated_notes, _N_DOWNSAMPLE.value,
                                        _RANDOM_SEED.value)

  # Process notes.
  features_and_labels = (
      (non_rated_notes, rated_notes)
      | beam.Flatten()
      | "ReshuffleJoin" >> beam.Reshuffle()
      | "ProcessAPData" >> beam.ParDo(data_lib.ProcessAPData(), section_markers)
      | "FilterAPData" >> beam.Filter(data_lib.filter_by_labels)
      | "ReshuffleForSubjectId" >> beam.Reshuffle()
      | "RekeyBySubjectId" >> beam.Map(lambda x: (x[1].subject_id, x[1]))
      | "GroupBySubjectId" >> beam.GroupByKey()
      | "OneNoteIdPerRatedSubjectId" >> beam.ParDo(
          data_lib.OneNoteIdPerRatedSubjectId(), seed=_RANDOM_SEED.value)
      | "RekeyByNoteId" >> beam.Map(lambda x: (x.note_id, x))
      | "ApplyAugmentations" >> beam.ParDo(data_lib.ApplyAugmentations(),
                                           cur_augmentation_config,
                                           _RANDOM_SEED.value)
      | "GetFeaturesAndLabels" >> beam.ParDo(
          data_lib.ProcessFeaturesAndLabels(vocab, _MAX_SEQ_LENGTH.value))
      | "ReshuffleFeaturesAndLabels" >> beam.Reshuffle())

  # Convert and save tf examples:
  data_lib.convert_and_save_tf_examples(features_and_labels, output_path,
                                        _DEBUG_OUTPUT.value)


def main(unused_argv):
  del unused_argv

  # Read vocab and section markers
  vocab = data_lib.read_vocab(_VOCAB_PATH.value)
  section_markers = note_section_lib.get_markers(_SECTION_MARKERS_PATH.value)

  cur_augmentation_config = augmentation_config.DEFAULT_AUGMENTATION_CONFIG

  # Run pipeline:
  with beam.Pipeline() as root:
    run_pipeline(root, _INPUT_NOTE_EVENTS.value, _INPUT_RATINGS.value,
                 _OUTPUT_PATH.value, vocab, section_markers,
                 cur_augmentation_config)


if __name__ == "__main__":
  app.run(main)
