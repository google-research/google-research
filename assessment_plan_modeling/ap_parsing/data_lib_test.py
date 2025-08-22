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

"""Tests for data_lib."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import numpy as np
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import augmentation_lib as aug_lib
from assessment_plan_modeling.ap_parsing import data_lib
from assessment_plan_modeling.ap_parsing import tokenizer_lib
from assessment_plan_modeling.note_sectioning import note_section_lib


class APParsingDataLibTest(tf.test.TestCase):

  def test_get_converted_labels(self):
    ap_text = "\n".join([
        "50 yo m with hx of copd, dm2", "#. COPD ex: started on abx in ED.",
        "  - continue abx."
    ])
    tokens = tokenizer_lib.tokenize(ap_text)

    labels = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=32,
            end_char=39),  # span_text="COPD ex"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
            start_char=41,
            end_char=63),  # span_text="started on abx in ED.\n"
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            start_char=67,
            end_char=80,
            action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS
        ),  # span_text="continue abx."
    ]

    converted_labels = data_lib.generate_model_labels(labels, tokens)

    expected_fragment_labels = np.zeros(45)
    expected_fragment_labels[21] = 1  # B-PT COPD ex
    expected_fragment_labels[22:24] = 2  # I-PT COPD ex

    expected_fragment_labels[26] = 3  # B-PD started on abx in ED
    expected_fragment_labels[27:35] = 4  # I-PD started on abx in ED

    expected_fragment_labels[41] = 5  # B-AI continue abx
    expected_fragment_labels[42:44] = 6  # I-AI continue abx

    expected_ai_labels = np.zeros(45)
    expected_ai_labels[41:44] = 1  # continue abx - medications

    self.assertAllEqual(converted_labels["fragment_type"],
                        expected_fragment_labels)
    self.assertAllEqual(converted_labels["action_item_type"],
                        expected_ai_labels)

  def test_get_token_features(self):
    ap_text = "50 yo m with hx of copd, dm2\n#. COPD Ex"

    #        0    1             23456      7    8
    vocab = [" ", "\n"] + list("-:.,#") + ["2", "50"] + [
        "abx",
        "continue",
        "copd",
        "dm",
        "ed",
        "ex",
        "hx",
        "in",
        "m",
        "of",
    ]

    tokens = tokenizer_lib.tokenize(ap_text)

    token_features = data_lib.generate_token_features(tokens, vocab)

    expected_features = {
        #    OOV is 1
        "token_ids": [
            11, 3, 2, 3, 20, 3, 2, 3, 18, 3, 21, 3, 14, 8, 3, 15, 10, 4, 9, 7,
            3, 14, 3, 17
        ],
        "token_type": [
            3, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 2, 5, 1, 3, 5, 2, 2, 5, 1, 5,
            1
        ],
        "is_upper": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0
        ],
        "is_title": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1
        ]
    }

    for key in token_features:
      self.assertAllClose(token_features[key], expected_features[key], msg=key)

  def test_extract_ap_sections(self):
    section_markers = {
        "hpi": ["history of present illness"],
        "a&p": ["assessment and plan"],
    }
    note = data_lib.Note(
        note_id=1,
        text="hpi:\n 50yof with hx of dm2.\na&p:\n # dm2:\n-RISS",
        subject_id=0,
        category="PHYSICIAN")

    expected = [note_section_lib.Section(28, 46, ["assessment and plan"])]
    self.assertEqual(
        list(data_lib.extract_ap_sections(note.text, section_markers)),
        expected)

    # multi section
    note = data_lib.Note(
        note_id=1,
        text="hpi:\n 50yof with hx of dm2.\na&p: DM2\na&p:\n # dm2:\n-RISS",
        subject_id=0,
        category="PHYSICIAN")

    expected = [
        note_section_lib.Section(28, 37, ["assessment and plan"]),
        note_section_lib.Section(37, 55, ["assessment and plan"]),
    ]
    self.assertEqual(
        list(data_lib.extract_ap_sections(note.text, section_markers)),
        expected)

  def test_process_rating_labels(self):
    rating_labels = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=0,
            end_char=50),  # before
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=45,
            end_char=65),  # partially contained
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=50,
            end_char=150),  # exactly matches section
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=100,
            end_char=105),  # contained
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=150,
            end_char=155),  # after
    ]

    expected = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=0,
            end_char=100),
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=50,
            end_char=55)
    ]
    self.assertEqual(
        data_lib.process_rating_labels(rating_labels,
                                       note_section_lib.Section(50, 150, [])),
        expected)


class ProcessAPDataDoFnTest(absltest.TestCase):

  def test_usage(self):
    section_markers = {
        "hpi": ["history of present illness"],
        "a&p": ["assessment and plan"],
    }
    ap_texts = ["a&p:\n # dm2:\n-RISS", "a&p:\n # COPD:\n-nebs"]
    notes_with_ratings = [("0", {
        "notes": [
            data_lib.Note(
                note_id=0,
                text="blablabla\n" + ap_texts[0],
                subject_id=0,
                category="PHYSICIAN")
        ],
        "ratings": [[
            ap_parsing_lib.LabeledCharSpan(
                span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                start_char=19,
                end_char=22),
            ap_parsing_lib.LabeledCharSpan(
                span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
                start_char=24,
                end_char=28)
        ]],
        "note_partition": ["val"]
    })] + [("1", {
        "notes": [
            data_lib.Note(
                note_id=1,
                text="blablabla\n" + ap_texts[1],
                subject_id=1,
                category="PHYSICIAN")
        ],
        "ratings": [],
        "note_partition": []
    })]

    expected = [
        ("0|10",
         data_lib.APData(
             partition=data_lib.Partition.VAL,
             note_id="0",
             subject_id="0",
             ap_text=ap_texts[0],
             char_offset=10,
             tokens=tokenizer_lib.tokenize(ap_texts[0]),
             labeled_char_spans=[
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                     start_char=8,
                     end_char=11),
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                     action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
                     start_char=14,
                     end_char=18)
             ])),
        ("1|10",
         data_lib.APData(
             partition=data_lib.Partition.NONRATED,
             note_id="1",
             subject_id="1",
             ap_text=ap_texts[1],
             char_offset=10,
             tokens=tokenizer_lib.tokenize(ap_texts[1]),
             labeled_char_spans=[
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                     start_char=8,
                     end_char=12),
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                     start_char=15,
                     end_char=19)
             ]))
    ]
    with test_pipeline.TestPipeline() as p:
      results = (
          p
          | beam.Create(notes_with_ratings)
          | beam.ParDo(
              data_lib.ProcessAPData(filter_inorganic_threshold=0),
              section_markers))
      util.assert_that(results, util.equal_to(expected))

  def test_multiratings(self):
    section_markers = {
        "hpi": ["history of present illness"],
        "a&p": ["assessment and plan"],
    }
    ap_text = "a&p:\n # dm2:\n-RISS"
    notes_with_ratings = [("0", {
        "notes": [
            data_lib.Note(
                note_id=0,
                text="blablabla\n" + ap_text,
                subject_id=0,
                category="PHYSICIAN")
        ],
        "ratings":
            [[
                ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                    start_char=19,
                    end_char=22),
                ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    start_char=24,
                    end_char=28)
            ],
             [
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                     start_char=18,
                     end_char=22),
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                     start_char=25,
                     end_char=28)
             ]],
        "note_partition": ["test", "test"]
    })]

    expected = [
        ("0|10",
         data_lib.APData(
             partition=data_lib.Partition.TEST,
             note_id="0",
             subject_id="0",
             ap_text=ap_text,
             char_offset=10,
             tokens=tokenizer_lib.tokenize(ap_text),
             labeled_char_spans=[
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                     start_char=8,
                     end_char=11),
                 ap_parsing_lib.LabeledCharSpan(
                     span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                     start_char=14,
                     end_char=18)
             ]))
    ] * 2
    with test_pipeline.TestPipeline() as p:
      results = (
          p
          | beam.Create(notes_with_ratings)
          | beam.ParDo(
              data_lib.ProcessAPData(filter_inorganic_threshold=0),
              section_markers))
      util.assert_that(results, util.equal_to(expected))


class ApplyAugmentationsDoFnTest(absltest.TestCase):

  def test_usage(self):
    augmentation_config = aug_lib.AugmentationConfig(
        augmentation_sequences=[
            aug_lib.AugmentationSequence(
                name="test",
                augmentation_sequence=[
                    aug_lib.ChangeDelimAugmentation(
                        fragment_types=[
                            ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE
                        ],
                        delims=["\n"])
                ])
        ],
        augmentation_number_deterministic=1)

    ap_data = [
        (
            "0|10",
            data_lib.APData(
                note_id=0,
                subject_id=0,
                ap_text="a&p:\n # dm2:\n-RISS",
                labeled_char_spans=[
                    ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                        start_char=8,
                        end_char=11),  # span_text="dm2",
                    ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                        action_item_type=ap_parsing_lib.ActionItemType
                        .MEDICATIONS,
                        start_char=14,
                        end_char=18)  # span_text="RISS",
                ])),
    ]
    expected = [
        *ap_data,
        (
            "0|10",
            data_lib.APData(
                note_id=0,
                subject_id=0,
                ap_text="a&p\ndm2:\n- RISS",
                tokens=tokenizer_lib.tokenize("a&p\ndm2:\n- RISS"),
                labeled_char_spans=[
                    ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                        start_char=4,
                        end_char=7),  # span_text="dm2",
                    ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                        action_item_type=ap_parsing_lib.ActionItemType
                        .MEDICATIONS,
                        start_char=11,
                        end_char=15)  # span_text="RISS",
                ],
                augmentation_name="test")),
    ]

    with test_pipeline.TestPipeline() as p:
      results = (
          p
          | beam.Create(ap_data)
          | beam.ParDo(data_lib.ApplyAugmentations(), augmentation_config))
      util.assert_that(results, util.equal_to(expected))


if __name__ == "__main__":
  absltest.main()
