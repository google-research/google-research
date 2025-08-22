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

"""Tests for augmentation_lib."""

from typing import Tuple
from absl.testing import absltest
from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import augmentation_lib as aug_lib


def tuple_fragment(fragment):
  return (str(fragment.labeled_char_span), fragment.text, fragment.prefix_delim,
          fragment.suffix_delim)


def fragments_tuple(cluster):
  return tuple(
      set([tuple_fragment(fragment) for fragment in cluster.fragments]))


class StructuredAPTest(absltest.TestCase):

  def test_build(self):
    ap = "\n".join(
        ["50 yo m with hx of dm2, copd", "dm2: on insulin", "- RISS"])
    labeled_char_spans = [
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            start_char=29,
            end_char=32),
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
            start_char=34,
            end_char=44),
        ap_parsing_lib.LabeledCharSpan(
            span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
            start_char=47,
            end_char=51),
    ]

    expected = aug_lib.StructuredAP(
        prefix_text="50 yo m with hx of dm2, copd",
        problem_clusters=[
            aug_lib.ProblemCluster(fragments=[
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                        start_char=29,
                        end_char=32),
                    text="dm2",
                    prefix_delim=aug_lib._DefaultDelims.PROBLEM_TITLE_PREFIX,
                    suffix_delim=aug_lib._DefaultDelims.PROBLEM_TITLE_SUFFIX),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType
                        .PROBLEM_DESCRIPTION,
                        start_char=34,
                        end_char=44),
                    text="on insulin",
                    prefix_delim=aug_lib._DefaultDelims
                    .PROBLEM_DESCRIPTION_PREFIX,
                    suffix_delim=""),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                        start_char=47,
                        end_char=51),
                    text="RISS",
                    prefix_delim=aug_lib._DefaultDelims.ACTION_ITEM_PREFIX,
                    suffix_delim=""),
            ])
        ])

    structured_ap = aug_lib.StructuredAP.build(ap, labeled_char_spans)
    self.assertEqual(structured_ap, expected)

  def test_compile(self):
    structured_ap = aug_lib.StructuredAP(
        prefix_text="50 yo m with hx of dm2, copd",
        problem_clusters=[
            aug_lib.ProblemCluster(fragments=[
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                        start_char=29,
                        end_char=32),
                    text="dm2",
                    prefix_delim="\n*. ",
                    suffix_delim=": "),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType
                        .PROBLEM_DESCRIPTION,
                        start_char=34,
                        end_char=44),
                    text="on insulin",
                    prefix_delim="",
                    suffix_delim=""),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                        start_char=47,
                        end_char=51),
                    text="RISS",
                    prefix_delim="\n- ",
                    suffix_delim=""),
            ])
        ])

    expected = "50 yo m with hx of dm2, copd\n*. dm2: on insulin\n- RISS"
    result, _ = structured_ap.compile()
    self.assertEqual(result, expected)

  def test_compile_with_labels(self):
    structured_ap = aug_lib.StructuredAP(
        prefix_text="50 yo m with hx of dm2, copd",
        problem_clusters=[  # spans are kept from *original* text.
            aug_lib.ProblemCluster(fragments=[
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                        start_char=29,
                        end_char=32),
                    text="dm2",
                    prefix_delim="\n*. ",
                    suffix_delim=": "),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType
                        .PROBLEM_DESCRIPTION,
                        start_char=34,
                        end_char=44),
                    text="on insulin",
                    prefix_delim="",
                    suffix_delim=""),
                aug_lib.ProblemClusterFragment(
                    labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                        span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                        start_char=47,
                        end_char=51),
                    text="RISS",
                    prefix_delim="\n- ",
                    suffix_delim=""),
            ])
        ])

    expected = (
        "50 yo m with hx of dm2, copd\n*. dm2: on insulin\n- RISS",
        [
            ap_parsing_lib.LabeledCharSpan(
                span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                start_char=32,
                end_char=35),  # span_text="dm2"
            ap_parsing_lib.LabeledCharSpan(
                span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
                start_char=37,
                end_char=47),  # span_text="on insulin"
            ap_parsing_lib.LabeledCharSpan(
                span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                start_char=50,
                end_char=54),  # span_text="RISS"
        ])
    result_ap_text, result_labeled_char_spans = structured_ap.compile()
    self.assertEqual((result_ap_text, result_labeled_char_spans), expected)


class AugmentationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.problem_clusters = [
        aug_lib.ProblemCluster(fragments=[
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                    start_char=29,
                    end_char=32),
                text="dm2",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType
                    .PROBLEM_DESCRIPTION,
                    start_char=34,
                    end_char=44),
                text="on insulin",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
                    start_char=47,
                    end_char=51),
                text="RISS",
                prefix_delim="",
                suffix_delim="")
        ]),
        aug_lib.ProblemCluster(fragments=[
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                    start_char=52,
                    end_char=58),
                text="anemia",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType
                    .OBSERVATIONS_LABS,
                    start_char=59,
                    end_char=64),
                text="trend",
                prefix_delim="",
                suffix_delim="")
        ]),
        aug_lib.ProblemCluster(fragments=[
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                    start_char=65,
                    end_char=69),
                text="COPD",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
                    start_char=70,
                    end_char=74),
                text="nebs",
                prefix_delim="",
                suffix_delim="")
        ]),
        aug_lib.ProblemCluster(fragments=[
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
                    start_char=75,
                    end_char=81),
                text="sepsis",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType
                    .PROBLEM_DESCRIPTION,
                    start_char=82,
                    end_char=93),
                text="dd pna, uti",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType
                    .PROBLEM_DESCRIPTION,
                    start_char=94,
                    end_char=117),
                text="yesterday without fever",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType.MEDICATIONS,
                    start_char=118,
                    end_char=127),
                text="cont. abx",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType
                    .OBSERVATIONS_LABS,
                    start_char=128,
                    end_char=131),
                text="cis",
                prefix_delim="",
                suffix_delim=""),
            aug_lib.ProblemClusterFragment(
                labeled_char_span=ap_parsing_lib.LabeledCharSpan(
                    span_type=ap_parsing_lib.LabeledSpanType.ACTION_ITEM,
                    action_item_type=ap_parsing_lib.ActionItemType.CONSULTS,
                    start_char=132,
                    end_char=142),
                text="id consult",
                prefix_delim="",
                suffix_delim="")
        ])
    ]
    self.ap = aug_lib.StructuredAP(
        problem_clusters=self.problem_clusters, prefix_text="")

  def test_shuffle_clusters(self):
    aug = aug_lib.ShuffleClusters()

    augmented_ap = aug(self.ap, seed=0)
    set_problem_clusters = set(
        [fragments_tuple(cluster) for cluster in self.problem_clusters])
    set_aug_clusters = set(
        [fragments_tuple(cluster) for cluster in augmented_ap.problem_clusters])
    self.assertEqual(set_problem_clusters, set_aug_clusters)

  def test_shuffle_fragments(self):
    aug = aug_lib.ShuffleFragments()

    augmented_ap = aug(self.ap, seed=0)
    self.assertEqual(
        fragments_tuple(self.problem_clusters[0]),
        fragments_tuple(augmented_ap.problem_clusters[0]))

  def test_number_title_augmentation(self):
    aug = aug_lib.NumberTitlesAugmentation(["\n{:d})"])

    augmented_ap = aug(self.ap, seed=0)
    expected = self.ap
    for i, cluster in enumerate(expected.problem_clusters):
      cluster.fragments[0].prefix_delim = f"\n{i+1})"
    self.assertEqual(expected, augmented_ap)

  def test_change_delim_augmentation(self):
    aug = aug_lib.ChangeDelimAugmentation(
        fragment_types=[
            ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE,
            ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION,
            ap_parsing_lib.LabeledSpanType.ACTION_ITEM
        ],
        delims=["*"])

    augmented_ap = aug(self.ap, seed=0)
    expected = self.ap
    for cluster in expected.problem_clusters:
      for fragment in cluster.fragments:
        fragment.prefix_delim = "*"

    self.assertEqual(expected, augmented_ap)

  def test_apply_augmentations(self):
    augs = aug_lib.AugmentationSequence(
        name="test",
        augmentation_sequence=[
            aug_lib.NumberTitlesAugmentation(["\n{}."]),
            aug_lib.ChangeDelimAugmentation(
                [ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION],
                ["\n-- "]),
            aug_lib.ChangeDelimAugmentation(
                [ap_parsing_lib.LabeledSpanType.ACTION_ITEM], ["\n--- "])
        ])

    results = aug_lib.apply_augmentations(self.ap, augs, seed=0)
    expected = self.ap
    for i, cluster in enumerate(expected.problem_clusters):
      for fragment in cluster.fragments:
        if fragment.labeled_char_span.span_type == ap_parsing_lib.LabeledSpanType.PROBLEM_TITLE:
          prefix_delim = f"\n{i+1}."
        elif fragment.labeled_char_span.span_type == ap_parsing_lib.LabeledSpanType.PROBLEM_DESCRIPTION:
          prefix_delim = "\n-- "
        elif fragment.labeled_char_span.span_type == ap_parsing_lib.LabeledSpanType.ACTION_ITEM:
          prefix_delim = "\n--- "
        fragment.prefix_delim = prefix_delim

    self.assertEqual(expected, results)


if __name__ == "__main__":
  absltest.main()
