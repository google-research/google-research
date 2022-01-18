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

# Lint as: python3
"""Tests for wt5.metrics."""

from absl.testing import absltest

from t5.evaluation import test_utils
from wt5.wt5 import metrics


class MetricsTest(test_utils.BaseMetricsTest):

  def test_esnli_metric(self):
    ref = "this is a string"
    ref2 = "this is another string"
    self.assertDictClose(
        metrics.esnli_metric(
            [{"label": 0, "explanations": [ref]},
             {"label": 1, "explanations": [ref]},
             {"label": 2, "explanations": [ref2]}],
            [{"label": 0, "explanations": [ref]},
             {"label": 1, "explanations": [ref]},
             {"label": 2, "explanations": [ref2]}]),
        {"accuracy": 100., "bleu": 100., "expln1_length": 16}
        )

  def test_esnli_metric_empty_explanations(self):
    ref = "this is a string"
    ref2 = "this is another string"
    self.assertDictClose(
        metrics.esnli_metric(
            [{"label": 0, "explanations": [ref]},
             {"label": 1, "explanations": []},
             {"label": 2, "explanations": [ref2]}],
            [{"label": 0, "explanations": [ref]},
             {"label": 1, "explanations": []},
             {"label": 2, "explanations": [ref2]}]),
        {"accuracy": 100., "bleu": 100., "expln1_length": 16}
        )

  def test_esnli_metric_multiple_explanations(self):
    ref = "this is a string"
    ref2 = "this is another string"
    self.assertDictClose(
        metrics.esnli_metric(
            [{"label": 0, "explanations": [ref, ""]},
             {"label": 1, "explanations": ["", ref]},
             {"label": 2, "explanations": ["", ref2]}],
            [{"label": 0, "explanations": [ref]},
             {"label": 1, "explanations": [ref]},
             {"label": 2, "explanations": [ref2]}]),
        {"accuracy": 100., "bleu": 100., "expln1_length": 16}
        )

  def test_esnli_metric_multiple_predicted_explanations(self):
    ref = "this is a string"
    self.assertDictClose(
        metrics.esnli_metric(
            [{"label": 0, "explanations": [ref, ""]}],
            [{"label": 0, "explanations": [ref, ref]}],
        ),
        {"accuracy": 100., "bleu": 100., "expln1_length": 16}
    )

  def test_extractive_explanations_metric(self):
    self.assertDictClose(
        metrics.extractive_explanations_metric([
            {
                "label": 0,
                "overlap_spans": [(1, 2), (3, 4)],
                "span_array": [0, 0, 0],
            },
            {
                "label": 1,
                "overlap_spans": [(13, 16), (24, 28)],
                "span_array": [0, 0, 1, 1],
            },
        ], [{
            "label": 1,
            "overlap_spans": [(7, 10)],
            "span_array": [1, 1, 1],
            "explanations": ["a", "b", "c"],
        }, {
            "label": 1,
            "overlap_spans": [(13, 16), (24, 28)],
            "span_array": [0, 0, 1, 1],
            "explanations": ["a", "b"],
        }]), {
            "accuracy": 50.,
            "f1": 50.,
            "partial match f1": 50.,
            "avg_explanation_count": 2.5,
        })

  def test_extractive_explanations_metric_zero_positives(self):
    self.assertDictClose(
        metrics.extractive_explanations_metric(
            [{"label": 1, "overlap_spans": [(7, 10)],
              "span_array": []}],
            [{"label": 0, "overlap_spans": [], "span_array": []}]),
        {"accuracy": 0., "f1": 0.0, "partial match f1": 0.0,
         "avg_explanation_count": 0.0})

  def test_extractive_explanations_metric_partial_match(self):
    self.assertDictClose(
        metrics.extractive_explanations_metric([{
            "label": 1,
            "overlap_spans": [(10, 15), (15, 20)],
            "span_array": [0, 0, 1, 1],
        }], [{
            "label": 1,
            "overlap_spans": [],
            "span_array": [1, 0, 0, 1],
            "explanations": ["a", "b"],
        }]), {
            "accuracy": 100.,
            "f1": 0.0,
            "partial match f1": 50.,
            "avg_explanation_count": 2.,
        })


if __name__ == "__main__":
  absltest.main()
