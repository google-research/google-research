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

"""Tests for module model_performance_analysis.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import model_performance_analysis
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import test_util
import util

TEST_FIXTURE_TABLE_NAME = "test_table_fixture"

FLAGS = flags.FLAGS


def _write_to_file(contents, directory):
  tmpfile_name = tempfile.mktemp(dir=directory)
  with tf.io.gfile.GFile(tmpfile_name, "w") as f:
    f.write(contents)
  return tmpfile_name


class ModelPerformanceAnalysisTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no true label has a clan",
          family_to_clan_mapping={},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [],  # Empty DataFrame.
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name="accurate prediction, family has a clan",
          family_to_clan_mapping={"PF12345": "CL9876"},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name=("inaccurate prediction, actual has a clan, "
                         "predicted does not"),
          family_to_clan_mapping={"PF12345": "CL9876"},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF99999", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", None, 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name="two accurate predictions",
          family_to_clan_mapping={"PF12345": "CL9876"},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5],
               ["C3Y9H3_BRAFL/10-173", "PF12345", "PF12345", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5],
               ["C3Y9H3_BRAFL/10-173", "CL9876", "CL9876", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name=("one accurate prediction,"
                         "one incorrect prediction, all have clans"),
          family_to_clan_mapping={
              "PF12345": "CL9876",
              "PF99999": "CL5555"
          },
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5],
               ["C3Y9H3_BRAFL/10-173", "PF12345", "PF99999", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5],
               ["C3Y9H3_BRAFL/10-173", "CL9876", "CL5555", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name=("one accurate prediction, one incorrect prediction, "
                         "incorrect doesn't have clan"),
          family_to_clan_mapping={"PF12345": "CL9876"},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5],
               ["C3Y9H3_BRAFL/10-173", "PF12345", "PF99999", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5],
               ["C3Y9H3_BRAFL/10-173", "CL9876", None, 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name="different family but same clan",
          family_to_clan_mapping={
              "PF12345": "CL9876",
              "PF99999": "CL9876"
          },
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345", "PF12345", 0.5],
               ["C3Y9H3_BRAFL/10-173", "PF12345", "PF99999", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5],
               ["C3Y9H3_BRAFL/10-173", "CL9876", "CL9876", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ),
      dict(
          testcase_name=("accurate prediction, has clan, includes "
                         "version string"),
          family_to_clan_mapping={"PF12345": "CL9876"},
          input_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "PF12345.6", "PF12345.6", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
          expected_predictions=pd.DataFrame(
              [["A4YXG4_BRASO/106-134", "CL9876", "CL9876", 0.5]],
              columns=util.PREDICTION_FILE_COLUMN_NAMES),
      ))
  def testFamilyPredictionsToClanPredictions(
      self, family_to_clan_mapping, input_predictions, expected_predictions):
    actual = model_performance_analysis.family_predictions_to_clan_predictions(
        input_predictions, family_to_clan_mapping)

    np.testing.assert_array_equal(actual.values, expected_predictions.values)

  @parameterized.named_parameters(
      dict(
          testcase_name="all predictions correct, one prediction",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=1.,
      ),
      dict(
          testcase_name="all predictions incorrect, one prediction",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "A4YXG4_BRASO/106-134,PF00999,PF01000"),
          expected_accuracy=0.,
      ),
      dict(
          testcase_name="all predictions correct, two predictions",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "C3Y9H3_BRAFL/10-173,PF00999,PF00999\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=1.,
      ),
      dict(
          testcase_name="one prediction correct, two predictions",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "C3Y9H3_BRAFL/10-173,PF00999,PF01000\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=.5,
      ),
      dict(
          testcase_name=("one class completely right, another completely wrong,"
                         " classes different size"),
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "K7IN40_NASVI/4-163,PF00999,PF01000\n"
                                 "C3Y9H3_BRAFL/10-173,PF00999,PF01000\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=.5,
      ),
  )
  def testMeanPerClassAccuracy(self, input_predictions_csv, expected_accuracy):
    input_predictions_dataframe = pd.read_csv(
        io.StringIO(input_predictions_csv))

    actual = model_performance_analysis.mean_per_class_accuracy(
        input_predictions_dataframe)
    self.assertAlmostEqual(actual, expected_accuracy)

  @parameterized.named_parameters(
      dict(
          testcase_name="all predictions correct, one prediction",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=1.,
      ),
      dict(
          testcase_name="all predictions incorrect, one prediction",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "A4YXG4_BRASO/106-134,PF00999,PF01000"),
          expected_accuracy=0.,
      ),
      dict(
          testcase_name="all predictions correct, two predictions",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "C3Y9H3_BRAFL/10-173,PF00999,PF00999\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=1.,
      ),
      dict(
          testcase_name="one prediction correct, two predictions",
          input_predictions_csv=("sequence_name,true_label,predicted_label\n"
                                 "C3Y9H3_BRAFL/10-173,PF00999,PF01000\n"
                                 "A4YXG4_BRASO/106-134,PF01000,PF01000"),
          expected_accuracy=.5,
      ),
  )
  def testRawUnweightedAccuracy(self, input_predictions_csv, expected_accuracy):
    input_predictions_dataframe = pd.read_csv(
        io.StringIO(input_predictions_csv))

    actual = model_performance_analysis.raw_unweighted_accuracy(
        input_predictions_dataframe)
    self.assertAlmostEqual(actual, expected_accuracy)

  def testNumTrainExamplesPerClass(self):
    actual = model_performance_analysis.num_examples_per_class(
        self.connection, TEST_FIXTURE_TABLE_NAME)
    expected = pd.DataFrame([["PF10417.8", 2], ["PF09876.5", 1]],
                            columns=["family_accession", "num_examples"],
                            dtype=np.uint64)

    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  def testAverageSequenceLengthPerClass(self):
    actual = model_performance_analysis.average_sequence_length_per_class(
        self.connection, TEST_FIXTURE_TABLE_NAME)
    expected = pd.DataFrame(
        [
            # In the test fixture table, the sequences have length 4 and 1.
            ["PF10417.8", 2.5],
            # The only example for this class has sequence length 3.
            ["PF09876.5", 3]
        ],
        columns=["family_accession", "average_length"])

    pd.testing.assert_frame_equal(actual, expected, check_less_precise=True)

  def testFamiliesWithMoreThanNExamples(self):
    input_dataframe = pd.DataFrame(
        [["PF00001", 10], ["PF00002", 1]],
        columns=[
            model_performance_analysis.FAMILY_ACCESSION_KEY,
            model_performance_analysis.NUM_EXAMPLES_KEY
        ])

    expected_output = ["PF00001"]

    actual_dataframe = (
        model_performance_analysis.families_with_more_than_n_examples(
            input_dataframe, 3))

    np.testing.assert_array_equal(actual_dataframe, expected_output)

  def testMeanPerClassAccuracyForLargeClasses(self):
    # Predictions are correct except on the small class.
    input_predictions_dataframe = pd.DataFrame(
        [["A4YXG4_BRASO/106-134", "PF00001", "PF00001", 0.5],
         ["ANOTHER_THING/106-134", "PF00002", "PF00003", 0.5]],
        columns=util.PREDICTION_FILE_COLUMN_NAMES)
    size_of_training_set_by_family = pd.DataFrame(
        [["PF00001", 10], ["PF00002", 1]],
        columns=[
            model_performance_analysis.FAMILY_ACCESSION_KEY,
            model_performance_analysis.NUM_EXAMPLES_KEY
        ])
    expected = 1.

    actual = (
        model_performance_analysis
        .mean_class_per_accuracy_for_only_large_classes(
            input_predictions_dataframe, 2, size_of_training_set_by_family))

    self.assertEqual(actual, expected)

  def testAccuracyByFamily(self):
    input_predictions_dataframe = pd.DataFrame(
        [["A4YXG4_BRASO/106-134", "PF00001", "PF00001", 0.5],
         ["SOMETHING_ELSE/106-134", "PF00001", "PF00001", 0.5],
         ["ANOTHER_THING/106-134", "PF00002", "PF00003", 0.5]],
        columns=util.PREDICTION_FILE_COLUMN_NAMES)
    expected = pd.DataFrame(
        [["PF00001", 1.], ["PF00002", 0.]],
        columns=[util.TRUE_LABEL_KEY, model_performance_analysis.ACCURACY_KEY])

    actual = model_performance_analysis.accuracy_by_family(
        input_predictions_dataframe)
    pd.testing.assert_frame_equal(actual, expected, check_less_precise=True)

  def testAccuracyBySizeOfTrainingSet(self):
    family_predictions = pd.DataFrame(
        [["A4YXG4_BRASO/106-134", "PF00001", "PF00001", 0.5],
         ["SOMETHING_ELSE/106-134", "PF00001", "PF00001", 0.5],
         ["ANOTHER_THING/106-134", "PF00002", "PF00003", 0.5]],
        columns=util.PREDICTION_FILE_COLUMN_NAMES)

    size_of_training_set_by_family = pd.DataFrame(
        [["PF00001", 10], ["PF00002", 1]],
        columns=[
            model_performance_analysis.FAMILY_ACCESSION_KEY,
            model_performance_analysis.NUM_EXAMPLES_KEY
        ])

    # PF00001 has 100% accuracy, and PF00002 has 0%.
    expected = pd.DataFrame([[10, 1.0], [1, 0.0]],
                            columns=[
                                model_performance_analysis.FAMILY_ACCESSION_KEY,
                                model_performance_analysis.ACCURACY_KEY
                            ])

    actual = model_performance_analysis.accuracy_by_size_of_family(
        family_predictions, size_of_training_set_by_family)

    np.testing.assert_allclose(actual, expected)

  def testAccuracyBySequenceLength(self):
    family_predictions = pd.DataFrame(
        [["A4YXG4_BRASO/106-134", "PF00001", "PF00001", 0.5],
         ["SOMETHING_ELSE/106-134", "PF00001", "PF00001", 0.5],
         ["ANOTHER_THING/106-134", "PF00002", "PF00003", 0.5]],
        columns=util.PREDICTION_FILE_COLUMN_NAMES)

    size_of_training_set_by_family = pd.DataFrame(
        [["PF00001", 3.2], ["PF00002", 6]],
        columns=[
            model_performance_analysis.FAMILY_ACCESSION_KEY,
            model_performance_analysis.AVERAGE_SEQUENCE_LENGTH_KEY
        ])

    # PF00001 has 100% accuracy, and PF00002 has 0%.
    expected = pd.DataFrame([[3.2, 1.0], [6, 0.0]],
                            columns=[
                                model_performance_analysis.FAMILY_ACCESSION_KEY,
                                model_performance_analysis.ACCURACY_KEY
                            ])

    actual = model_performance_analysis.accuracy_by_sequence_length(
        family_predictions, size_of_training_set_by_family)

    np.testing.assert_allclose(actual, expected)

  def testPCAEmbeddingForSequences(self):
    num_pca_dims = 2

    class InferrerStub(object):
      """Stub for getting fake activations."""

      def get_activations(self, list_of_seqs):
        len_of_embedding = 21
        np.random.seed(0)
        return np.random.uniform(size=(len(list_of_seqs), len_of_embedding))

    list_of_seqs = ["A", "Y", "D"]
    # To avoid testing the implementation of PCA in sklearn, but to exercise
    # the code path in pca_embedding_for_sequences, we assert the shapes are
    # correct.
    actual = model_performance_analysis.pca_embedding_for_sequences(
        list_of_seqs, InferrerStub(), num_pca_dims)
    expected_shape = (len(list_of_seqs), num_pca_dims)
    self.assertEqual(actual.shape, expected_shape)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty",
          key="key",
          input_df_csv="key",
          expected={},
      ),
      dict(
          testcase_name="one duplicate",
          key="key",
          input_df_csv=("key\n"
                        "val1\n"
                        "val1"),
          expected={2: 1},
      ),
      dict(
          testcase_name="one entry",
          key="key",
          input_df_csv=("key\n"
                        "val1"),
          expected={1: 1},
      ),
      dict(
          testcase_name="more than one entry",
          key="key",
          input_df_csv=("key\n"
                        "val1\n"
                        "val2"),
          expected={1: 2},
      ),
      dict(
          testcase_name="more than one key",
          key="key2",
          input_df_csv=("key1,key2\n"
                        "val1,val3\n"
                        "val2,val3"),
          expected={2: 1},
      ),
  )
  def testGroupBySizeHistogramData(self, key, input_df_csv, expected):
    input_df = pd.read_csv(io.StringIO(input_df_csv))

    actual = model_performance_analysis._group_by_size_histogram_data(
        input_df, key)

    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty predictions dataframe",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF01000,0.5"),
          expected_csv="sequence_name,true_label,predicted_label,score,domain_evalue\n",
      ),
      dict(
          testcase_name="meets threshold",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1e-2"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF01000,0.5"),
          expected_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1e-2"),
      ),
      dict(
          testcase_name="does not meet threshold",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1e-2"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF01000,2.0"),
          expected_csv="sequence_name,true_label,predicted_label,score,domain_evalue\n",
      ),
      dict(
          testcase_name="predicted family is not true family, meets thresh",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF12345.6,1.0,1e-2"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF12345.6,0.5"),
          expected_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF12345.6,1.0,1e-2"),
      ),
      dict(
          testcase_name="two predictions, both meet",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1e-2\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0,1.0"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF01000,0.0"),
          expected_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1e-2\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0,1.0"),
      ),
      dict(
          testcase_name="two predictions, one meets",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,1.0,1.0\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0,1e-2"),
          gathering_thresholds_csv=("true_label,score\n"
                                    "PF01000,50.0"),
          expected_csv=(
              "sequence_name,true_label,predicted_label,score,domain_evalue\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0,1e-2"),
      ),
  )
  def testFilterHmmerFirstPassByGatheringThreshold(self, input_predictions_csv,
                                                   gathering_thresholds_csv,
                                                   expected_csv):
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))

    gathering_thresholds_df = pd.read_csv(io.StringIO(gathering_thresholds_csv))

    expected = pd.read_csv(io.StringIO(expected_csv))

    actual = (
        model_performance_analysis
        ._filter_hmmer_first_pass_by_gathering_threshold(
            input_predictions_df, gathering_thresholds_df))

    np.testing.assert_equal(actual.columns.values, expected.columns.values)
    np.testing.assert_equal(actual.values, expected.values)

  @parameterized.named_parameters(
      dict(
          testcase_name="two predictions on different sequences",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF09999.9,PF09999.9,77298.0\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          family_to_clan={
              "PF09999.9": "CL00001",
              "PF01000.2": "CL00001"
          },
          expected_count=0,
      ),
      dict(
          testcase_name="two predictions on the same seq, both in a clan",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF09999.9,2810.0\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          family_to_clan={
              "PF09999.9": "CL00001",
              "PF01000.2": "CL00001"
          },
          expected_count=1,
      ),
      dict(
          testcase_name="two predictions on the same seq, not in clan",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF09999.9,120398.0\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          family_to_clan={},
          expected_count=0,
      ),
  )
  def testHadMoreThanOnePredictionAndInClan(self, input_predictions_csv,
                                            family_to_clan, expected_count):
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))

    actual_count = (
        model_performance_analysis._had_more_than_one_prediction_and_in_clan(
            input_predictions_df, family_to_clan))
    self.assertEqual(actual_count, expected_count)

  def testFilterHmmerFirstPassByGatheringThresholdFailsRepeatedAccession(self):
    # Gathering thresholds has duplicates.
    gathering_thresholds_csv = ("true_label,score\n"
                                "PF01000,10.0\n"
                                "PF01000,10.0")

    input_predictions_csv = (
        "sequence_name,true_label,predicted_label,score,domain_evalue\n"
        "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0,1e-2")

    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))

    gathering_thresholds_df = pd.read_csv(io.StringIO(gathering_thresholds_csv))

    with self.assertRaisesRegex(ValueError, "duplicated"):
      (model_performance_analysis
       ._filter_hmmer_first_pass_by_gathering_threshold(
           input_predictions_df, gathering_thresholds_df))

  def testFilterHmmerFirstPassByGatheringThresholdFailsKeyError(self):
    gathering_thresholds_csv = ("true_label,score\n" "PF09999,2.0")

    # Predicted label is not in gathering thresholds.
    input_predictions_csv = ("sequence_name,true_label,predicted_label,score\n"
                             "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0")

    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))

    gathering_thresholds_df = pd.read_csv(io.StringIO(gathering_thresholds_csv))

    with self.assertRaisesRegex(KeyError, "PF01000"):
      (model_performance_analysis
       ._filter_hmmer_first_pass_by_gathering_threshold(
           input_predictions_df, gathering_thresholds_df))

  @parameterized.named_parameters(
      dict(
          testcase_name="threshold between two scores, both correct",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,3.4\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          percentile_thresholds=[.5],
          expected_csv=("threshold,precision,recall\n"
                        "3.4,1,1"),
      ),
      dict(
          testcase_name="threshold zero, all correct",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF01000.2,3.4\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          percentile_thresholds=[0],
          expected_csv=("threshold,precision,recall\n"
                        "3.4,1,1"),
      ),
      dict(
          testcase_name="threshold zero, none correct",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF99999.2,3.4"),
          percentile_thresholds=[0],
          expected_csv=("threshold,precision,recall\n"
                        "3.4,0,0"),
      ),
      dict(
          testcase_name="threshold one, all correct",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          percentile_thresholds=[1],
          expected_csv=("threshold,precision,recall\n"
                        "99,1,1"),
      ),
      dict(
          testcase_name="threshold one, none correct",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF99999.2,3.4"),
          percentile_thresholds=[1],
          expected_csv=("threshold,precision,recall\n"
                        "3.4,0,0"),
      ),
      dict(
          testcase_name="multiple thresholds, one wrong with low score",
          input_predictions_csv=(
              "sequence_name,true_label,predicted_label,score\n"
              "K7IN40_NASVI/4-163,PF01000.2,PF99999.2,3.4\n"
              "ABCDEF_NASVI/4-163,PF01000.2,PF01000.2,99.0"),
          percentile_thresholds=[0, .6],
          expected_csv=(
              "threshold,precision,recall\n"
              "3.4,.5,.5\n"  # Includes the incorrect call.
              "99,1,.5"),
      ),
  )
  def testPrecisionRecallDataFrame(self, input_predictions_csv,
                                   percentile_thresholds, expected_csv):
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))

    actual = model_performance_analysis.precision_recall_dataframe(
        input_predictions_df, percentile_thresholds)
    expected = pd.read_csv(io.StringIO(expected_csv))

    np.testing.assert_array_equal(actual.columns.values,
                                  expected.columns.values)
    np.testing.assert_allclose(actual.values, expected.values)

  def testReadBlundellStyleCSV(self):
    content = """sequence_name,true_label,predicted_label,score,domain_evalue
K9QVG5_NOSS7/1-77,PF18480.1,PREDICTED_LABEL,1.0,1e-2
H8GVR0_DEIGI/15-66,TRUE_LABEL,PF12911.7,1.0,1e-2"""

    sequence_names = ["K9QVG5_NOSS7/1-77", "H8GVR0_DEIGI/15-66"]

    actual = model_performance_analysis.read_blundell_style_csv(
        io.StringIO(content), all_sequence_names=sequence_names)

    expected = pd.DataFrame([
        {
            "sequence_name": "H8GVR0_DEIGI/15-66",
            "predicted_label": "PF12911.7",
            "true_label": "TRUE_LABEL",
            "score": 1.0,
            "domain_evalue": 1e-2,
        },
        {
            "sequence_name": "K9QVG5_NOSS7/1-77",
            "predicted_label": "PREDICTED_LABEL",
            "true_label": "PF18480.1",
            "score": 1.0,
            "domain_evalue": 1e-2,
        },
    ])

    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  def testReadBlundellStyleCSVRaisesWithMissingSequences(self):
    content = """sequence_name,true_label,predicted_label,score,domain_evalue
K9QVG5_NOSS7/1-77,PF18480.1,PREDICTED_LABEL,1.0,1e-2"""

    with self.assertRaisesRegex(ValueError, "missing.*missing-seq"):
      model_performance_analysis.read_blundell_style_csv(
          io.StringIO(content),
          all_sequence_names=["K9QVG5_NOSS7/1-77", "missing-seq"])

  def testReadBlundellStyleCSVNaNWithMissingSequences(self):
    content = """sequence_name,true_label,predicted_label,score,domain_evalue
K9QVG5_NOSS7/1-77,PF18480.1,PREDICTED_LABEL,1.0,1e-2"""

    sequence_names = ["K9QVG5_NOSS7/1-77", "H8GVR0_DEIGI/15-66"]

    actual = model_performance_analysis.read_blundell_style_csv(
        io.StringIO(content),
        all_sequence_names=sequence_names,
        raise_if_missing=False)

    expected = pd.DataFrame([
        {
            "sequence_name": "H8GVR0_DEIGI/15-66",
            "predicted_label": np.nan,
            "true_label": np.nan,
            "score": np.nan,
            "domain_evalue": np.nan,
        },
        {
            "sequence_name": "K9QVG5_NOSS7/1-77",
            "predicted_label": "PREDICTED_LABEL",
            "true_label": "PF18480.1",
            "score": 1.0,
            "domain_evalue": 1e-2
        },
    ])

    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  def testReadBlundellStyleCSVRaisesWithExtraSequences(self):
    content = """sequence_name,true_label,predicted_label,score,domain_evalue
K9QVG5_NOSS7/1-77,PF18480.1,PREDICTED_LABEL,1.0,1e-2
EXTRA,PF18480.1,PREDICTED_LABEL,1.0,1e-2
"""
    with self.assertRaisesRegex(ValueError, "extra"):
      model_performance_analysis.read_blundell_style_csv(
          io.StringIO(content), all_sequence_names=["K9QVG5_NOSS7/1-77"])

  def testReadBlundellStyleCSVDeduplicate(self):
    content = """sequence_name,true_label,predicted_label,score,domain_evalue
K9QVG5_NOSS7/1-77,PF18480.1,BAD,0.0,1000.0
K9QVG5_NOSS7/1-77,PF18480.1,GOOD,1.0,1e-2"""

    actual = model_performance_analysis.read_blundell_style_csv(
        io.StringIO(content), deduplicate=True)

    expected = pd.DataFrame([
        {
            "sequence_name": "K9QVG5_NOSS7/1-77",
            "predicted_label": "GOOD",
            "true_label": "PF18480.1",
            "score": 1.0,
            "domain_evalue": 1e-2
        },
    ])

    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  @parameterized.named_parameters(
      dict(
          testcase_name=" one sequence",
          input_predictions_csv=(
              "percent_seq_identity,sequence_name,predicted_label,true_label\n"
              "10.0,SEQ1/9-90,PF00619.21,PF00619.21"),
          distance_metric="percent_seq_identity",
          nbins=1,
          expected={"SEQ1/9-90": pd.Interval(9.9, 10., closed="right")},
      ),
      dict(
          testcase_name=" two sequences, two bins",
          input_predictions_csv=(
              "percent_seq_identity,sequence_name,predicted_label,true_label\n"
              "10.0,SEQ1/9-90,PF00619.21,PF00619.21\n"
              "20.0,SEQ2/9-90,PF00619.21,PF00619.21"),
          distance_metric="percent_seq_identity",
          nbins=2,
          expected={
              "SEQ1/9-90": pd.Interval(9.9, 15., closed="right"),
              "SEQ2/9-90": pd.Interval(15., 20., closed="right")
          },
      ),
      dict(
          testcase_name=" two sequences, one bin",
          input_predictions_csv=(
              "percent_seq_identity,sequence_name,predicted_label,true_label\n"
              "10.0,SEQ1/9-90,PF00619.21,PF00619.21\n"
              "20.0,SEQ2/9-90,PF00619.21,PF00619.21"),
          distance_metric="percent_seq_identity",
          nbins=1,
          expected={
              "SEQ1/9-90": pd.Interval(9.9, 20., closed="right"),
              "SEQ2/9-90": pd.Interval(9.9, 20., closed="right"),
          },
      ),
      dict(
          testcase_name=" one sequence, different metric col name",
          input_predictions_csv=(
              "DIFF_SEQ_ID,sequence_name,predicted_label,true_label\n"
              "20.0,SEQ2/9-90,PF00619.21,PF00619.21"),
          distance_metric="DIFF_SEQ_ID",
          nbins=1,
          expected={"SEQ2/9-90": pd.Interval(19.9, 20.0, closed="right")}),
  )
  def testAccuracyBySeqIdEqualSized(self, input_predictions_csv,
                                    distance_metric, expected, nbins):
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))
    actual = model_performance_analysis.accuracy_by_sequence_identity_equal_sized(
        df=input_predictions_df,
        distance_metric=distance_metric,
        num_bins=nbins)
    self.assertDictEqual(actual, expected)

  def testAccuracyBySeqIdEqualWidth(self):
    input_predictions_csv = (
        "percent_seq_identity,sequence_name,predicted_label,true_label\n"
        "10.0,SEQ10/9-90,PF00619.21,PF00619.21\n"
        "20.0,SEQ20/9-90,PF99999.99,PF00000.00\n"  # Prediction is wrong.
        "25.0,SEQ25/9-90,PF00619.21,PF00619.21\n"
        "30.0,SEQ30/9-90,PF00619.21,PF00619.21\n"
        "40.0,SEQ40/9-90,PF00619.21,PF00619.21")
    distance_metric = "percent_seq_identity"

    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))
    actual = model_performance_analysis.accuracy_by_metric_equal_width(
        df=input_predictions_df,
        metric_col_name=distance_metric,
        min_metric_value=20.,
        max_metric_value=40,
        bin_width=10.)

    expected = {
        # Expect seq id of 10 to be left out because min_seq_id=10.
        "SEQ10/9-90": np.nan,
        # Expect 2 values in first bin (seq identity of 20, 25).
        "SEQ20/9-90": pd.Interval(20.0, 30.0, closed="left"),
        "SEQ25/9-90": pd.Interval(20.0, 30.0, closed="left"),
        "SEQ30/9-90": pd.Interval(30.0, 40.0, closed="left"),
        # Expect seq id of 40 to be left out because the method is
        # right-exclusive.
        "SEQ40/9-90": np.nan
    }

    # Assert dicts equal, allowing for np.nan for left-out sequences.
    self.assertEqual(len(actual), len(expected))
    for key in expected:
      if not isinstance(expected[key], pd.Interval) and np.isnan(expected[key]):
        self.assertTrue(np.isnan(actual[key]))
      else:
        self.assertEqual(expected, actual, "{} != {}".format(actual, expected))

  def testFilterToIsInClan(self):
    input_predictions_csv = ("true_label,predicted_label\n"
                             "PF00619.21,SOMETHING_ELSE\n")
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))
    clan_dict = {"PF00619": "CL00001"}
    expected = input_predictions_csv  # No filtering should be done.
    actual = model_performance_analysis.filter_to_is_in_clan(
        input_predictions_df, clan_dict)
    self.assertEqual(actual.to_csv(index=False), expected)

  def testFilterToNotInClan(self):
    input_predictions_csv = ("true_label,predicted_label\n"
                             "PF00619.21,SOMETHING_ELSE\n")
    input_predictions_df = pd.read_csv(io.StringIO(input_predictions_csv))
    clan_dict = {"PF00619": "CL00001"}
    expected = "true_label,predicted_label\n"
    actual = model_performance_analysis.filter_to_not_in_clan(
        input_predictions_df, clan_dict)
    self.assertEqual(actual.to_csv(index=False), expected)

  @parameterized.parameters(
      dict(
          predicted_labels1=["b", "b", "a", "b"],
          predicted_labels2=["a", "a", "a", "b"],
          expected_table=[[1, 1], [1, 1]]),
      dict(
          predicted_labels1=["a", "b", "a", "b"],
          predicted_labels2=["a", "a", "a", "b"],
          expected_table=[[1, 0], [1, 2]]),
      dict(
          predicted_labels1=["a", "b", "a", "b"],
          predicted_labels2=["b", "a", "a", "b"],
          expected_table=[[1, 0], [2, 1]]),
      dict(
          predicted_labels1=["a", "b", "a", "a"],
          predicted_labels2=["b", "a", "a", "b"],
          expected_table=[[0, 0], [3, 1]]),
  )
  def test_contingency_table(self, predicted_labels1, predicted_labels2,
                             expected_table):
    true_labels = ["a", "b", "a", "a"]
    uids = ["id0", "id1", "id2", "id3"]

    # Shuffle the order of the data for model 2, to ensure that the code
    # is properly joining on uid, instead of relying on the list order.
    index_reordering = [1, 0, 3, 2]
    uids2 = np.array(uids)[index_reordering]
    predicted_labels2 = np.array(predicted_labels2)[index_reordering]
    true_labels2 = np.array(true_labels)[index_reordering]

    df1 = pd.DataFrame(
        dict(pred=predicted_labels1, uid=uids, true_label=true_labels))
    df2 = pd.DataFrame(
        dict(pred=predicted_labels2, uid=uids2, true_label=true_labels2))

    actual_table = model_performance_analysis.joint_correctness_contingency_table(
        df1,
        df2,
        uid_key="uid",
        prediction_key="pred",
        true_label_key="true_label")
    np.testing.assert_allclose(expected_table, actual_table)

  @parameterized.named_parameters(
      dict(
          testcase_name="tied",
          contingency_table=[[0, 1], [1, 0]],
          expect_p_value_small=False),
      dict(
          testcase_name="counts_too_small",
          contingency_table=[[0, 3], [1, 0]],
          expect_p_value_small=False),
      dict(
          testcase_name="conclusive",
          contingency_table=[[0, 8], [1, 0]],
          expect_p_value_small=True),
      dict(
          testcase_name="conclusive_but_in_wrong_direction",
          contingency_table=[[0, 1], [8, 0]],
          expect_p_value_small=False),
      dict(
          testcase_name="diagonal_entries_no_effect",
          contingency_table=[[1000, 8], [1, 1000]],
          expect_p_value_small=True))
  def test_binomial_test(self,
                         contingency_table,
                         expect_p_value_small,
                         p_value_thresh=0.05):
    p_value = model_performance_analysis._run_binomial_test(
        contingency_table, verbose=False)
    if expect_p_value_small:
      self.assertLess(p_value, p_value_thresh)
    else:
      self.assertGreater(p_value, p_value_thresh)

  @parameterized.named_parameters(
      dict(
          testcase_name="tied",
          contingency_table=[[0, 1], [1, 0]],
          expect_p_value_small=False),
      dict(
          testcase_name="counts_too_small",
          contingency_table=[[0, 3], [1, 0]],
          expect_p_value_small=False),
      dict(
          testcase_name="conclusive",
          contingency_table=[[0, 8], [1, 0]],
          expect_p_value_small=True),
      dict(
          testcase_name="diagonal_entries_no_effect",
          contingency_table=[[1000, 8], [1, 1000]],
          expect_p_value_small=True))
  def test_mcnemar_test(self,
                        contingency_table,
                        expect_p_value_small,
                        p_value_thresh=0.05):
    p_value = model_performance_analysis._run_mcnemar_test(
        contingency_table, verbose=False)
    if expect_p_value_small:
      self.assertLess(p_value, p_value_thresh)
    else:
      self.assertGreater(p_value, p_value_thresh)

  def testLoadShardedCsvTest(self):
    input_csv_dir = tempfile.mkdtemp()
    _write_to_file("""A,B,C""", input_csv_dir)
    _write_to_file("""D,E,F""", input_csv_dir)
    input_columns = ["letter_1", "letter_2", "letter_3"]

    expected = pd.read_csv(
        io.StringIO("letter_1,letter_2,letter_3\n"
                    "A,B,C\n"
                    "D,E,F"))

    actual = model_performance_analysis.load_sharded_df_csvs(
        input_csv_dir, column_names=input_columns)

    test_util.assert_dataframes_equal(
        self, actual, expected, sort_by_column="letter_1")

  def testLoadShardedCsvPassColumns(self):
    input_csv_dir = tempfile.mkdtemp()
    _write_to_file("""A,B,C""", input_csv_dir)
    _write_to_file("""D,E,F""", input_csv_dir)
    input_columns = ["letter_1", "letter_2", "letter_3"]

    expected = pd.read_csv(
        io.StringIO("letter_1,letter_2,letter_3\n"
                    "A,B,C\n"
                    "D,E,F"))

    actual = model_performance_analysis.load_sharded_df_csvs(
        input_csv_dir, column_names=input_columns)

    test_util.assert_dataframes_equal(
        self, actual, expected, sort_by_column="letter_1")

  def testLoadShardedCsvIgnoreFirstLine(self):
    input_csv_dir = tempfile.mkdtemp()
    _write_to_file("""col1,col2,col3\nA,B,C""", input_csv_dir)
    _write_to_file("""col1,col2,col3\nD,E,F""", input_csv_dir)
    input_columns = ["letter_1", "letter_2", "letter_3"]

    expected = pd.read_csv(
        io.StringIO("letter_1,letter_2,letter_3\n"
                    "A,B,C\n"
                    "D,E,F"))

    actual = model_performance_analysis.load_sharded_df_csvs(
        input_csv_dir, column_names=input_columns, ignore_first_line=True)

    test_util.assert_dataframes_equal(
        self, actual, expected, sort_by_column="letter_1")

  def testLoadShardedCsvUseGivenHeader(self):
    input_csv_dir = tempfile.mkdtemp()
    _write_to_file("""col1,col2,col3\nA,B,C""", input_csv_dir)
    _write_to_file("""col1,col2,col3\nD,E,F""", input_csv_dir)

    expected = pd.read_csv(io.StringIO("col1,col2,col3\n" "A,B,C\n" "D,E,F"))

    actual = model_performance_analysis.load_sharded_df_csvs(
        input_csv_dir, use_given_header=True)

    test_util.assert_dataframes_equal(
        self, actual, expected, sort_by_column="col1")

  def testLoadShardedCsvFailsOnConflictingArgumentsGiven(self):
    input_columns = ["letter_1", "letter_2", "letter_3"]
    with self.assertRaisesRegex(ValueError, "Cannot pass both"):
      model_performance_analysis.load_sharded_df_csvs(
          "", column_names=input_columns, use_given_header=True)

    with self.assertRaisesRegex(ValueError, "Cannot pass both"):
      model_performance_analysis.load_sharded_df_csvs(
          "", ignore_first_line=True, use_given_header=True)

  def testLoadShardedCsvFailsOnNonUniformColumns(self):
    input_csv_dir = tempfile.mkdtemp()
    _write_to_file("""col1,col2,col3\nA,B,C""", input_csv_dir)
    _write_to_file("""XXX,YYY,ZZZ\nD,E,F""", input_csv_dir)

    with self.assertRaisesRegex(ValueError, "YYY"):
      model_performance_analysis.load_sharded_df_csvs(
          input_csv_dir, use_given_header=True)


if __name__ == "__main__":
  absltest.main()
