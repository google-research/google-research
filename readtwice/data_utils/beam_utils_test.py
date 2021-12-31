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

import json
from typing import Optional

from absl.testing import absltest
import apache_beam as beam
from apache_beam.options import pipeline_options
import apache_beam.testing.util as beam_testing
import attr
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from readtwice.data_utils import beam_utils
from readtwice.data_utils import data_utils


class BeamUtilsTest(absltest.TestCase):

  def _run_pipeline(self, pipeline):
    options = pipeline_options.PipelineOptions(
        runner='DirectRunner', direct_running_mode='in_memory')
    p = beam.Pipeline(options=options)
    pipeline(p)
    p.run().wait_until_finish()

  def test_read_files_to_tokenized_documents(self):
    input_text = u"""
      A small doc.
      Another sentence.

      Second doc is smaller.
    """
    input_file = self.create_tempfile(content=input_text, mode='wt')

    vocab = u"""[PAD]
                [UNK]
                [CLS]
                [SEP]
                [MASK]
                a
                small
                doc
                an
                ##other
                second
                is
                ##er
                .
    """
    vocab_file = self.create_tempfile(content=vocab, mode='wt')

    def pipeline(root):
      tokenized_documents = (
          root | beam.Create([input_file.full_path])
          | beam_utils.ReadFilesToTokenizedDocuments(
              vocab_path=vocab_file.full_path, do_lower_case=True))
      expected = [
          data_utils.TokenizedBertDocument([
              data_utils.TokenizedSentence(
                  token_ids=[5, 6, 7, 13],
                  is_continuation=[0, 0, 0, 0],
                  tokens=['a', 'small', 'doc', '.'],
                  raw_text='A small doc.',
                  annotations=[]),
              data_utils.TokenizedSentence(
                  token_ids=[8, 9, 1, 13],
                  is_continuation=[0, 1, 0, 0],
                  tokens=['an', '##other', '[UNK]', '.'],
                  raw_text='Another sentence.',
                  annotations=[]),
          ]),
          data_utils.TokenizedBertDocument([
              data_utils.TokenizedSentence(
                  token_ids=[10, 7, 11, 6, 12, 13],
                  is_continuation=[0, 0, 0, 0, 1, 0],
                  tokens=['second', 'doc', 'is', 'small', '##er', '.'],
                  raw_text='Second doc is smaller.',
                  annotations=[]),
          ])
      ]
      beam_testing.assert_that(tokenized_documents,
                               beam_testing.equal_to(expected))

    self._run_pipeline(pipeline)

  def test_calculate_statistics(self):
    num_observations = 2001

    # `num_observations` evenly spaced numbers from 0 through 1000, inclusive.
    numbers = [
        x / (num_observations - 1) * 1000 for x in range(num_observations)
    ]

    def pipeline(root):
      stats = (
          root | beam.Create(numbers)
          | beam_utils.CalculateStatistics.Globally())
      expected = [
          beam_utils.SummaryStatistics(
              count=num_observations,
              min=0.,
              mean=500,
              max=1000,
              quantiles=[
                  (.001, 1),
                  (.002, 2),
                  (.003, 3),
                  (.004, 4),
                  (.005, 5),
                  (.01, 10),
                  (.02, 20),
                  (.03, 30),
                  (.04, 40),
                  (.05, 50),
                  (.1, 100),
                  (.15, 150),
                  (.2, 200),
                  (.25, 250),
                  (.3, 300),
                  (.35, 350),
                  (.4, 400),
                  (.45, 450),
                  (.5, 500),
                  (.55, 550),
                  (.6, 600),
                  (.65, 650),
                  (.7, 700),
                  (.75, 750),
                  (.8, 800),
                  (.85, 850),
                  (.9, 900),
                  (.95, 950),
                  (.96, 960),
                  (.97, 970),
                  (.98, 980),
                  (.99, 990),
                  (.995, 995),
                  (.996, 996),
                  (.997, 997),
                  (.998, 998),
                  (.999, 999),
              ])
      ]
      beam_testing.assert_that(stats, beam_testing.equal_to(expected))

    self._run_pipeline(pipeline)

  def test_calculate_statistics_empty_input(self):
    numbers = []

    def pipeline(root):
      stats = (
          root | beam.Create(numbers)
          | beam_utils.CalculateStatistics.Globally()
          | beam.Map(str))

      # We compare the `str` output since object equality testing will fail
      # because `float('nan') != float('nan')`.
      expected = [
          str(
              beam_utils.SummaryStatistics(
                  count=0,
                  min=float('inf'),
                  mean=float('nan'),
                  max=float('-inf'),
                  quantiles=[]))
      ]
      beam_testing.assert_that(stats, beam_testing.equal_to(expected))

    self._run_pipeline(pipeline)

  def test_summary_statistics_json_serialization(self):
    stats = beam_utils.SummaryStatistics(
        count=10,
        min=-1,
        mean=0.1,
        max=1,
        quantiles=[(0, -1), (0.25, -0.5), (0.5, 0), (0.75, 0.5), (1, 1)])

    json_str = json.dumps(attr.asdict(stats))
    reloaded_stats = beam_utils.SummaryStatistics.from_dict(
        json.loads(json_str))
    self.assertEqual(stats, reloaded_stats)

    json_str2 = json.dumps(stats.to_dict_for_json())
    reloaded_stats2 = beam_utils.SummaryStatistics.from_dict(
        json.loads(json_str2))
    self.assertEqual(stats, reloaded_stats2)

  def test_summary_statistics_json_serialization_empty_stats(self):
    stats = beam_utils.SummaryStatistics(
        count=0,
        min=float('inf'),
        mean=float('nan'),
        max=float('-inf'),
        quantiles=[])

    json_str = json.dumps(attr.asdict(stats))
    reloaded_stats = beam_utils.SummaryStatistics.from_dict(
        json.loads(json_str))
    # We compare the `str` output since object equality testing will fail
    # because `float('nan') != float('nan')`.
    self.assertEqual(str(stats), str(reloaded_stats))

    json_str2 = json.dumps(stats.to_dict_for_json())
    reloaded_stats2 = beam_utils.SummaryStatistics.from_dict(
        json.loads(json_str2))
    # We compare the `str` output since object equality testing will fail
    # because `float('nan') != float('nan')`.
    self.assertEqual(str(stats), str(reloaded_stats2))

  def test_pack_examples(self):
    small_example = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [1, 2, 3, 4]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [0, 1, 1, 1]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    large_example = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [1, 2, 3, 4, 5, 6, 7]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [0, 0, 1, 1, 1, 2, 2]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    input_examples = [
        large_example, small_example, small_example, small_example
    ]

    expected_packed_examples = [
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 1, 2, 3, 4, 0, 0]
                  }
                }
              }
              feature {
                key: "sentence_ids"
                value {
                  int64_list {
                    value: [0, 1, 1, 1, 2, 3, 3, 3, 0, 0]
                  }
                }
              }
              feature {
                key: "global_token_ids"
                value {
                  int64_list {
                    value: [1, 1, 1, 1, 0]
                  }
                }
              }
              feature {
                key: "long_breakpoints"
                value {
                  int64_list {
                    value: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
                  }
                }
              }
              feature {
                key: "global_breakpoints"
                value {
                  int64_list {
                    value: [0, 1, 0, 1, 0]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ["packed"]
                  }
                }
              }
            }
            """, tf.train.Example()),
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "sentence_ids"
                value {
                  int64_list {
                    value: [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "global_token_ids"
                value {
                  int64_list {
                    value: [1, 1, 0, 0 ,0]
                  }
                }
              }
              feature {
                key: "long_breakpoints"
                value {
                  int64_list {
                    value: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "global_breakpoints"
                value {
                  int64_list {
                    value: [0, 1, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ["flushed"]
                  }
                }
              }
            }
            """, tf.train.Example()),
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 5, 6, 7, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "sentence_ids"
                value {
                  int64_list {
                    value: [0, 0, 1, 1, 1, 2, 2, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "global_token_ids"
                value {
                  int64_list {
                    value: [1, 1, 1, 0, 0]
                  }
                }
              }
              feature {
                key: "long_breakpoints"
                value {
                  int64_list {
                    value: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "global_breakpoints"
                value {
                  int64_list {
                    value: [0, 0, 1, 0, 0]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ["flushed"]
                  }
                }
              }
            }
            """, tf.train.Example())
    ]

    def pipeline(root):
      example_packer = beam_utils.PriorityExamplePacker(
          priority_feature='token_ids',
          max_lengths=dict(token_ids=10, sentence_ids=10, global_token_ids=5),
          breakpoint_features=dict(
              token_ids='long_breakpoints',
              global_token_ids='global_breakpoints'),
          cumulative_features=['sentence_ids'],
          min_packing_fraction=0.75,
          max_cache_len=5)

      result = (
          root | beam.Create(input_examples)
          | beam_utils.PackExamples(example_packer)
          | beam.Map(str))

      beam_testing.assert_that(
          result,
          beam_testing.equal_to([str(x) for x in expected_packed_examples]))

    self._run_pipeline(pipeline)

  def test_priority_example_packer(self):

    def make_test_example(
        num_values,
        token_pad_num = None,
        global_token_pad_num = None,
        packing_status = None):
      """Makes a tf.Example for testing packing logic.

      The result will have the following features:
      1. An int64 `token_ids` feature containing `num_values` values counting
        from 1 upwards. If `token_pad_num` is supplied, additional `0` values
        will be appended to the right until the number of values reaches
        `token_pad_num`.
      2. An int64 `global_token_ids` feature containing a single `1` value.
        If `global_token_pad_num` is supplied, additional `0` values
        will be appended to the right until the number of values reaches
        `global_token_pad_num`.
      3. If `packing_status` is supplied, a feature by that name will be added
        with the corresponding bytes value.

      Args:
        num_values: Positive integer number of `token_ids` in the example.
        token_pad_num: Optional length to pad `token_ids` to. Must not be less
          than `num_values`.
        global_token_pad_num: Optional length to pad `global_token_ids` to. Must
          be positive.
        packing_status: Optional bytes value to record in a `packing_status`
          feature.

      Returns:
        The example.
      """
      values = [x + 1 for x in range(num_values)]
      if token_pad_num is not None:
        while len(values) < token_pad_num:
          values.append(0)
      result = tf.train.Example()
      result.features.feature['token_ids'].int64_list.value.extend(values)

      global_values = [1]
      if global_token_pad_num is not None:
        remaining_len = global_token_pad_num - len(global_values)
        global_values.extend([0] * remaining_len)
      result.features.feature['global_token_ids'].int64_list.value.extend(
          global_values)

      if packing_status is not None:
        result.features.feature['packing_status'].bytes_list.value.append(
            packing_status)

      return result

    # Some example calls to `make_test_example()` for illustration:
    self.assertEqual(
        text_format.Parse(
            """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [1, 2, 3, 4, 5]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [1]
              }
            }
          }
        }
        """, tf.train.Example()), make_test_example(5))

    self.assertEqual(
        text_format.Parse(
            """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [1, 0, 0, 0]
              }
            }
          }
          feature {
            key: "packing_status"
            value {
              bytes_list {
                value: ['untouched']
              }
            }
          }
        }
        """, tf.train.Example()),
        make_test_example(
            6,
            token_pad_num=10,
            global_token_pad_num=4,
            packing_status=b'untouched'))

    # Proceed to test `PriorityExamplePacker`.

    # `breakpoint_features` and `cumulative_features` are tested in
    # `test_pack_examples()` above, so we omit them here for brevity.
    packer = beam_utils.PriorityExamplePacker(
        priority_feature='token_ids',
        max_lengths=dict(token_ids=10, global_token_ids=4),
        breakpoint_features={},
        cumulative_features=[],
        min_packing_fraction=0.85,
        max_cache_len=2)

    self.assertEqual([], packer.add_example(make_test_example(8)))
    self.assertEqual([], packer.add_example(make_test_example(4)))
    self.assertEqual([
        make_test_example(
            8,
            token_pad_num=10,
            global_token_pad_num=4,
            packing_status=b'evicted')
    ], packer.add_example(make_test_example(7)))
    self.assertEqual([
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 5, 6, 7, 1, 2, 0]
                  }
                }
              }
              feature {
                key: "global_token_ids"
                value {
                  int64_list {
                    value: [1, 1, 0, 0]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ['packed']
                  }
                }
              }
            }
            """, tf.train.Example())
    ], packer.add_example(make_test_example(2)))
    self.assertEqual([], packer.add_example(make_test_example(1)))
    self.assertEqual([], packer.add_example(make_test_example(1)))
    self.assertEqual([], packer.add_example(make_test_example(5)))

    # We satisfy minimum `global_token_ids` instead of `token_ids` here.
    self.assertEqual([
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 1, 1, 1, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "global_token_ids"
                value {
                  int64_list {
                    value: [1, 1, 1, 1]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ['packed']
                  }
                }
              }
            }
            """, tf.train.Example())
    ], packer.add_example(make_test_example(1)))

    self.assertEqual([
        make_test_example(
            9,
            token_pad_num=10,
            global_token_pad_num=4,
            packing_status=b'untouched')
    ], packer.add_example(make_test_example(9)))

    with self.assertRaises(ValueError):
      packer.add_example(make_test_example(11))

    with self.assertRaises(ValueError):
      packer.add_example(make_test_example(3, global_token_pad_num=5))

    self.assertEqual([], packer.add_example(make_test_example(8)))
    self.assertEqual([
        make_test_example(
            8,
            token_pad_num=10,
            global_token_pad_num=4,
            packing_status=b'flushed'),
        make_test_example(
            5,
            token_pad_num=10,
            global_token_pad_num=4,
            packing_status=b'flushed')
    ], packer.flush_examples())
    self.assertEqual([], packer.flush_examples())

  def test_priority_example_packer_for_read_it_twice_model(self):
    """Test for PriorityExamplePacker with read-it-twice model's data format.

    For brevity reasons, we omit `is_continuation` feature.
    Note that in all examples below the `block_length` is 5
    """

    # Test case 1: Simple combination of blocks
    packer = beam_utils.PriorityExamplePacker(
        priority_feature='token_ids',
        max_lengths=dict(token_ids=20, block_ids=4),
        breakpoint_features={},
        cumulative_features=[],
        min_packing_fraction=1.0,
        max_cache_len=2)

    self.assertEqual([],
                     packer.add_example(
                         text_format.Parse(
                             """
      features {
        feature {
          key: "token_ids"
          value {
            int64_list {
              value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
          }
        }
        feature {
          key: "block_ids"
          value {
            int64_list {
              value: [1, 1]
            }
          }
        }
      }
      """, tf.train.Example())))

    self.assertEqual([],
                     packer.add_example(
                         text_format.Parse(
                             """
      features {
        feature {
          key: "token_ids"
          value {
            int64_list {
              value: [1, 0, 0, 0, 0]
            }
          }
        }
        feature {
          key: "block_ids"
          value {
            int64_list {
              value: [2]
            }
          }
        }
      }
      """, tf.train.Example())))

    self.assertEqual([
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                  }
                }
              }
              feature {
                key: "block_ids"
                value {
                  int64_list {
                    value: [1, 1, 2, 5]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ['packed']
                  }
                }
              }
            }
            """, tf.train.Example())
    ],
                     packer.add_example(
                         text_format.Parse(
                             """
      features {
        feature {
          key: "token_ids"
          value {
            int64_list {
              value: [1, 0, 0, 0, 0]
            }
          }
        }
        feature {
          key: "block_ids"
          value {
            int64_list {
              value: [5]
            }
          }
        }
      }
      """, tf.train.Example())))

    # Test case 2: The block is expected to be filled in `flush_examples`
    packer = beam_utils.PriorityExamplePacker(
        priority_feature='token_ids',
        max_lengths=dict(token_ids=20, block_ids=4),
        breakpoint_features={},
        cumulative_features=[],
        min_packing_fraction=1.0,
        max_cache_len=2,
        padding_token_ids=dict(token_ids=-1, block_ids=0))

    self.assertEqual([],
                     packer.add_example(
                         text_format.Parse(
                             """
      features {
        feature {
          key: "token_ids"
          value {
            int64_list {
              value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
          }
        }
        feature {
          key: "block_ids"
          value {
            int64_list {
              value: [1, 1]
            }
          }
        }
      }
      """, tf.train.Example())))

    self.assertEqual([],
                     packer.add_example(
                         text_format.Parse(
                             """
      features {
        feature {
          key: "token_ids"
          value {
            int64_list {
              value: [1, 0, 0, 0, 0]
            }
          }
        }
        feature {
          key: "block_ids"
          value {
            int64_list {
              value: [2]
            }
          }
        }
      }
      """, tf.train.Example())))

    self.assertEqual([
        text_format.Parse(
            """
            features {
              feature {
                key: "token_ids"
                value {
                  int64_list {
                    value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                            1, 0, 0, 0, 0, -1, -1, -1, -1, -1]
                  }
                }
              }
              feature {
                key: "block_ids"
                value {
                  int64_list {
                    value: [1, 1, 2, 0]
                  }
                }
              }
              feature {
                key: "packing_status"
                value {
                  bytes_list {
                    value: ['flushed']
                  }
                }
              }
            }
            """, tf.train.Example())
    ], packer.flush_examples())

  def test_singletons_to_list(self):

    def pipeline(root):
      number_singleton = root | 'CreateNumber' >> beam.Create([3.14])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      tuple_singleton = root | 'CreateList' >> beam.Create([(1, 2, 3)])

      expected = [[3.14, 'test', (1, 2, 3)]]
      result = beam_utils.singletons_to_list(
          [number_singleton, string_singleton, tuple_singleton])

      beam_testing.assert_that(result, beam_testing.equal_to(expected))

    self._run_pipeline(pipeline)

  def test_singletons_to_list_raises_if_not_singleton(self):

    def pipeline(root):
      numbers_not_singleton = root | 'CreateNumber' >> beam.Create([3.14, -1])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      tuple_singleton = root | 'CreateList' >> beam.Create([(1, 2, 3)])

      return beam_utils.singletons_to_list(
          [numbers_not_singleton, string_singleton, tuple_singleton])

    with self.assertRaises((ValueError, RuntimeError)):
      self._run_pipeline(pipeline)

  def test_singletons_to_list_raises_for_empty_list(self):

    def pipeline(unused_root):
      return beam_utils.singletons_to_list([])

    with self.assertRaises(ValueError):
      self._run_pipeline(pipeline)

  def test_singletons_to_dict(self):

    def pipeline(root):
      number_singleton = root | 'CreateNumber' >> beam.Create([3.14])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      list_singleton = root | 'CreateList' >> beam.Create([[1, 2, 3]])

      expected = [dict(number=3.14, string='test', list=[1, 2, 3])]
      result = beam_utils.singletons_to_dict(
          number=number_singleton, string=string_singleton, list=list_singleton)

      beam_testing.assert_that(result, beam_testing.equal_to(expected))

    self._run_pipeline(pipeline)

  def test_singletons_to_dict_raises_if_not_singleton(self):

    def pipeline(root):
      numbers_not_singleton = root | 'CreateNumber' >> beam.Create([3.14, -1])
      string_singleton = root | 'CreateString' >> beam.Create(['test'])
      list_singleton = root | 'CreateList' >> beam.Create([[1, 2, 3]])

      return beam_utils.singletons_to_dict(
          number=numbers_not_singleton,
          string=string_singleton,
          list=list_singleton)

    with self.assertRaises((ValueError, RuntimeError)):
      self._run_pipeline(pipeline)

  def test_singletons_to_dict_raises_for_empty_kwargs(self):

    def pipeline(unused_root):
      return beam_utils.singletons_to_dict()

    with self.assertRaises(ValueError):
      self._run_pipeline(pipeline)


if __name__ == '__main__':
  absltest.main()
