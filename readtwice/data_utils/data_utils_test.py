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

import copy
import os

from absl.testing import absltest
import mock
import tensorflow.compat.v1 as tf

from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
from readtwice.data_utils import data_utils
from readtwice.data_utils import tokenization

_EXAMPLE_BERT_PRETRAINING_TEXT = u"""
Some text to test Unicode handling: 力加勝北区ᴵᴺᵀᵃছজটডণত
Text should be one-sentence-per-line.
Empty lines separate documents.

Here is the start of a new document.


Yet another document.
With a second sentence.

"""


class DataUtilsTest(compare.ProtoAssertions, absltest.TestCase):

  def test_tokenized_bert_document_num_tokens(self):
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(token_ids=[0, 1, 2]),
        data_utils.TokenizedSentence(token_ids=[10, 11]),
        data_utils.TokenizedSentence(token_ids=[20, 21, 22, 23])
    ])

    self.assertEqual(9, document.num_tokens())

  def test_tokenized_bert_document_to_tf_example(self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[0, 1, 2], is_continuation=[0, 1, 1]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23], is_continuation=[0, 0, 1, 0])
    ])

    expected_without_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [0, 0, 0, 1, 1, 2, 2, 2, 2]
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
    self.assertProtoEqual(expected_without_global_cls,
                           document.to_tf_example(global_sentence_token_id=1))

    expected_with_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [1, 1, 1, 2, 2, 3, 3, 3, 3]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [101, 1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())
    self.assertProtoEqual(
        expected_with_global_cls,
        document.to_tf_example(
            global_sentence_token_id=1,
            include_global_cls_token=True,
            global_cls_token_id=101))

  def test_tokenized_bert_document_to_tf_example_fixed_blocks_exact(self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[0, 1, 2], is_continuation=[0, 1, 1]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23], is_continuation=[0, 0, 1, 0])
    ])

    expected_without_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [0, 0, 0, 1, 1, 1, 2, 2, 2]
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
    self.assertProtoEqual(
        expected_without_global_cls,
        document.to_tf_example(
            global_sentence_token_id=1,
            global_mode=data_utils.BertDocumentGlobalMode.FIXED_BLOCKS,
            fixed_blocks_num_tokens_per_block=3,
        ))

    expected_with_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [1, 1, 1, 2, 2, 2, 3, 3, 3]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [101, 1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())
    self.assertProtoEqual(
        expected_with_global_cls,
        document.to_tf_example(
            global_sentence_token_id=1,
            global_mode=data_utils.BertDocumentGlobalMode.FIXED_BLOCKS,
            fixed_blocks_num_tokens_per_block=3,
            include_global_cls_token=True,
            global_cls_token_id=101))

  def test_tokenized_bert_document_to_tf_example_fixed_blocks_with_remainder(
      self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[0, 1, 2], is_continuation=[0, 1, 1]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23], is_continuation=[0, 0, 1, 0])
    ])

    expected_without_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [0, 0, 0, 0, 1, 1, 1, 1, 2]
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
    self.assertProtoEqual(
        expected_without_global_cls,
        document.to_tf_example(
            global_sentence_token_id=1,
            global_mode=data_utils.BertDocumentGlobalMode.FIXED_BLOCKS,
            fixed_blocks_num_tokens_per_block=4,
        ))

    expected_with_global_cls = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [0, 1, 2, 10, 11, 20, 21, 22, 23]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 1, 1, 0, 0, 0, 0, 1, 0]
              }
            }
          }
          feature {
            key: "sentence_ids"
            value {
              int64_list {
                value: [1, 1, 1, 1, 2, 2, 2, 2, 3]
              }
            }
          }
          feature {
            key: "global_token_ids"
            value {
              int64_list {
                value: [101, 1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())
    self.assertProtoEqual(
        expected_with_global_cls,
        document.to_tf_example(
            global_sentence_token_id=1,
            global_mode=data_utils.BertDocumentGlobalMode.FIXED_BLOCKS,
            fixed_blocks_num_tokens_per_block=4,
            include_global_cls_token=True,
            global_cls_token_id=101))

  def test_tokenized_bert_document_to_tf_strided_large_example(self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[1, 2, 3], is_continuation=[0, 1, 1]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23], is_continuation=[0, 0, 1, 0]),
        data_utils.TokenizedSentence(
            token_ids=[31, 32, 33, 34], is_continuation=[0, 0, 0, 0])
    ])

    expected_without_overlap = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 20, 21, 22, 23, 31,
                        101, 32, 33, 34, 0, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProtoEqual(
        expected_without_overlap,
        document.to_tf_strided_large_example(
            overlap_length=0,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101))

    expected_with_overlap_1 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 11, 20, 21, 22, 23,
                        101, 23, 31, 32, 33, 34]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProtoEqual(
        expected_with_overlap_1,
        document.to_tf_strided_large_example(
            overlap_length=1,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101))

    expected_with_overlap_3 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 10, 11, 20, 21, 22,
                        101, 20, 21, 22, 23, 31,
                        101, 23, 31, 32, 33, 34]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 0, 0, 1,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1, 1]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProtoEqual(
        expected_with_overlap_3,
        document.to_tf_strided_large_example(
            overlap_length=3,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101))

  def test_tokenized_bert_document_to_tf_strided_large_example_with_annotations_2(
      self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[1, 2, 3, 4, 5, 6],
            is_continuation=[0, 0, 0, 0, 0, 0],
            annotations=[
                data_utils.Annotation(0, 5, label=1),
                data_utils.Annotation(2, 4, label=2),
            ]),
        data_utils.TokenizedSentence(
            token_ids=[7, 8, 9],
            is_continuation=[0, 0, 0],
            annotations=[
                data_utils.Annotation(0, 1, label=3),
                data_utils.Annotation(2, 2, label=4),
            ]),
    ])

    expected_without_overlap = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 4,
                        101, 5, 6, 7, 8,
                        101, 9, 0, 0, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [1, 3, 0,
                        1, 1, 3,
                        1, 0, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [4, 4, 0,
                        2, 1, 4,
                        1, 0, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 2, 0,
                        1, 2, 3,
                        4, 0, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected_without_overlap,
        document.to_tf_strided_large_example(
            overlap_length=0,
            block_length=5,
            padding_token_id=0,
            prefix_token_ids=101,
            answer_only_strictly_inside_annotations=False,
            max_num_annotations=3))

    expected_with_overlap_1 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 4,
                        101, 4, 5, 6, 7,
                        101, 7, 8, 9, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [1, 3, 0,
                        1, 1, 4,
                        1, 3, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [4, 4, 0,
                        3, 2, 4,
                        2, 3, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 2, 0,
                        1, 2, 3,
                        3, 4, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected_with_overlap_1,
        document.to_tf_strided_large_example(
            overlap_length=1,
            block_length=5,
            padding_token_id=0,
            prefix_token_ids=101,
            answer_only_strictly_inside_annotations=False,
            max_num_annotations=3))


#     # TODO(urikz): Uncomment when the function supports adding annotations
#     # from previous sentences in the overlapping region.
#     expected_with_overlap_2 = text_format.Parse(
#         """
#         features {
#           feature {
#             key: "token_ids"
#             value {
#               int64_list {
#                 value: [101, 1, 2, 3, 4,
#                         101, 3, 4, 5, 6,
#                         101, 5, 6, 7, 8,
#                         101, 7, 8, 9, 0]
#               }
#             }
#           }
#           feature {
#             key: "is_continuation"
#             value {
#               int64_list {
#                 value: [0, 0, 0, 0, 0,
#                         0, 0, 0, 0, 0,
#                         0, 0, 0, 0, 0,
#                         0, 0, 0, 0, 0]
#               }
#             }
#           }
#           feature {
#             key: "block_ids"
#             value {
#               int64_list {
#                 value: [1, 1, 1, 1]
#               }
#             }
#           }
#           feature {
#             key: "annotation_begins"
#             value {
#               int64_list {
#                 value: [1, 3, 0,
#                         1, 1, 0,
#                         1, 1, 3,
#                         1, 3, 0]
#               }
#             }
#           }
#           feature {
#             key: "annotation_ends"
#             value {
#               int64_list {
#                 value: [4, 4, 0,
#                         4, 4, 0,
#                         2, 1, 4,
#                         2, 3, 0]
#               }
#             }
#           }
#           feature {
#             key: "annotation_labels"
#             value {
#               int64_list {
#                 value: [1, 2, 0,
#                         1, 2, 0,
#                         1, 2, 3,
#                         3, 4, 0]
#               }
#             }
#           }
#           feature {
#             key: "prefix_length"
#             value {
#               int64_list {
#                 value: [1, 1, 1, 1]
#               }
#             }
#           }
#         }
#         """, tf.train.Example())

#     self.assertProtoEqual(
#         expected_with_overlap_2,
#         document.to_tf_strided_large_example(
#             overlap_length=2,
#             block_length=5,
#             padding_token_id=0,
#             prefix_token_ids=101,
#             max_num_annotations=3))

  def test_tokenized_bert_document_to_tf_strided_large_example_multi_overlap(
      self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(token_ids=[1, 2], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[30, 31], is_continuation=[0, 0]),
    ])

    expected_with_overlap_4 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 10, 11, 20, 21,
                        101, 10, 11, 20, 21, 30, 31]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0]
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
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProtoEqual(
        expected_with_overlap_4,
        document.to_tf_strided_large_example(
            overlap_length=4,
            block_length=7,
            padding_token_id=0,
            prefix_token_ids=101))

    expected_with_overlap_3 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 10, 11, 20, 21,
                        101, 11, 20, 21, 30, 31, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0]
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
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProtoEqual(
        expected_with_overlap_3,
        document.to_tf_strided_large_example(
            overlap_length=3,
            block_length=7,
            padding_token_id=0,
            prefix_token_ids=101))

  def test_tokenized_bert_document_to_tf_strided_large_example_with_annotations(
      self):
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[1, 2, 3],
            is_continuation=[0, 1, 1],
            annotations=[
                data_utils.Annotation(0, 1),
                data_utils.Annotation(2, 2),
            ]),
        data_utils.TokenizedSentence(
            token_ids=[10, 11], is_continuation=[0, 0]),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23],
            is_continuation=[0, 0, 1, 0],
            annotations=[data_utils.Annotation(2, 3)]),
        data_utils.TokenizedSentence(
            token_ids=[31, 32, 33, 34],
            is_continuation=[0, 0, 0, 0],
            annotations=[data_utils.Annotation(3, 3)])
    ])

    expected_without_overlap = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 20, 21, 22, 23, 31,
                        101, 32, 33, 34, 0, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [1, 3, 3, 0, 3, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [2, 3, 4, 0, 3, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 1, 1, 0, 1, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected_without_overlap,
        document.to_tf_strided_large_example(
            overlap_length=0,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101,
            max_num_annotations=2))

    expected_with_overlap_1 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 11, 20, 21, 22, 23,
                        101, 23, 31, 32, 33, 34]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [1, 3, 4, 0, 5, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [2, 3, 5, 0, 5, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 1, 1, 0, 1, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected_with_overlap_1,
        document.to_tf_strided_large_example(
            overlap_length=1,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101,
            max_num_annotations=2))

    expected_with_overlap_3 = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [101, 1, 2, 3, 10, 11,
                        101, 10, 11, 20, 21, 22,
                        101, 20, 21, 22, 23, 31,
                        101, 23, 31, 32, 33, 34]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 0, 0,
                        0, 0, 0, 0, 0, 1,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0]
              }
            }
          }
          feature {
            key: "block_ids"
            value {
              int64_list {
                value: [1, 1, 1, 1]
              }
            }
          }
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [1, 3, 0, 0, 3, 0, 5, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [2, 3, 0, 0, 4, 0, 5, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 1, 0, 0, 1, 0, 1, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1, 1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected_with_overlap_3,
        document.to_tf_strided_large_example(
            overlap_length=3,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=101,
            max_num_annotations=2))

    # Additional test to verify tricky "overlap" case, when annotation
    # starts with is_continuation=1 token.
    # We leave `tokens` empty since they're not needed for example generation.
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            is_continuation=[0, 1, 1, 1, 0, 0, 0, 0],
            annotations=[data_utils.Annotation(2, 4)]),
    ])

    expected = text_format.Parse(
        """
        features {
          feature {
            key: "token_ids"
            value {
              int64_list {
                value: [111, 1, 2, 3, 4, 5,
                        111, 5, 6, 7, 8, 0]
              }
            }
          }
          feature {
            key: "is_continuation"
            value {
              int64_list {
                value: [0, 0, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0]
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
          feature {
            key: "answer_annotation_begins"
            value {
              int64_list {
                value: [3, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_ends"
            value {
              int64_list {
                value: [5, 0]
              }
            }
          }
          feature {
            key: "answer_annotation_labels"
            value {
              int64_list {
                value: [1, 0]
              }
            }
          }
          feature {
            key: "prefix_length"
            value {
              int64_list {
                value: [1, 1]
              }
            }
          }
        }
        """, tf.train.Example())

    self.assertProto2Contains(
        expected,
        document.to_tf_strided_large_example(
            overlap_length=3,
            block_length=6,
            padding_token_id=0,
            prefix_token_ids=111,
            max_num_annotations=1))

  def test_tokenized_bert_document_to_document_text(self):
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[0, 1, 2], raw_text='Beginning sentence.'),
        data_utils.TokenizedSentence(token_ids=[10, 11], raw_text='Next one.'),
        data_utils.TokenizedSentence(
            token_ids=[20, 21, 22, 23], raw_text='The final sentence!')
    ])

    self.assertEqual(
        '\n'.join(['Beginning sentence.', 'Next one.', 'The final sentence!']),
        document.to_document_text())

  def test_tokenized_bert_document_to_document_text_unavailable(self):
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(token_ids=[0, 1, 2]),
        data_utils.TokenizedSentence(token_ids=[10, 11]),
        data_utils.TokenizedSentence(token_ids=[20, 21, 22, 23])
    ])

    with self.assertRaises(ValueError):
      document.to_document_text()

  def test_expand_file_patterns(self):
    temp_dir = self.create_tempdir()
    dir_path = temp_dir.full_path
    temp_file1 = temp_dir.create_file(file_path='file1.txt')
    temp_file2 = temp_dir.create_file(
        file_path=os.path.join('subdir', 'file2.txt'))
    temp_file3 = temp_dir.create_file(
        file_path=os.path.join('subdir', 'file3.txt'))

    file_patterns = (
        os.path.join(dir_path, 'file1.txt') + ',' +
        os.path.join(dir_path, 'subdir', '*'))
    expected = [
        temp_file1.full_path, temp_file2.full_path, temp_file3.full_path
    ]
    self.assertEqual(expected, data_utils.expand_file_patterns(file_patterns))

    file_patterns_with_unmatched = (
        os.path.join(dir_path, 'file1.txt') + ',' +
        os.path.join(dir_path, 'unmatched.txt'))
    with self.assertRaises(ValueError):
      data_utils.expand_file_patterns(file_patterns_with_unmatched)

    self.assertEqual([temp_file1.full_path],
                     data_utils.expand_file_patterns(
                         file_patterns_with_unmatched,
                         ignore_unmatched_patterns=True))

  def test_read_text_lines(self):
    filepath = self.create_tempfile(
        content=_EXAMPLE_BERT_PRETRAINING_TEXT, mode='wt').full_path
    expected = _EXAMPLE_BERT_PRETRAINING_TEXT.splitlines()
    self.assertEqual(expected, list(data_utils.read_text_file_lines(filepath)))

  def test_parse_bert_pretraining_text(self):
    lines = _EXAMPLE_BERT_PRETRAINING_TEXT.splitlines()
    expected = [
        data_utils.BertDocument([
            u'Some text to test Unicode handling: 力加勝北区ᴵᴺᵀᵃছজটডণত',
            'Text should be one-sentence-per-line.',
            'Empty lines separate documents.',
        ]),  #
        data_utils.BertDocument(['Here is the start of a new document.']),  #
        data_utils.BertDocument(
            ['Yet another document.', 'With a second sentence.'])
    ]
    self.assertEqual(expected,
                     list(data_utils.parse_bert_pretraining_text(lines)))

  def test_tokenize_document_for_bert_with_annotations(self):
    tokenizer = mock.create_autospec(tokenization.FullTokenizer, instance=True)
    tokenizer.tokenize_full_output.side_effect = [
        tokenization.TokenizationResult(
            ['▁A', '▁small', '▁sentence', '.'], [5, 6, 7, 10],
            [0, 1, 7, 16], [0, 1, 7, 16], ['A', ' small', ' sentence', '.'],
            [0, 0, 0, 1]),
        tokenization.TokenizationResult([], [], [], [], [], []),
        tokenization.TokenizationResult(['▁An', 'other', '▁sentence', '.'],
                                        [8, 9, 7, 10], [0, 2, 7, 16],
                                        [0, 2, 7, 16],
                                        ['An', 'other', ' sentence', '.'],
                                        [0, 1, 0, 1]),
    ]
    tokenizer.convert_tokens_to_ids.side_effect = [
        [5, 6, 7, 10],  #
        [],  #
        [8, 9, 7, 10],
    ]
    # Set `sp_model` to signal we're using a SentencePiece model.
    tokenizer.sp_model = True

    document = data_utils.BertDocument([
        data_utils.Sentence('A small sentence.',
                            [data_utils.Annotation(2, 6, 'small')]),
        data_utils.Sentence(''),  # Empty sentences are dropped.
        data_utils.Sentence('Another sentence.',
                            [data_utils.Annotation(0, 15, 'Another sentence')])
    ])

    expected = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[5, 6, 7, 10],
            is_continuation=[0, 0, 0, 1],
            tokens=['▁A', '▁small', '▁sentence', '.'],
            raw_text='A small sentence.',
            annotations=[data_utils.Annotation(1, 1, ' small')]),
        data_utils.TokenizedSentence(
            token_ids=[8, 9, 7, 10],
            is_continuation=[0, 1, 0, 1],
            tokens=['▁An', 'other', '▁sentence', '.'],
            raw_text='Another sentence.',
            annotations=[data_utils.Annotation(0, 2, 'Another sentence')])
    ])
    self.assertEqual(expected,
                     data_utils.tokenize_document_for_bert(document, tokenizer))

  def test_realign_annotations(self):
    # invalid input interval
    with self.assertRaises(ValueError):
      data_utils.realign_annotations(0, -1, [0])
    with self.assertRaises(ValueError):
      data_utils.realign_annotations(1, 0, [0])

    # interval is not in the original sequence
    with self.assertRaises(ValueError):
      data_utils.realign_annotations(-10, -1, [0, 2, 4])
    with self.assertRaises(ValueError):
      data_utils.realign_annotations(0, 0, [1, 2, 99])

    self.assertEqual(data_utils.realign_annotations(0, 0, [0]), (0, 0))
    self.assertEqual(data_utils.realign_annotations(0, 100, [0]), (0, 0))

    self.assertEqual(data_utils.realign_annotations(2, 100, [0, 2]), (1, 1))
    self.assertEqual(data_utils.realign_annotations(100, 100, [0, 2]), (1, 1))

    offsets = [0, 2, 5, 10]
    self.assertEqual(data_utils.realign_annotations(0, 1, offsets), (0, 0))
    self.assertEqual(data_utils.realign_annotations(0, 2, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(0, 3, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(0, 4, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(0, 5, offsets), (0, 2))
    self.assertEqual(data_utils.realign_annotations(0, 9, offsets), (0, 2))
    self.assertEqual(data_utils.realign_annotations(0, 10, offsets), (0, 3))
    self.assertEqual(data_utils.realign_annotations(0, 11, offsets), (0, 3))

    self.assertEqual(data_utils.realign_annotations(1, 1, offsets), (0, 0))
    self.assertEqual(data_utils.realign_annotations(1, 2, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(1, 3, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(1, 4, offsets), (0, 1))
    self.assertEqual(data_utils.realign_annotations(1, 5, offsets), (0, 2))

    self.assertEqual(data_utils.realign_annotations(2, 2, offsets), (1, 1))
    self.assertEqual(data_utils.realign_annotations(2, 3, offsets), (1, 1))
    self.assertEqual(data_utils.realign_annotations(2, 4, offsets), (1, 1))
    self.assertEqual(data_utils.realign_annotations(2, 5, offsets), (1, 2))

    self.assertEqual(data_utils.realign_annotations(4, 4, offsets), (1, 1))
    self.assertEqual(data_utils.realign_annotations(4, 5, offsets), (1, 2))

  def extract_tokenized_mentions(self, tokenizer, text, mention):
    annotations = []
    start_pos = text.find(mention)
    while start_pos != -1:
      self.assertEqual(text[start_pos:start_pos + len(mention)], mention)
      annotations.append(
          data_utils.Annotation(start_pos, start_pos + len(mention) - 1,
                                mention))
      start_pos = text.find(mention, start_pos + 1)

    self.assertNotEmpty(annotations)
    document = data_utils.BertDocument([data_utils.Sentence(text, annotations)])
    tokenized_document = data_utils.tokenize_document_for_bert(
        document, tokenizer)
    self.assertLen(tokenized_document.sentences, 1)
    return [
        annotation.text
        for annotation in tokenized_document.sentences[0].annotations
    ]

  def get_path(self, path):
    return os.path.join(absltest.get_default_test_srcdir(), path)

  def test_split_tokenized_documents_perfect_splitting(self):
    document = data_utils.TokenizedBertDocument(
        [data_utils.TokenizedSentence(token_ids=[i]) for i in range(12)])
    expected = [
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[0]),
            data_utils.TokenizedSentence(token_ids=[1]),
            data_utils.TokenizedSentence(token_ids=[2]),
            data_utils.TokenizedSentence(token_ids=[3]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[4]),
            data_utils.TokenizedSentence(token_ids=[5]),
            data_utils.TokenizedSentence(token_ids=[6]),
            data_utils.TokenizedSentence(token_ids=[7]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[8]),
            data_utils.TokenizedSentence(token_ids=[9]),
            data_utils.TokenizedSentence(token_ids=[10]),
            data_utils.TokenizedSentence(token_ids=[11]),
        ]),
    ]

    self.assertEqual(
        expected,
        data_utils.split_tokenized_documents(
            document, max_tokens=4, max_sentences=9999))

  def test_split_tokenized_documents_imperfect_splitting_is_balanced(self):
    document = data_utils.TokenizedBertDocument(
        [data_utils.TokenizedSentence(token_ids=[i]) for i in range(11)])
    expected = [
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[0]),
            data_utils.TokenizedSentence(token_ids=[1]),
            data_utils.TokenizedSentence(token_ids=[2]),
            data_utils.TokenizedSentence(token_ids=[3]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[4]),
            data_utils.TokenizedSentence(token_ids=[5]),
            data_utils.TokenizedSentence(token_ids=[6]),
            data_utils.TokenizedSentence(token_ids=[7]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[8]),
            data_utils.TokenizedSentence(token_ids=[9]),
            data_utils.TokenizedSentence(token_ids=[10]),
        ]),
    ]

    self.assertEqual(
        expected,
        data_utils.split_tokenized_documents(
            document, max_tokens=5, max_sentences=9999))

  def test_split_tokenized_documents_splits_before_overly_long_sentence(self):
    document = data_utils.TokenizedBertDocument(
        [data_utils.TokenizedSentence(token_ids=[i] * 5) for i in range(3)])
    expected = [
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[0] * 5),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[1] * 5),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[2] * 5),
        ]),
    ]

    self.assertEqual(
        expected,
        data_utils.split_tokenized_documents(
            document, max_tokens=9, max_sentences=9999))

  def test_split_tokenized_documents_respects_max_sentences(self):
    document = data_utils.TokenizedBertDocument(
        [data_utils.TokenizedSentence(token_ids=[i]) for i in range(9)])
    expected = [
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[0]),
            data_utils.TokenizedSentence(token_ids=[1]),
            data_utils.TokenizedSentence(token_ids=[2]),
            data_utils.TokenizedSentence(token_ids=[3]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[4]),
            data_utils.TokenizedSentence(token_ids=[5]),
            data_utils.TokenizedSentence(token_ids=[6]),
            data_utils.TokenizedSentence(token_ids=[7]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[8]),
        ]),
    ]

    self.assertEqual(
        expected,
        data_utils.split_tokenized_documents(
            document, max_tokens=9999, max_sentences=4))

  def test_split_tokenized_documents_empty_trailing_sentences(self):
    # Make sure having an empty trailing sentence doesn't result in a division
    # by zero error. (In practice, we should probably just remove empty
    # sentences.)
    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(token_ids=[0]),
        data_utils.TokenizedSentence(token_ids=[]),
        data_utils.TokenizedSentence(token_ids=[1]),
        data_utils.TokenizedSentence(token_ids=[2]),
        data_utils.TokenizedSentence(token_ids=[]),
        data_utils.TokenizedSentence(token_ids=[3]),
        data_utils.TokenizedSentence(token_ids=[4]),
        data_utils.TokenizedSentence(token_ids=[]),
        data_utils.TokenizedSentence(token_ids=[]),
    ])
    expected = [
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[0]),
            data_utils.TokenizedSentence(token_ids=[]),
            data_utils.TokenizedSentence(token_ids=[1]),
            data_utils.TokenizedSentence(token_ids=[2]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[]),
            data_utils.TokenizedSentence(token_ids=[3]),
            data_utils.TokenizedSentence(token_ids=[4]),
        ]),
        data_utils.TokenizedBertDocument([
            data_utils.TokenizedSentence(token_ids=[]),
            data_utils.TokenizedSentence(token_ids=[]),
        ]),
    ]

    self.assertEqual(
        expected,
        data_utils.split_tokenized_documents(
            document, max_tokens=3, max_sentences=9999))

  def test_sentence_strip_whitespace(self):
    self.assertEqual(
        data_utils.Sentence(
            'aba caba daba',
            [data_utils.Annotation(4, 7, 'caba')]).strip_whitespaces(),
        data_utils.Sentence('aba caba daba',
                            [data_utils.Annotation(4, 7, 'caba')]))
    self.assertEqual(
        data_utils.Sentence(
            ' aba caba daba',
            [data_utils.Annotation(5, 8, 'caba')]).strip_whitespaces(),
        data_utils.Sentence('aba caba daba',
                            [data_utils.Annotation(4, 7, 'caba')]))
    self.assertEqual(
        data_utils.Sentence(
            '  aba caba daba     ',
            [data_utils.Annotation(6, 9, 'caba')]).strip_whitespaces(),
        data_utils.Sentence('aba caba daba',
                            [data_utils.Annotation(4, 7, 'caba')]))
    self.assertEqual(
        data_utils.Sentence('  aba caba daba     ', [
            data_utils.Annotation(6, 9, 'caba'),
            data_utils.Annotation(0, 1, '  '),
            data_utils.Annotation(15, 18, '   '),
        ]).strip_whitespaces(),
        data_utils.Sentence('aba caba daba',
                            [data_utils.Annotation(4, 7, 'caba')]))
    self.assertEqual(
        data_utils.Sentence(
            ' aba caba daba',
            [data_utils.Annotation(0, 3, ' aba')]).strip_whitespaces(),
        data_utils.Sentence('aba caba daba',
                            [data_utils.Annotation(0, 2, 'aba')]))

  def test_split_tokenized_sentences(self):

    def cut_sentence(document, split_lengths):
      self.assertLen(document.sentences, 1)
      sentence = document.sentences[0]
      split_lengths.append(len(sentence.token_ids))
      sentences = []
      start_index = 0
      for end_index in split_lengths:
        token_ids = sentence.token_ids[start_index:end_index]
        if sentence.is_continuation is not None:
          is_continuation = sentence.is_continuation[start_index:end_index]
        else:
          is_continuation = None
        annotations = []
        for annotation in sentence.annotations or []:
          if start_index <= annotation.begin and annotation.end < end_index:
            annotation = copy.copy(annotation)
            annotation.begin -= start_index
            annotation.end -= start_index
            annotations.append(annotation)
        sentences.append(
            data_utils.TokenizedSentence(
                token_ids=token_ids,
                is_continuation=is_continuation,
                annotations=annotations))
        start_index = end_index
      return data_utils.TokenizedBertDocument(sentences=sentences)

    def assert_split(document, max_tokens, min_tokens_for_graceful_split,
                     expected_split_length):
      actual_document = data_utils.split_tokenized_sentences(
          document,
          max_tokens=max_tokens,
          min_tokens_for_graceful_split=min_tokens_for_graceful_split)
      expected_document = cut_sentence(document, expected_split_length)
      self.assertEqual(expected_document, actual_document)

    document = data_utils.TokenizedBertDocument(
        [data_utils.TokenizedSentence(token_ids=[i for i in range(20)])])

    assert_split(document, 10, 1, [10])
    assert_split(document, 8, 1, [8, 16])
    assert_split(document, 7, 1, [7, 14])
    assert_split(document, 5, 1, [5, 10, 15])

    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[i for i in range(20)],
            is_continuation=[i % 2 for i in range(20)])
    ])

    assert_split(document, 10, 1, [10])
    assert_split(document, 8, 1, [8, 16])
    assert_split(document, 7, 1, [6, 12, 18])
    assert_split(document, 5, 1, [4, 8, 12, 16])
    assert_split(document, 7, 7, [7, 14])
    assert_split(document, 5, 5, [5, 10, 15])

    document = data_utils.TokenizedBertDocument([
        data_utils.TokenizedSentence(
            token_ids=[i for i in range(20)],
            annotations=[
                data_utils.Annotation(0, 1),
                data_utils.Annotation(5, 10),
                data_utils.Annotation(13, 19),
            ])
    ])
    assert_split(document, 10, 5, [5, 13])
    assert_split(document, 10, 10, [10])
    assert_split(document, 8, 5, [5, 13])
    assert_split(document, 7, 6, [7, 13])
    assert_split(document, 7, 7, [7, 14])


if __name__ == '__main__':
  absltest.main()
