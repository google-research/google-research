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

"""Tests for generate_examples_lib."""

import json
import os

from absl.testing import absltest
from etcmodel.models import tokenization
from etcmodel.models.openkp import generate_examples_lib as lib

VOCAB_PATH = 'etcmodel/models/openkp/test_data/vocab.txt'

LONG_WORD_FIRST_OCCURRENCE1 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    5,
    6,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
]

LONG_WORD_FIRST_OCCURRENCE2 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    5,
    6,
    9,
    10,
    11,
    11,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
]

LONG_WORD_FIRST_OCCURRENCE3 = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
]


class GenerateExamplesLibTest(absltest.TestCase):

  def test_open_kp_example_from_json(self):
    example_json = r"""
    {
      "url": "http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html",
      "text": "Star Trek Discovery Season 1 Director",
      "VDOM": "[{\"Id\":0,\"text\":\"Star Trek Discovery Season 1\",\"feature\":[44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0,44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0],\"start_idx\":0,\"end_idx\":5},{\"Id\":0,\"text\":\"Director\",\"feature\":[208.0,49.0,138.0,15.0,0.0,0.0,0.0,0.0,12.0,1.0,198.0,564.0,138.0,15.0,1.0,0.0,0.0,0.0,12.0,1.0],\"start_idx\":5,\"end_idx\":6}]",
      "KeyPhrases": [
        [
          "Star",
          "Trek"
        ],
        [
          "Jason",
          "Isaacs"
        ],
        [
          "Doug",
          "Jones"
        ]
      ]
    }
    """
    example = lib.OpenKpExample.from_json(example_json)

    expected = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text='Star Trek Discovery Season 1 Director',
        vdom=[
            lib.VdomElement(
                id=0,
                text='Star Trek Discovery Season 1',
                features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                parent_features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                start_idx=0,
                end_idx=5),
            lib.VdomElement(
                id=0,
                text='Director',
                features=lib.VdomFeatures(
                    x_coord=208.0,
                    width=49.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=False,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                parent_features=lib.VdomFeatures(
                    x_coord=198.0,
                    width=564.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                start_idx=5,
                end_idx=6)
        ],
        key_phrases=[
            lib.KeyPhrase(['Star', 'Trek']),
            lib.KeyPhrase(['Jason', 'Isaacs']),
            lib.KeyPhrase(['Doug', 'Jones']),
        ])

    self.assertEqual(expected, example)

  def test_open_kp_example_from_json_unlabeled(self):
    example_json = r"""
    {
      "url": "http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html",
      "text": "Star Trek Discovery Season 1 Director",
      "VDOM": "[{\"Id\":0,\"text\":\"Star Trek Discovery Season 1\",\"feature\":[44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0,44.0,728.0,78.0,45.0,1.0,0.0,1.0,0.0,20.0,0.0],\"start_idx\":0,\"end_idx\":5},{\"Id\":0,\"text\":\"Director\",\"feature\":[208.0,49.0,138.0,15.0,0.0,0.0,0.0,0.0,12.0,1.0,198.0,564.0,138.0,15.0,1.0,0.0,0.0,0.0,12.0,1.0],\"start_idx\":5,\"end_idx\":6}]"
    }
    """
    example = lib.OpenKpExample.from_json(example_json)

    expected = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text='Star Trek Discovery Season 1 Director',
        vdom=[
            lib.VdomElement(
                id=0,
                text='Star Trek Discovery Season 1',
                features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                parent_features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                start_idx=0,
                end_idx=5),
            lib.VdomElement(
                id=0,
                text='Director',
                features=lib.VdomFeatures(
                    x_coord=208.0,
                    width=49.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=False,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                parent_features=lib.VdomFeatures(
                    x_coord=198.0,
                    width=564.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                start_idx=5,
                end_idx=6)
        ],
        key_phrases=None)

    self.assertEqual(expected, example)

  def test_vdom_element_from_dict(self):
    vdom_element_json = """
    {
      "Id": 0,
      "text": "Director",
      "feature": [
        208.0,
        49.0,
        138.0,
        15.0,
        0.0,
        0.0,
        0.0,
        0.0,
        12.0,
        1.0,
        198.0,
        564.0,
        138.0,
        15.0,
        1.0,
        0.0,
        0.0,
        0.0,
        12.0,
        1.0
      ],
      "start_idx": 5,
      "end_idx": 6
    }
    """
    vdom_element = lib.VdomElement.from_dict(json.loads(vdom_element_json))

    expected = lib.VdomElement(
        id=0,
        text='Director',
        features=lib.VdomFeatures(
            x_coord=208.0,
            width=49.0,
            y_coord=138.0,
            height=15.0,
            is_block=False,
            is_inline=False,
            is_heading=False,
            is_leaf=False,
            font_size=12,
            is_bold=True),
        parent_features=lib.VdomFeatures(
            x_coord=198.0,
            width=564.0,
            y_coord=138.0,
            height=15.0,
            is_block=True,
            is_inline=False,
            is_heading=False,
            is_leaf=False,
            font_size=12,
            is_bold=True),
        start_idx=5,
        end_idx=6)

    self.assertEqual(expected, vdom_element)

  def test_font_size_cutoffs(self):
    # Cutoffs must be unique.
    self.assertEqual(
        len(lib._FONT_SIZE_CUTOFFS), len(set(lib._FONT_SIZE_CUTOFFS)))

    # Cutoffs must be in increasing order.
    self.assertEqual(lib._FONT_SIZE_CUTOFFS, sorted(lib._FONT_SIZE_CUTOFFS))

  def test_font_size_to_font_id(self):
    self.assertEqual(0, lib.font_size_to_font_id(0))
    self.assertEqual(1, lib.font_size_to_font_id(1))
    self.assertEqual(1, lib.font_size_to_font_id(8))
    self.assertEqual(2, lib.font_size_to_font_id(9))
    self.assertEqual(17, lib.font_size_to_font_id(24))
    self.assertEqual(18, lib.font_size_to_font_id(25))
    self.assertEqual(18, lib.font_size_to_font_id(29))
    self.assertEqual(19, lib.font_size_to_font_id(30))
    self.assertEqual(19, lib.font_size_to_font_id(34))
    self.assertEqual(20, lib.font_size_to_font_id(35))
    self.assertEqual(20, lib.font_size_to_font_id(39))
    self.assertEqual(21, lib.font_size_to_font_id(44))
    self.assertEqual(22, lib.font_size_to_font_id(49))
    self.assertEqual(23, lib.font_size_to_font_id(50))
    self.assertEqual(23, lib.font_size_to_font_id(100))

    self.assertEqual(24, lib.FONT_ID_VOCAB_SIZE)

  def test_etc_features_from_open_kp_example(self):
    example = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text='Star Trek Discovery Season 1 Jason Isaacs Jason Isaacs and Doug',
        vdom=[
            lib.VdomElement(
                id=0,
                text='Star Trek Discovery Season 1 Jason',
                features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                parent_features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                start_idx=0,
                end_idx=6),
            lib.VdomElement(
                id=0,
                text='Isaacs Jason Isaacs and Doug',
                features=lib.VdomFeatures(
                    x_coord=208.0,
                    width=49.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=False,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                parent_features=lib.VdomFeatures(
                    x_coord=198.0,
                    width=564.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                start_idx=6,
                end_idx=11)
        ],
        key_phrases=[
            lib.KeyPhrase(['Star', 'Trek']),
            lib.KeyPhrase(['Jason', 'Isaacs'])
        ])

    bert_vocab_path = os.path.join(absltest.get_default_test_srcdir(),
                                   VOCAB_PATH)
    config = lib.EtcFeaturizationConfig(
        long_max_length=16,
        global_max_length=4,
        url_max_code_points=80,
        bert_vocab_path=bert_vocab_path,
        do_lower_case=True)
    tokenizer = tokenization.FullTokenizer(
        config.bert_vocab_path, do_lower_case=config.do_lower_case)
    etc_features = example.to_etc_features(tokenizer, config)
    expected = lib.OpenKpEtcFeatures(
        url_code_points=[
            104, 116, 116, 112, 58, 47, 47, 48, 49, 50, 51, 112, 117, 116, 108,
            111, 99, 107, 101, 114, 46, 99, 111, 109, 47, 119, 97, 116, 99, 104,
            47, 113, 100, 55, 107, 66, 111, 100, 75, 45, 115, 116, 97, 114, 45,
            116, 114, 101, 107, 45, 100, 105, 115, 99, 111, 118, 101, 114, 121,
            45, 115, 101, 97, 115, 111, 110, 45, 49, 46, 104, 116, 109, 108, -1,
            -1, -1, -1, -1, -1, -1
        ],
        label_start_idx=[0, 7, -1],
        label_phrase_len=[2, 2, -1],
        long_token_ids=[3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 0, 0, 0],
        long_word_idx=[0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 0, 0, 0],
        long_vdom_idx=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        long_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        long_word_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        long_word_first_occurrence=LONG_WORD_FIRST_OCCURRENCE1,
        global_token_ids=[1, 1, 1, 1],
        global_input_mask=[1, 1, 0, 0],
        global_x_coords=[44.0, 208.0, 0.0, 0.0],
        global_y_coords=[78.0, 138.0, 0.0, 0.0],
        global_widths=[728.0, 49.0, 0.0, 0.0],
        global_heights=[45.0, 15.0, 0.0, 0.0],
        global_font_ids=[13, 5, 0, 0],
        global_block_indicator=[1, 0, 0, 0],
        global_inline_indicator=[0, 0, 0, 0],
        global_heading_indicator=[1, 0, 0, 0],
        global_leaf_indicator=[0, 0, 0, 0],
        global_bold_indicator=[0, 1, 0, 0],
        global_parent_x_coords=[44.0, 198.0, 0.0, 0.0],
        global_parent_y_coords=[78.0, 138.0, 0.0, 0.0],
        global_parent_widths=[728.0, 564.0, 0.0, 0.0],
        global_parent_heights=[45.0, 15.0, 0.0, 0.0],
        global_parent_font_ids=[13, 5, 0, 0],
        global_parent_heading_indicator=[1, 0, 0, 0],
        global_parent_leaf_indicator=[0, 0, 0, 0],
        global_parent_bold_indicator=[0, 1, 0, 0])
    self.assertEqual(expected, etc_features)

  def test_etc_features_with_vdom_overflow(self):
    vdom = [
        lib.VdomElement(
            id=0,
            text='Star Trek Discovery Season 1 Jason',
            features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            parent_features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            start_idx=0,
            end_idx=5),
        lib.VdomElement(
            id=0,
            text='Isaacs Jason Isaacs and Doug',
            features=lib.VdomFeatures(
                x_coord=208.0,
                width=49.0,
                y_coord=138.0,
                height=15.0,
                is_block=False,
                is_inline=False,
                is_heading=False,
                is_leaf=False,
                font_size=12,
                is_bold=True),
            parent_features=lib.VdomFeatures(
                x_coord=198.0,
                width=564.0,
                y_coord=138.0,
                height=15.0,
                is_block=True,
                is_inline=False,
                is_heading=False,
                is_leaf=False,
                font_size=12,
                is_bold=True),
            start_idx=5,
            end_idx=8)
    ]

    text = 'Star Trek Discovery Season 1 Director Jason Isaacs'
    text += ' foo' * (20 - 8)
    vdom.extend([
        lib.VdomElement(
            id=0,
            text='foo',
            features=lib.VdomFeatures(
                x_coord=208.0,
                width=49.0,
                y_coord=138.0,
                height=15.0,
                is_block=False,
                is_inline=False,
                is_heading=False,
                is_leaf=True,
                font_size=12,
                is_bold=True),
            parent_features=lib.VdomFeatures(
                x_coord=3110.0,
                width=92.0,
                y_coord=123.0,
                height=75.0,
                is_block=True,
                is_inline=False,
                is_heading=False,
                is_leaf=True,
                font_size=13,
                is_bold=True),
            start_idx=start_idx,
            end_idx=start_idx + 1) for start_idx in range(8, 20)
    ])
    example = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text=text,
        vdom=vdom,
        key_phrases=[
            lib.KeyPhrase(['Star', 'Trek']),
            lib.KeyPhrase(['Jason', 'Isaacs']),
        ])
    bert_vocab_path = os.path.join(absltest.get_default_test_srcdir(),
                                   VOCAB_PATH)
    config = lib.EtcFeaturizationConfig(
        long_max_length=16,
        global_max_length=4,
        url_max_code_points=80,
        bert_vocab_path=bert_vocab_path,
        do_lower_case=True)
    tokenizer = tokenization.FullTokenizer(
        config.bert_vocab_path, do_lower_case=config.do_lower_case)
    etc_features = example.to_etc_features(tokenizer, config)
    expected = lib.OpenKpEtcFeatures(
        url_code_points=[
            104, 116, 116, 112, 58, 47, 47, 48, 49, 50, 51, 112, 117, 116, 108,
            111, 99, 107, 101, 114, 46, 99, 111, 109, 47, 119, 97, 116, 99, 104,
            47, 113, 100, 55, 107, 66, 111, 100, 75, 45, 115, 116, 97, 114, 45,
            116, 114, 101, 107, 45, 100, 105, 115, 99, 111, 118, 101, 114, 121,
            45, 115, 101, 97, 115, 111, 110, 45, 49, 46, 104, 116, 109, 108, -1,
            -1, -1, -1, -1, -1, -1
        ],
        label_start_idx=[0, 7, -1],
        label_phrase_len=[2, 2, -1],
        long_token_ids=[3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 13, 13, 0],
        long_word_idx=[0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 0],
        long_vdom_idx=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 0],
        long_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        long_word_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        long_word_first_occurrence=LONG_WORD_FIRST_OCCURRENCE2,
        global_token_ids=[1, 1, 1, 1],
        global_input_mask=[1, 1, 1, 1],
        global_x_coords=[44.0, 208.0, 208.0, 208.0],
        global_y_coords=[78.0, 138.0, 138.0, 138.0],
        global_widths=[728.0, 49.0, 49.0, 49.0],
        global_heights=[45.0, 15.0, 15.0, 15.0],
        global_font_ids=[13, 5, 5, 5],
        global_block_indicator=[1, 0, 0, 0],
        global_inline_indicator=[0, 0, 0, 0],
        global_heading_indicator=[1, 0, 0, 0],
        global_leaf_indicator=[0, 0, 1, 1],
        global_bold_indicator=[0, 1, 1, 1],
        global_parent_x_coords=[44.0, 198.0, 3110.0, 3110.0],
        global_parent_y_coords=[78.0, 138.0, 123.0, 123.0],
        global_parent_widths=[728.0, 564.0, 92.0, 92.0],
        global_parent_heights=[45.0, 15.0, 75.0, 75.0],
        global_parent_font_ids=[13, 5, 6, 6],
        global_parent_heading_indicator=[1, 0, 0, 0],
        global_parent_leaf_indicator=[0, 0, 1, 1],
        global_parent_bold_indicator=[0, 1, 1, 1])

    self.assertEqual(expected, etc_features)

  def test_etc_features_with_long_overflow(self):
    text = 'Star Wars and not Trek ' + ' '.join(['star'] * 12)
    vdom = [
        lib.VdomElement(
            id=0,
            text='Star Wars and not Trek',
            features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            parent_features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            start_idx=0,
            end_idx=5),
        lib.VdomElement(
            id=0,
            text=' '.join(['star'] * 99),
            features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            parent_features=lib.VdomFeatures(
                x_coord=44.0,
                width=728.0,
                y_coord=78.0,
                height=45.0,
                is_block=True,
                is_inline=False,
                is_heading=True,
                is_leaf=False,
                font_size=20,
                is_bold=False),
            start_idx=5,
            end_idx=17)
    ]
    example = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text=text,
        vdom=vdom,
        key_phrases=[
            lib.KeyPhrase(['Star', 'Wars']),
            lib.KeyPhrase(['Trek']),
        ])
    bert_vocab_path = os.path.join(absltest.get_default_test_srcdir(),
                                   VOCAB_PATH)
    config = lib.EtcFeaturizationConfig(
        long_max_length=16,
        global_max_length=4,
        url_max_code_points=80,
        bert_vocab_path=bert_vocab_path,
        do_lower_case=True)
    tokenizer = tokenization.FullTokenizer(
        config.bert_vocab_path, do_lower_case=config.do_lower_case)
    etc_features = example.to_etc_features(tokenizer, config)
    expected = lib.OpenKpEtcFeatures(
        url_code_points=[
            104, 116, 116, 112, 58, 47, 47, 48, 49, 50, 51, 112, 117, 116, 108,
            111, 99, 107, 101, 114, 46, 99, 111, 109, 47, 119, 97, 116, 99, 104,
            47, 113, 100, 55, 107, 66, 111, 100, 75, 45, 115, 116, 97, 114, 45,
            116, 114, 101, 107, 45, 100, 105, 115, 99, 111, 118, 101, 114, 121,
            45, 115, 101, 97, 115, 111, 110, 45, 49, 46, 104, 116, 109, 108, -1,
            -1, -1, -1, -1, -1, -1
        ],
        label_start_idx=[0, 4, -1],
        label_phrase_len=[2, 1, -1],
        long_token_ids=[3, 14, 11, 15, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        long_word_idx=[0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        long_vdom_idx=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        long_input_mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        long_word_input_mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        long_word_first_occurrence=LONG_WORD_FIRST_OCCURRENCE3,
        global_token_ids=[1, 1, 1, 1],
        global_input_mask=[1, 0, 0, 0],
        global_x_coords=[44.0, 0, 0, 0],
        global_y_coords=[78.0, 0, 0, 0],
        global_widths=[728.0, 0, 0, 0],
        global_heights=[45.0, 0, 0, 0],
        global_font_ids=[13, 0, 0, 0],
        global_block_indicator=[1, 0, 0, 0],
        global_inline_indicator=[0, 0, 0, 0],
        global_heading_indicator=[1, 0, 0, 0],
        global_leaf_indicator=[0, 0, 0, 0],
        global_bold_indicator=[0, 0, 0, 0],
        global_parent_x_coords=[44.0, 0, 0, 0],
        global_parent_y_coords=[78.0, 0, 0, 0],
        global_parent_widths=[728.0, 0, 0, 0],
        global_parent_heights=[45.0, 0, 0, 0],
        global_parent_font_ids=[13, 0, 0, 0],
        global_parent_heading_indicator=[1, 0, 0, 0],
        global_parent_leaf_indicator=[0, 0, 0, 0],
        global_parent_bold_indicator=[0, 0, 0, 0])

    self.assertEqual(expected, etc_features)

  def test_etc_features_fixed_global_blocks(self):
    example = lib.OpenKpExample(
        url='http://0123putlocker.com/watch/qd7kBodK-star-trek-discovery-season-1.html',
        text='Star Trek Discovery Season 1 Jason Isaacs Jason Isaacs and Doug',
        vdom=[
            lib.VdomElement(
                id=0,
                text='Star Trek Discovery Season 1 Jason',
                features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                parent_features=lib.VdomFeatures(
                    x_coord=44.0,
                    width=728.0,
                    y_coord=78.0,
                    height=45.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=True,
                    is_leaf=False,
                    font_size=20,
                    is_bold=False),
                start_idx=0,
                end_idx=6),
            lib.VdomElement(
                id=0,
                text='Isaacs Jason Isaacs and Doug',
                features=lib.VdomFeatures(
                    x_coord=208.0,
                    width=49.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=False,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                parent_features=lib.VdomFeatures(
                    x_coord=198.0,
                    width=564.0,
                    y_coord=138.0,
                    height=15.0,
                    is_block=True,
                    is_inline=False,
                    is_heading=False,
                    is_leaf=False,
                    font_size=12,
                    is_bold=True),
                start_idx=6,
                end_idx=11)
        ],
        key_phrases=[
            lib.KeyPhrase(['Star', 'Trek']),
            lib.KeyPhrase(['Jason', 'Isaacs'])
        ])

    bert_vocab_path = os.path.join(absltest.get_default_test_srcdir(),
                                   VOCAB_PATH)
    config = lib.EtcFeaturizationConfig(
        long_max_length=16,
        global_max_length=4,
        url_max_code_points=80,
        bert_vocab_path=bert_vocab_path,
        do_lower_case=True,
        fixed_block_len=4)
    tokenizer = tokenization.FullTokenizer(
        config.bert_vocab_path, do_lower_case=config.do_lower_case)
    etc_features = example.to_etc_features(tokenizer, config)
    expected = lib.OpenKpEtcFeatures(
        url_code_points=[
            104, 116, 116, 112, 58, 47, 47, 48, 49, 50, 51, 112, 117, 116, 108,
            111, 99, 107, 101, 114, 46, 99, 111, 109, 47, 119, 97, 116, 99, 104,
            47, 113, 100, 55, 107, 66, 111, 100, 75, 45, 115, 116, 97, 114, 45,
            116, 114, 101, 107, 45, 100, 105, 115, 99, 111, 118, 101, 114, 121,
            45, 115, 101, 97, 115, 111, 110, 45, 49, 46, 104, 116, 109, 108, -1,
            -1, -1, -1, -1, -1, -1
        ],
        label_start_idx=[5, 0, -1],
        label_phrase_len=[2, 2, -1],
        long_token_ids=[3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10, 11, 12, 0, 0, 0],
        long_word_idx=[0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 10, 0, 0, 0],
        long_vdom_idx=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 0, 0, 0],
        long_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        long_word_input_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        global_token_ids=[1, 1, 1, 1],
        global_input_mask=[1, 1, 1, 1],
        global_x_coords=[],
        global_y_coords=[],
        global_widths=[],
        global_heights=[],
        global_font_ids=[],
        global_block_indicator=[],
        global_inline_indicator=[],
        global_heading_indicator=[],
        global_leaf_indicator=[],
        global_bold_indicator=[],
        global_parent_x_coords=[],
        global_parent_y_coords=[],
        global_parent_widths=[],
        global_parent_heights=[],
        global_parent_font_ids=[],
        global_parent_heading_indicator=[],
        global_parent_leaf_indicator=[],
        global_parent_bold_indicator=[])
    self.assertEqual(expected, etc_features)


if __name__ == '__main__':
  absltest.main()
