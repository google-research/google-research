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

"""Tests for encoders.py."""

from absl.testing import absltest
from absl.testing import parameterized
from ipagnn.datasets.control_flow_programs.encoders import encoders


class EncodersTest(parameterized.TestCase):

  def test_simple_encoder_placeholder(self):
    encoder = encoders.SimplePythonSourceEncoder(
        base=10,
        num_digits=1,
        ops=['+=', '-=', '*=', '=', 'while >'],
        num_variables=10,
    )
    python_source = """
v0 += 1
v1 = 2
while v1 > 0:
  v1 -= 1
  _ = 0
    """.strip()
    encoded = encoder.encode(python_source)
    # Meaning: indent, op, var, operand
    # Offsets: 1, 10, 15, 1
    target = [
        1, 11, 16, 2,  # v0 += 1
        1, 14, 17, 3,
        1, 15, 17, 1,
        2, 12, 17, 2,
        2, 27, 27, 1,  # _ = 0
    ]
    self.assertEqual(encoded, target)

  def test_simple_encoder_code_representation(self):
    encoder = encoders.SimplePythonSourceEncoder(
        base=10,
        num_digits=1,
        ops=['+=', '-=', '*=', '=', 'while >'],
        num_variables=10,
    )
    python_source = """
v0 += 2
v0 -= 3
v0 *= 4
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 *= 3
v0 += 4
    """.strip()
    encoded = encoder.encode(python_source)
    # Meaning: indent, op, var, operand
    # Offsets: 1, 10, 15, 1
    target = [
        1, 11, 16, 3,  # v0 += 2
        1, 12, 16, 4,
        1, 13, 16, 5,
        1, 14, 17, 6,
        1, 15, 17, 1,
        2, 12, 17, 2,
        2, 11, 16, 3,
        2, 13, 16, 4,
        1, 11, 16, 5,  # v0 += 4
    ]
    self.assertEqual(encoded, target)

  def test_simple_encoder_encode_decode_round_trip(self):
    encoder = encoders.SimplePythonSourceEncoder(
        base=10,
        num_digits=1,
        ops=['+=', '-=', '*=', '=', 'while >'],
        num_variables=10,
    )
    python_source = """
v0 += 2
v0 -= 3
v0 *= 4
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 *= 3
v0 += 4
    """.strip()
    encoded = encoder.encode(python_source)
    decoded = encoder.decode(encoded)
    self.assertEqual(python_source, decoded)

  @parameterized.parameters(
      (10, 2, 4, [1, 0, 1, 0]),
      (10, 3, 5, [0, 0, 1, 0, 1]),
      (42, 10, 4, [0, 0, 4, 2]),
  )
  def test_as_nary_list(self, number, base, length, target):
    encoded = encoders.as_nary_list(number, base=base, length=length)
    self.assertEqual(encoded, target)

  @parameterized.parameters(
      ([1, 0, 1, 0], 2, 10),
      ([0, 0, 1, 0, 1], 3, 10),
      ([0, 0, 4, 2], 10, 42),
  )
  def test_nary_list_as_number(self, nary_list, base, target):
    decoded = encoders.nary_list_as_number(nary_list, base=base)
    self.assertEqual(decoded, target)

  def test_text_source_encoder_encode_decode_round_trip(self):
    encoder = encoders.TextSourceEncoder(fragment_length=15)
    python_source = """
v0 += 2
v0 -= 3
v0 *= 4
v1 = 5
while v1 > 0:
  v1 -= 1
  v0 += 2
  v0 *= 3
v0 += 4
    """.strip()
    encoded = encoder.encode(python_source)
    # pylint: disable=bad-whitespace, bad-continuation
    target = [
        121,  51,  35,  46,  64, 35,  53,  0,  0,  0,  0,  0,  0, 0, 0, 2,
        121,  51,  35,  48,  64, 35,  54,  0,  0,  0,  0,  0,  0, 0, 0, 2,
        121,  51,  35,  45,  64, 35,  55,  0,  0,  0,  0,  0,  0, 0, 0, 2,
        121,  52,  35,  64,  35, 56,   0,  0,  0,  0,  0,  0,  0, 0, 0, 2,
        122, 107, 108, 111, 104, 35, 121, 52, 35, 65, 35, 51, 61, 0, 0, 2,
         35,  35, 121,  52,  35, 48,  64, 35, 52,  0,  0,  0,  0, 0, 0, 2,
         35,  35, 121,  51,  35, 46,  64, 35, 53,  0,  0,  0,  0, 0, 0, 2,
         35,  35, 121,  51,  35, 45,  64, 35, 54,  0,  0,  0,  0, 0, 0, 2,
        121,  51,  35,  46,  64, 35,  55,  0,  0,  0,  0,  0,  0, 0, 0, 2,
    ]
    # pylint: enable=bad-whitespace, bad-continuation
    self.assertEqual(encoded, target)

    decoded = encoder.decode(encoded)
    self.assertEqual(decoded, python_source)

if __name__ == '__main__':
  absltest.main()
