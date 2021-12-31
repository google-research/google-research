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

"""Tests for template_programs.py."""

from absl import logging  # pylint: disable=unused-import
from absl.testing import absltest
import numpy as np

from ipagnn.datasets.control_flow_programs.program_generators import template_programs
from ipagnn.datasets.control_flow_programs.program_generators.templates import arithmetic as arithmetic_templates
from ipagnn.datasets.control_flow_programs.program_generators.templates import multivar as multivar_templates
from ipagnn.datasets.control_flow_programs.program_generators.templates import runtime_error as runtime_error_templates


class TemplateProgramsTest(absltest.TestCase):

  def test_short_arithmetic_programs(self):
    length = 2
    length_distribution = template_programs.get_train_length_distribution(
        length)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=arithmetic_templates.get_template_data)
    for _ in range(100):
      python_source = template_programs.generate_python_source(length, config)
      self.assertIn('v0', python_source)
      self.assertLen(python_source.split('\n'), length)

  def test_arithmetic_programs(self):
    length = 10
    length_distribution = template_programs.get_train_length_distribution(
        length)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=arithmetic_templates.get_template_data)
    for _ in range(100):
      python_source = template_programs.generate_python_source(length, config)
      self.assertIn('v0', python_source)
      self.assertLessEqual(len(python_source.split('\n')), length)

  def test_multivar_programs(self):
    length = 15
    length_distribution = template_programs.get_train_length_distribution(
        length, min_train_length=7)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=multivar_templates.get_template_data)
    rng = np.random.RandomState(0)
    for _ in range(100):
      python_source = template_programs.generate_python_source(
          length, config, rng)
      self.assertIn('v0', python_source)
      self.assertLessEqual(len(python_source.split('\n')), length)

  def test_longer_multivar_programs(self):
    length = 25
    length_distribution = template_programs.get_train_length_distribution(
        length, min_train_length=7)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=multivar_templates.get_template_data)
    rng = np.random.RandomState(0)
    for _ in range(5):
      python_source = template_programs.generate_python_source(
          length, config, rng)
      self.assertIn('v0', python_source)
      self.assertLessEqual(len(python_source.split('\n')), length)

  def test_longer_runtime_error_programs(self):
    length = 10
    length_distribution = template_programs.get_train_length_distribution(
        length, min_train_length=3)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=runtime_error_templates.get_template_data)
    rng = np.random.RandomState(0)
    for _ in range(50):
      python_source = template_programs.generate_python_source(
          length, config, rng)
      logging.info(python_source)
      self.assertIn('v0', python_source)
      self.assertLessEqual(len(python_source.split('\n')), length)

  def test_runtime_error_programs(self):
    length = 10
    length_distribution = template_programs.get_train_length_distribution(
        length, min_train_length=3)
    config = template_programs.TemplatesConfig(
        base=1000,
        length=length,
        max_value=9,
        mod=1000,
        output_mod=1000,
        length_distribution=length_distribution,
        template_data_fn=runtime_error_templates.get_template_data)
    for _ in range(50):
      python_source = template_programs.generate_python_source(length, config)
      logging.info(python_source)
      self.assertIn('v0', python_source)
      self.assertLessEqual(len(python_source.split('\n')), length)


if __name__ == '__main__':
  absltest.main()
