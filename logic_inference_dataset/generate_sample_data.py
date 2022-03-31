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

"""Generates and prints a few examples for illustration purposes.
"""

import random

from absl import app
import rules
import splits


def main(_):
  random.seed(0)
  rules.precompute_rules()

  (train, test) = splits.generate_training_and_test_sets_iid(
      50, 20, 500, 0.9, answer_at_the_end=False)
  print(f"train: {len(train)}, test: {len(test)}")
  for e in train[:5]:
    print("- Train ---------------")
    print("  inputs:  " + e.inputs)
    print("  targets: " + e.targets)

  for e in test[:5]:
    print("- Test ---------------")
    print("  inputs:  " + e.inputs)
    print("  targets: " + e.targets)


if __name__ == "__main__":
  app.run(main)
