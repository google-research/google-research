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

from act.models.action_classifier_model import ActionClassifierModel


class PACIFICActionClassifier(ActionClassifierModel):
  """Action classifier for PACIFIC dataset."""

  def prepend_icl_examples(self, inputs):
    icl_examples = """[Example]\nAssistant: 35 acquisitions
  Is the Assistant's response a clarifying question? Yes or No.
  No.
  [Example]
  Assistant: Which region are you asking about?
  Is the Assistant's response a clarifying question? Yes or No.
  Yes.
  [Example]
  Assistant: strengthens and scales the National Storage operating platform which drives efficiencies across the business.
  Is the Assistant's response a clarifying question? Yes or No.
  No.
  [Example]
  Assistant: What kind of recorded investment are you asking about?
  Is the Assistant's response a clarifying question? Yes or No.
  Yes.
  [Example]
  Assistant: The value of Lease Receivables was $1,419.
  Is the Assistant's response a clarifying question? Yes or No.
  No.
  [Example]
  Assistant: {}
  Is the Assistant's response a clarifying question? Yes or No.""".format(
        inputs
    )
    return icl_examples
