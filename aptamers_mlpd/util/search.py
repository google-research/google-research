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
"""Functions for interacting with the search.proto.
"""

from ..search import search_pb2


def create_choice_pb(sequence,
                     source_enum,
                     control_enum,
                     cluster_num=-1,
                     mutation_step=0,
                     seed_sequence=None,
                     model_score=None,
                     mutation_type=None):
  """Creates a Choice proto.

  Args:
    sequence: String DNA sequence of the aptamer selected.
    source_enum: Enumeration indicating how the aptamer was generated,
      for example, SOURCE_INVENTED or SOURNCE_IN_TEST.
    control_enum: Enumeration indicating whether this aptamer was picked as
      a control sequence, for example, NOT_CONTROL or POSITIVE_CONTROL.
    cluster_num: Integer of the id of the cluster that this sequence was in.
      Only used for sequences in the test and validation sets.
    mutation_step: Integer. The step number for mutations.
    seed_sequence: String sequence of the seed if applicable.
    model_score: Float value given by the model for this sequence.
    mutation_type: Enumeration indicating how the mutated sequence was created.
  Returns:
    A Choice proto.
  """
  choice_pb = search_pb2.Choice()
  choice_pb.aptamer_sequence = '%s' % sequence
  choice_pb.cluster_num = cluster_num
  choice_pb.source = source_enum
  choice_pb.control = control_enum
  choice_pb.seed_mutation.step = mutation_step
  if seed_sequence:
    choice_pb.seed_mutation.seed_sequence = '%s' % seed_sequence
  if mutation_type:
    choice_pb.seed_mutation.mutation_type = mutation_type
  if model_score:
    choice_pb.model_score = model_score

  return choice_pb
