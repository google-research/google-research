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

"""Ravens agents package."""

from ravens.agents.conv_mlp import PickPlaceConvMlpAgent
from ravens.agents.dummy import DummyAgent
from ravens.agents.form2fit import Form2FitAgent
from ravens.agents.gt_state import GtState6DAgent
from ravens.agents.gt_state import GtStateAgent
from ravens.agents.gt_state_2_step import GtState2StepAgent
from ravens.agents.gt_state_2_step import GtState3Step6DAgent
from ravens.agents.transporter import GoalNaiveTransporterAgent
from ravens.agents.transporter import GoalTransporterAgent
from ravens.agents.transporter import NoTransportTransporterAgent
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.agents.transporter import PerPixelLossTransporterAgent
from ravens.agents.transporter_6dof import Transporter6dAgent

names = {'dummy': DummyAgent,
         'transporter': OriginalTransporterAgent,
         'transporter_6d': Transporter6dAgent,
         'no_transport': NoTransportTransporterAgent,
         'per_pixel_loss': PerPixelLossTransporterAgent,
         'conv_mlp': PickPlaceConvMlpAgent,
         'form2fit': Form2FitAgent,
         'gt_state': GtStateAgent,
         'gt_state_2_step': GtState2StepAgent,
         'gt_state_6d': GtState6DAgent,
         'gt_state_6d_3_step': GtState3Step6DAgent,
         'transporter-goal': GoalTransporterAgent,
         'transporter-goal-naive': GoalNaiveTransporterAgent}
