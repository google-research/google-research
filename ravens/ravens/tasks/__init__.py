# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Ravens tasks."""

from ravens.tasks.align_box_corner import AlignBoxCorner
from ravens.tasks.assembling_kits import AssemblingKits
from ravens.tasks.assembling_kits import AssemblingKitsEasy
from ravens.tasks.block_insertion import BlockInsertion
from ravens.tasks.block_insertion import BlockInsertionEasy
from ravens.tasks.block_insertion import BlockInsertionNoFixture
from ravens.tasks.block_insertion import BlockInsertionSixDof
from ravens.tasks.block_insertion import BlockInsertionTranslation
from ravens.tasks.manipulating_rope import ManipulatingRope
from ravens.tasks.packing_boxes import PackingBoxes
from ravens.tasks.palletizing_boxes import PalletizingBoxes
from ravens.tasks.place_red_in_green import PlaceRedInGreen
from ravens.tasks.stack_block_pyramid import StackBlockPyramid
from ravens.tasks.stack_block_tower import StackBlockTower
from ravens.tasks.sweeping_piles import SweepingPiles
from ravens.tasks.task import Task
from ravens.tasks.towers_of_hanoi import TowersOfHanoi

names = {
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'stack-block-tower': StackBlockTower,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi
}
