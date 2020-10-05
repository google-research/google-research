# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Ravens models package."""

from ravens.tasks.aligning import Aligning
from ravens.tasks.cable import Cable
from ravens.tasks.defs_bags import BagAloneOpen
from ravens.tasks.defs_bags import BagItemsEasy
from ravens.tasks.defs_bags import BagItemsHard
from ravens.tasks.defs_cables import CableLineNoTarget
from ravens.tasks.defs_cables import CableRing
from ravens.tasks.defs_cables import CableRingNoTarget
from ravens.tasks.defs_cables import CableShape
from ravens.tasks.defs_cables import CableShapeNoTarget
from ravens.tasks.defs_cloths import ClothCover
from ravens.tasks.defs_cloths import ClothFlat
from ravens.tasks.defs_cloths import ClothFlatNoTarget
from ravens.tasks.hanoi import Hanoi
from ravens.tasks.insertion import Insertion
from ravens.tasks.insertion import InsertionEasy
from ravens.tasks.insertion import InsertionGoal
from ravens.tasks.insertion import InsertionSixDof
from ravens.tasks.insertion import InsertionTranslation
from ravens.tasks.kitting import Kitting
from ravens.tasks.kitting import KittingEasy
from ravens.tasks.packing import Packing
from ravens.tasks.palletizing import Palletizing
from ravens.tasks.pushing import Pushing
from ravens.tasks.sorting import Sorting
from ravens.tasks.stacking import Stacking
from ravens.tasks.sweeping import Sweeping
from ravens.tasks.task import Task

# New custom tasks for "Deformable Ravens". When adding these, double check:
#   Environment._is_new_cable_env()
#   Environment._is_cloth_env()
#   Environment._is_bag_env()
# and adjust those methods as needed.


names = {
    'sorting': Sorting,
    'insertion': Insertion,
    'insertion-easy': InsertionEasy,
    'insertion-translation': InsertionTranslation,
    'insertion-sixdof': InsertionSixDof,
    'insertion-goal': InsertionGoal,
    'hanoi': Hanoi,
    'aligning': Aligning,
    'stacking': Stacking,
    'sweeping': Sweeping,
    'pushing': Pushing,
    'palletizing': Palletizing,
    'kitting': Kitting,
    'kitting-easy': KittingEasy,
    'packing': Packing,
    'cable': Cable,
    'cable-shape': CableShape,
    'cable-shape-notarget': CableShapeNoTarget,
    'cable-line-notarget': CableLineNoTarget,
    'cable-ring': CableRing,
    'cable-ring-notarget': CableRingNoTarget,
    'cloth-flat': ClothFlat,
    'cloth-flat-notarget': ClothFlatNoTarget,
    'cloth-cover': ClothCover,
    'bag-alone-open': BagAloneOpen,
    'bag-items-easy': BagItemsEasy,
    'bag-items-hard': BagItemsHard,
}
