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

"""Import tasks and add them to cliport."""

from cliport.tasks import names
from tasks.put_block_in_bowl import PutBlockInBowlFullColors
from tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
from tasks.put_in_bowl import PutInBowl
from tasks.put_in_bowl import PutInBowlFullColor
from tasks.put_in_bowl import PutInBowlSeenColor
from tasks.put_in_bowl import PutInBowlSimple
from tasks.put_in_bowl import PutInBowlUnseenColor
from tasks.put_in_zone import PutInZone
from tasks.put_in_zone import PutInZoneFullColor
from tasks.put_in_zone import PutInZoneSeenColor
from tasks.put_in_zone import PutInZoneUnseenColor


names.update({
    'put-in-bowl-seen': PutInBowlSeenColor,
    'put-in-bowl-simple': PutInBowlSimple,
    'put-in-bowl-unseen': PutInBowlUnseenColor,
    'put-in-bowl-full': PutInBowlFullColor,
    'put-block-in-bowl-full': PutBlockInBowlFullColors,
    'put-block-in-bowl-seen': PutBlockInBowlSeenColors,
    'put-block-in-blue-bowl-unseen': PutBlockInBowlUnseenColors,
    'put-in-zone-seen': PutInZoneSeenColor,
    'put-in-zone-unseen': PutInZoneUnseenColor,
    'put-in-zone-full': PutInZoneFullColor,
    'put-in-zone': PutInZone,
})
