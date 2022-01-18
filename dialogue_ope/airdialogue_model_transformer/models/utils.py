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

INTENT_DICT = {
    "book": 0,
    "change": 1,
    "cancel": 2,
    0: "book",
    1: "change",
    2: "cancel",
}

STATUS_DICT = {
    "book": 0,
    "no_flight": 1,
    "no_reservation": 2,
    "cancel": 3,
    0: "book",
    1: "no_flight",
    2: "no_reservation",
    3: "cancel",
}


def intent_to_status(flight, res, intent):
  """
    return status, flight
    """
  if intent == 0:
    if flight == 0:
      return 1, 0
    else:
      return 0, flight
  if intent == 1:
    if not res:
      return 2, 0
    else:
      if flight == 0:
        return 1, 0
      else:
        return 0, flight
  if intent == 2:
    if not res:
      return 2, 0
    else:
      return 3, 0
