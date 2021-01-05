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

import os
from gtd.io import Workspace

# Set location of local data directory from environment variable
env_var = "HRL_DATA"
if env_var not in os.environ:
  assert False, env_var + " environmental variable must be set."
root = os.environ[env_var]

# define workspace
workspace = Workspace(root)
workspace.add_dir("experiments", "experiments")

workspace.add_dir("visualization_rooms", "visualization_rooms")
workspace.add_dir("montezuma_rooms", "visualization_rooms/montezuma")
workspace.add_dir("pitfall_rooms", "visualization_rooms/pitfall")
workspace.add_dir("private_eye_rooms", "visualization_rooms/private_eye")

workspace.add_dir("whitelist", "whitelist")
workspace.add_file("montezuma_whitelist", "whitelist/montezuma.txt")
workspace.add_file("pitfall_whitelist", "whitelist/pitfall.txt")
workspace.add_file("private_eye_whitelist", "whitelist/private_eye.txt")

workspace.add_file("arial", "arial.ttf")


def room_dir(domain):
  """Given a domain, returns the corresponding visualization_rooms subdir.

    Args:
        domain (str): e.g. MontezumaRevengeNoFrameskip-v4

    Returns:
        path (str): path to rooms dir
    """
  if "MontezumaRevenge" in domain:
    return workspace.montezuma_rooms
  elif "Pitfall" in domain:
    return workspace.pitfall_rooms
  elif "PrivateEye" in domain:
    return workspace.private_eye_rooms
  else:
    raise ValueError("{} not a supported domain.")


def whitelist_file(domain):
  """Given a domain, returns the corresponding whitelist subdir.

    Args:
        domain (str): e.g. MontezumaRevengeNoFrameskip-v4

    Returns:
        path (str): path to whitelist file
    """
  if "MontezumaRevenge" in domain:
    return workspace.montezuma_whitelist
  elif "Pitfall" in domain:
    return workspace.pitfall_whitelist
  elif "Pitfall" in domain:
    return workspace.private_eye_whitelist
  else:
    raise ValueError("{} not a supported domain.")
