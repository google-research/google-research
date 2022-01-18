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

import numpy as np
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl import data
from PIL import Image, ImageDraw, ImageFont


class Justification(object):

  def __init__(self, path, graph, text):
    self._path = path
    self._graph = graph
    self._text = text

  def visualize(self, state):
    """Returns a PIL Image visualizing this Justification."""

    def plot_state(canvas, state, value):
      canvas[int(state.pixel_y) - 3:int(state.pixel_y) + 3,
             int(state.pixel_x) - 3:int(state.pixel_x) + 3] = value

    def match(state1, state2):
      """Returns True if both states have the same room number and

            inventory
      """
      return state1.room_number == state2.room_number and \
              np.array_equal(state1.match_attributes, state2.match_attributes)

    canvas = state.unmodified_pixels
    x_lines, y_lines = AS.AbstractState.bucket_lines()
    x_lines = [int(x) for x in x_lines]
    y_lines = [int(x) for x in y_lines]

    canvas[:, x_lines] = np.array([0., 0., 255.])
    canvas[y_lines, :] = np.array([0., 0., 255.])

    abstract_state = AS.AbstractState(state)

    # Only draw the nodes that match on inventory and room number
    feasible_set = self._graph.feasible_set
    for feasible_node in feasible_set:
      if not match(feasible_node.abstract_state, abstract_state):
        continue

      color = np.array([36., 255., 36.])
      plot_state(canvas, feasible_node.abstract_state, color)

    for goal_edge in self._path:
      goal = goal_edge.end.abstract_state
      if match(goal, abstract_state):
        plot_state(canvas, goal, np.array([255., 109., 182.]))

    # Plot current position
    plot_state(canvas, abstract_state, np.array([255., 255., 109.]))

    image = Image.fromarray(canvas, "RGB")
    width, height = image.size
    image = image.resize((width * 2, height * 2))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), self._text, (255, 255, 255))

    font = ImageFont.truetype(data.workspace.arial, 8)
    for node in self._graph.nodes:
      if match(node.abstract_state, abstract_state):
        draw.text((node.abstract_state.pixel_x * 2 - 4,
                   node.abstract_state.pixel_y * 2 - 4),
                  str(node.uid), (255, 255, 255),
                  font=font)
    return image
