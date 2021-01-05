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

"""Simple interface for displaying state and accepting commands from user."""
import threading
import cv2
import inputs
import numpy as np


class BaseRenderer(object):
  """The base class for rendering a scene."""

  def render(self, env, env_state, trajs=None, hide_state=False):
    raise NotImplementedError()


class ThreadedRenderer(BaseRenderer):
  """A threaded renderer, that receives commands from user."""

  def __init__(self, width=200):
    self._width = width
    self._img = None
    self._action = np.array([0.0, 0.0])
    self._display_thread = threading.Thread(target=self.display_fn, daemon=True)
    self._display_thread.start()
    if not inputs.devices.gamepads:
      self._joystick_thread = threading.Thread(target=self.joystick_fn,
                                               daemon=True)
      self._joystick_thread.start()

  def display_fn(self):
    curr_img = None
    cv2.namedWindow("jax_particles", cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE)
    while True:
      if curr_img is not self._img:
        curr_img = self._img
        cv2.imshow("jax_particles", curr_img)
      key = cv2.waitKey(10)
      if key != -1:
        self._action = key_to_action(key)

  def joystick_fn(self):
    while True:
      events = inputs.get_gamepad()
      for event in events:
        if event.ev_type == "Absolute" and event.code == "ABS_Z":  # X value
          self._action[0] = (event.state-127.5)/127.5
        if event.ev_type == "Absolute" and event.code == "ABS_RZ":  # Y value
          self._action[1] = (event.state-127.5)/127.5

  def render(self, env, env_state, trajs=None, hide_state=False):
    # trajs are a list of lists of s,o,a,r dictionaries
    self._img = draw_scene(env, env_state, trajs, self._width, hide_state)

  def get_action(self):
    action = self._action
    return action


def draw_scene(env, env_state, trajs, width, hide_state):
  """Draws the scene of a state with trajectories overlayed."""
  if trajs is None:
    trajs = []
  scale = width/(env.max_p[0] - env.min_p[0])
  height = (int)((env.max_p[1] - env.min_p[1])*scale)
  offset = np.array([width/2.0, height/2.0])

  img = (255*np.ones((width, height, 3))).astype(np.uint8)

  for traj in trajs:
    img = draw_traj(img, env, traj, width)

  if not hide_state:
    p = env_state[0]
    for i, entity in enumerate(env.entities):
      pos = tuple((p[i, :]*scale + offset).astype(dtype=np.int32))
      rad = (int)(entity.radius*scale)
      color = entity.color
      color_alpha = entity.color_alpha
      img = draw_circle(img, pos, rad, color, color_alpha)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_circle(img, pos, rad, color, alpha):
  """Generic circle drawing function."""
  overlay = img.copy()
  cv2.circle(overlay, pos, rad, color, -1, lineType=cv2.cv2.LINE_AA)
  return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)


def draw_line(img, start, end, color, alpha):
  """Generic line drawing function."""
  overlay = img.copy()
  cv2.line(overlay, start, end, color, 1, lineType=cv2.cv2.LINE_AA)
  return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)


def draw_traj(img, env, traj, width, color=None):
  """Draw an individual trajectory."""
  if len(traj) > 1:
    scale = width/(env.max_x - env.min_x)
    height = (int)((env.max_y - env.min_y)*scale)
    offset = np.array([width/2.0, height/2.0])
    initial_state = env.get_state()  # to reset when done

    entities = env.get_entities()

    env.set_state(traj[0]["s"])
    old_entity_states = [entity.state for entity in env.get_entities()]
    for i in range(1, len(traj)):
      env.set_state(traj[i]["s"])
      entity_states = [entity.state for entity in env.get_entities()]

      for elems in zip(entities, old_entity_states, entity_states):
        entity, entity_state_start, entity_state_end = elems
        pos_start = tuple((entity_state_start.p*scale + offset).astype(np.int))
        pos_end = tuple((entity_state_end.p*scale + offset).astype(np.int))
        if pos_start[0] != pos_end[0] or pos_start[1] != pos_end[1]:
          color = (tuple((entity.color*255).astype(np.int).tolist())
                   if color is None else color)
          img = draw_line(img, pos_start, pos_end, color, entity.color_alpha)

      old_entity_states = entity_states

    env.set_state(initial_state)
  return img


def key_to_action(key):
  """Maps keyboard keys to actions."""
  action = np.array([0.0, 0.0])
  if key == ord("a") or key == 81:  # left
    action[0] -= 1.0
  if key == ord("w") or key == 82:  # up
    action[1] -= 1.0
  if key == ord("d") or key == 83:  # right
    action[0] += 1.0
  if key == ord("s") or key == 84:  # down
    action[1] += 1.0

  return action
