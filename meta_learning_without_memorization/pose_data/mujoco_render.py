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

"""Create pose regression dataset.

Given a CAD directory structured like:
CAD
  Object Class
    Object files

Render objects at different rotations and save the image and angle of rotation.

Warning: Overwrites pusher.xml in gym package.
"""
from __future__ import print_function
import contextlib
import logging
import os
import pickle
import random
import tempfile

from absl import app
from absl import flags
import gym
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import stl
from stl import mesh
import tensorflow.compat.v1 as tf

logging.getLogger("stl").setLevel(logging.ERROR)

flags.DEFINE_string("CAD_dir", None,
                    "Directory of CAD models.")
flags.DEFINE_string("data_dir", None,
                    "Directory where generated data is stored.")
FLAGS = flags.FLAGS


def find_mins_maxs(obj):
  """Find the max dimensions.

  So we can know the bounding box, getting the height, width, length
  (because these are the step size)...

  Args:
    obj: Mesh object.

  Returns:
    minx: Extreme dimension.
    maxx: Extreme dimension.
    miny: Extreme dimension.
    maxy: Extreme dimension.
    minz: Extreme dimension.
    maxz: Extreme dimension.
  """
  minx = maxx = miny = maxy = minz = maxz = None
  for p in obj.points:
    # p contains (x, y, z)
    if minx is None:
      minx = p[stl.Dimension.X]
      maxx = p[stl.Dimension.X]
      miny = p[stl.Dimension.Y]
      maxy = p[stl.Dimension.Y]
      minz = p[stl.Dimension.Z]
      maxz = p[stl.Dimension.Z]
    else:
      maxx = max(p[stl.Dimension.X], maxx)
      minx = min(p[stl.Dimension.X], minx)
      maxy = max(p[stl.Dimension.Y], maxy)
      miny = min(p[stl.Dimension.Y], miny)
      maxz = max(p[stl.Dimension.Z], maxz)
      minz = min(p[stl.Dimension.Z], minz)
  return minx, maxx, miny, maxy, minz, maxz


class MJCModel(object):
  """Mujoco Model."""

  def __init__(self, name):
    self.name = name
    self.root = MJCTreeNode("mujoco").add_attr("model", name)

  @contextlib.contextmanager
  def asfile(self):
    """Usage information.

    Usage:
    model = MJCModel('reacher')
    with model.asfile() as f:
        print f.read()  # prints a dump of the model

    Yields:
      f: File.
    """
    with tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".xml", delete=True) as f:
      self.root.write(f)
      f.seek(0)
      yield f

  def open(self):
    self.file = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".xml", delete=True)
    self.root.write(self.file)
    self.file.seek(0)
    return self.file

  def save(self, path):
    with open(path, "w") as f:
      self.root.write(f)

  def close(self):
    self.file.close()


class MJCModelRegen(MJCModel):

  def __init__(self, name, regen_fn):
    super(MJCModelRegen, self).__init__(name)
    self.regen_fn = regen_fn

  def regenerate(self):
    self.root = self.regen_fn().root


class MJCTreeNode(object):
  """Mujoco Tree Node."""

  def __init__(self, name):
    self.name = name
    self.attrs = {}
    self.children = []

  def add_attr(self, key, value):
    if isinstance(value, str):
      pass
    elif isinstance(value, list) or isinstance(value, np.ndarray):
      value = " ".join([str(val) for val in value])

    self.attrs[key] = value
    return self

  def __getattr__(self, name):

    def wrapper(**kwargs):
      newnode = MJCTreeNode(name)
      for (k, v) in kwargs.items():  # iteritems in python2
        newnode.add_attr(k, v)
      self.children.append(newnode)
      return newnode

    return wrapper

  def dfs(self):
    yield self
    if self.children:
      for child in self.children:
        for node in child.dfs():
          yield node

  def write(self, ostream, tabs=0):
    """Write out the object as a string."""
    contents = " ".join(['%s="%s"' % (k, v) for (k, v) in self.attrs.items()])
    if self.children:

      ostream.write("\t" * tabs)
      ostream.write("<%s %s>\n" % (self.name, contents))
      for child in self.children:
        child.write(ostream, tabs=tabs + 1)
      ostream.write("\t" * tabs)
      ostream.write("</%s>\n" % self.name)
    else:
      ostream.write("\t" * tabs)
      ostream.write("<%s %s/>\n" % (self.name, contents))

  def __str__(self):
    s = "<" + self.name
    s += " ".join(['%s="%s"' % (k, v) for (k, v) in self.attrs.items()])
    return s + ">"


def pusher(obj_scale=None,
           obj_mass=None,
           obj_damping=None,
           object_pos=(0.45, -0.05, -0.275),
           distr_scale=None,
           axisangle=(0, 0, 1, 1.5),
           distr_mass=None,
           distr_damping=None,
           goal_pos=(0.45, -0.05, -0.3230),
           distractor_pos=(0.45, -0.05, -0.275),
           mesh_file=None,
           mesh_file_path=None,
           distractor_mesh_file=None,
           friction=(.8, .1, .1),
           table_texture=None,
           distractor_texture=None,
           obj_texture=None,
           table_pos="0 0.5 -0.325",
           table_size="1 1 0.1"):
  """Create the pusher Mujoco object from the mesh file."""
  object_pos, goal_pos, distractor_pos, friction = \
      list(object_pos), list(goal_pos), list(distractor_pos), list(friction)
  # For now, only supports one distractor

  #     if obj_scale is None:
  #         #obj_scale = random.uniform(0.5, 1.0)
  #         obj_scale = 4.0
  if obj_mass is None:
    obj_mass = random.uniform(0.2, 2.0)
  if obj_damping is None:
    obj_damping = random.uniform(0.2, 5.0)
  obj_damping = str(obj_damping)

  if distractor_mesh_file:
    if distr_scale is None:
      distr_scale = random.uniform(0.5, 1.0)
    if distr_mass is None:
      distr_mass = random.uniform(0.2, 2.0)
    if distr_damping is None:
      distr_damping = random.uniform(0.2, 5.0)
    distr_damping = str(distr_damping)

  mjcmodel = MJCModel("arm3d")
  mjcmodel.root.compiler(
      inertiafromgeom="true", angle="radian", coordinate="local")
  mjcmodel.root.option(
      timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
  default = mjcmodel.root.default()
  default.joint(armature="0.04", damping=1, limited="true")
  default.geom(
      friction=friction,
      density="300",
      margin="0.002",
      condim="1",
      contype="0",
      conaffinity="0")

  # Make table
  worldbody = mjcmodel.root.worldbody()
  worldbody.light(diffuse=".5 .5 .5", pos="0 0 5", dir="0 0 -1")
  if table_texture:
    worldbody.geom(
        name="table",
        material="table",
        type="plane",
        pos="0 0.5 -0.325",
        size="1 1 0.1",
        contype="1",
        conaffinity="1")
  else:
    worldbody.geom(
        name="table",
        type="plane",
        pos=table_pos,
        size=table_size,
        contype="1",
        conaffinity="1")

  # Process object physical properties
  if mesh_file is not None:
    mesh_object = mesh.Mesh.from_file(mesh_file)

    vol, cog, inertia = mesh_object.get_mass_properties()
    inertia = np.abs(inertia)
    vol = np.abs(vol)
    cog = np.abs(cog)

    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh_object)
    max_length = max((maxx - minx), max((maxy - miny), (maxz - minz)))

    if max_length > 0.5:
      obj_scale = 5
    else:
      obj_scale = 7

    scale = obj_scale * 0.0012 * (200.0 / max_length)
    # print('max_length=', max_length)
    object_density = np.abs(obj_mass / (vol * scale * scale * scale))
    object_pos[0] -= scale * (minx + maxx) / 2.0
    object_pos[1] -= scale * (miny + maxy) / 2.0
    object_pos[2] = -0.324 - scale * minz
    object_scale = scale
  if distractor_mesh_file is not None:
    distr_mesh_object = mesh.Mesh.from_file(distractor_mesh_file)
    vol, cog, inertia = distr_mesh_object.get_mass_properties()

    inertia = np.abs(inertia)

    minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(distr_mesh_object)
    max_length = max((maxx - minx), max((maxy - miny), (maxz - minz)))
    distr_scale = distr_scale * 0.0012 * (200.0 / max_length)
    distr_density = distr_mass / (vol * distr_scale * distr_scale * distr_scale)
    distractor_pos[0] -= distr_scale * (minx + maxx) / 2.0
    distractor_pos[1] -= distr_scale * (miny + maxy) / 2.0
    distractor_pos[2] = -0.324 - distr_scale * minz

  ## MAKE DISTRACTOR
  if distractor_mesh_file:
    distractor = worldbody.body(name="distractor", pos=distractor_pos)
    if distractor_mesh_file is None:
      distractor.geom(
          rgba="1 1 1 1",
          type="cylinder",
          size="0.05 0.05 0.05",
          density="0.00001",
          contype="1",
          conaffinity="0")
    else:
      if distractor_texture:
        distractor.geom(
            material="distractor",
            conaffinity="0",
            contype="1",
            density=str(distr_density),
            mesh="distractor_mesh",
            rgba="1 1 1 1",
            type="mesh")
      else:
        distractor.geom(
            conaffinity="0",
            contype="1",
            density=str(distr_density),
            mesh="distractor_mesh",
            rgba="1 1 1 1",
            type="mesh")
    distractor.joint(
        name="distractor_slidey",
        type="slide",
        pos="0 0 0",
        axis="0 1 0",
        range="-10.3213 10.3",
        damping=distr_damping)
    distractor.joint(
        name="distractor_slidex",
        type="slide",
        pos="0 0 0",
        axis="1 0 0",
        range="-10.3213 10.3",
        damping=distr_damping)

  # MAKE TARGET OBJECT
  obj = worldbody.body(name="object", pos=object_pos, axisangle=axisangle)
  if mesh_file is None:
    obj.geom(
        rgba="1 1 1 1",
        type="cylinder",
        size="0.05 0.05 0.05",
        density="0.00001",
        contype="1",
        conaffinity="0")
  else:
    if obj_texture:
      obj.geom(
          material="object",
          conaffinity="0",
          contype="1",
          density=str(object_density),
          mesh="object_mesh",
          rgba="1 1 1 1",
          type="mesh")
    else:
      obj.geom(
          conaffinity="0",
          contype="1",
          density=str(object_density),
          mesh="object_mesh",
          rgba="1 1 1 1",
          type="mesh")
  obj.joint(
      name="obj_slidey",
      type="slide",
      pos="0 0 0",
      axis="0 1 0",
      range="-10.3213 10.3",
      damping=obj_damping)
  obj.joint(
      name="obj_slidex",
      type="slide",
      pos="0 0 0",
      axis="1 0 0",
      range="-10.3213 10.3",
      damping=obj_damping)

  goal = worldbody.body(name="goal", pos=goal_pos)
  goal.geom(
      rgba="1 0 0 0",
      type="cylinder",
      size="0.08 0.001 0.1",
      density="0.00001",
      contype="0",
      conaffinity="0")
  goal.joint(
      name="goal_slidey",
      type="slide",
      pos="0 0 0",
      axis="0 1 0",
      range="-10.3213 10.3",
      damping="0.5")
  goal.joint(
      name="goal_slidex",
      type="slide",
      pos="0 0 0",
      axis="1 0 0",
      range="-10.3213 10.3",
      damping="0.5")

  asset = mjcmodel.root.asset()
  if table_texture:
    asset.texture(name="table", file=table_texture, type="2d")
    asset.material(
        shininess="0.3",
        specular="1",
        name="table",
        rgba="0.9 0.9 0.9 1",
        texture="table")
  asset.mesh(
      file=mesh_file_path, name="object_mesh",
      scale=[object_scale] * 3)  # figure out the proper scale
  if obj_texture:
    asset.texture(name="object", file=obj_texture)
    asset.material(
        shininess="0.3",
        specular="1",
        name="object",
        rgba="0.9 0.9 0.9 1",
        texture="object")

  actuator = mjcmodel.root.actuator()
  actuator.motor(joint="goal_slidex", ctrlrange="-2.0 2.0", ctrllimited="true")
  tips_arm = worldbody.body(name="tips_arm", pos=goal_pos)
  tips_arm.geom(
      rgba="1 0 0 0",
      type="cylinder",
      size="0.08 0.001 0.1",
      density="0.00001",
      contype="0",
      conaffinity="0")
  return mjcmodel


def render_images(stl_file, dest_dir):
  """Render images of rotated object."""
  (sub_path, obj) = os.path.split(stl_file)
  obj = obj[:-4]  # Strip extension
  sub_name = os.path.split(sub_path)[-1]
  des_path = os.path.join(dest_dir, "rotate", sub_name, obj)
  if not os.path.exists(des_path):
    os.makedirs(des_path)

  ylabels = []
  for i in range(100):
    # np.random.seed(i**2)
    # random.seed(i**2)
    x_pos = 0.5  # random.uniform(0,1)
    y_pos = 0  # random.uniform(-1,1)
    angle_pos = random.uniform(0, 2 * np.pi)
    model = pusher(
        mesh_file=stl_file,
        mesh_file_path=stl_file,
        object_pos=(x_pos, y_pos, 0.3),
        axisangle=[0, 0, 1, angle_pos],
        table_pos="0.5 0.2 -0.4",
        table_size="1.0 1.6 1")
    model.save(os.path.join(gym.__path__[0][:-4],
                            "gym/envs/mujoco/assets/pusher.xml"))
    # print(GYM_PATH + '/gym/envs/mujoco/assets/pusher.xml')
    # copy2(args.obj_filepath, GYM_PATH+'/gym/envs/mujoco/assets')

    env = gym.envs.make("Pusher-v2")

    screen = env.render(mode="rgb_array")
    # res = cv2.resize(screen, dsize=(128,128), interpolation=cv2.INTER_AREA)
    res = screen
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(res)
    ax.set_position([0, 0, 1, 1])
    plt.savefig(des_path + "/" + str(i) + ".png")
    plt.close()

    ylabel = [i, x_pos, y_pos, angle_pos]
    ylabels.append(ylabel)

  pickle.dump(
      ylabels,
      open(des_path + "/" + "label_" + sub_name + obj, "wb"),
      protocol=2)


def main(_):
  CAD_dir = FLAGS.CAD_dir  # pylint: disable=invalid-name

  # Skip these because they are symmetric
  classes_to_skip = ["bottle", "train"]

  for obj_class in tf.io.gfile.listdir(CAD_dir):
    if obj_class in classes_to_skip:
      continue
    for obj in tf.io.gfile.listdir(os.path.join(CAD_dir, obj_class)):
      if obj.endswith(".stl"):
        print("Rendering %s from object class %s" % (obj, obj_class))
        try:
          render_images(os.path.join(CAD_dir, obj_class, obj), FLAGS.data_dir)
        except Exception:  # pylint: disable=broad-except
          print("Failed to render %s from object class %s" % (obj, obj_class))

if __name__ == "__main__":
  app.run(main)
