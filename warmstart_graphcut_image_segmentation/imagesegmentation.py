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

"""This file is adapted from "imagesegmentation.py" in the github repo

"image-segmentation".

A great portion of the functions are reused from the original file, while the
others are changed or re-implemented to work in our problem framework.

The functions used to:
    - resize the input image
    - collect manually selected seeds and store them for later use on image
    sequences
    - construct the graph based on the given seeds and other parameters
    - perform imagesegmentation on an image by calling the other funcs above and
    the ford-fulkerson subroutine from augmentingPath.py
One can run this file on a chosen image to choose a set of seeds, and then apply
those seeds to the whole sequence.
"""

from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse
from math import exp, pow
from collections import defaultdict
from augmentingPath import augmentingPath
import time

# currently only support "ap" which is Ford-Fulkerson, later might include other
# graph-cut algorithms such as push-relabel.
graphCutAlgo = {"ap": augmentingPath}
ALL_GROUPS = ["head", "birdhouse", "shoe", "dog"]
"""
The following parameter settings are adapted from the original "imagesegmentation.py" file,
with some new parameters introduced.
"""

SIGMA = 60
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 5
LOADSEEDS = False
SCALE = 100
# drawing = False

default_size = 30
radius = 10
thickness = -1  # fill the whole circle
"""
This is function show_image() in the original "imagesegmentation.py".
"""


def show_image(image):
  windowname = "Segmentation"
  cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
  cv2.startWindowThread()
  cv2.imshow(windowname, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return


"""
Adapted from plantSeed() in the original "imagesegmentation.py", but slightly changed the function drawLines().
When storing the seeds in the matrix "seeds", the original plantSeed() draws a ball around the human-selected target seed;
whereas the new function puts only the selected target sees in the matrix for more convenient use later.
These matrices will be read during warm-start and the seeds will be applied to all images from the same sequence.
"""


def plantSeed(image):

  def drawLines(x, y, pixelType):
    if pixelType == OBJ:
      color, code = OBJCOLOR, OBJCODE
    else:
      color, code = BKGCOLOR, BKGCODE
    cv2.circle(image, (x, y), radius, color, thickness)
    cv2.circle(seeds, (x // SF, y // SF), 0, code, thickness)
    return

  def onMouse(event, x, y, flags, pixelType):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
      drawing = True
      drawLines(x, y, pixelType)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
      drawLines(x, y, pixelType)
    elif event == cv2.EVENT_LBUTTONUP:
      drawing = False
    return

  def paintSeeds(pixelType):
    print("Planting", pixelType, "seeds")
    global drawing
    drawing = False
    windowname = "Plant " + pixelType + " seeds"
    cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(windowname, onMouse, pixelType)
    while (1):
      cv2.imshow(windowname, image)
      # when pressed ESC, close the window
      if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.destroyAllWindows()
    return

  seeds = np.zeros(image.shape, dtype="uint8")
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  # enlarge the pictures by SF ratio to make it easier to select seeds
  image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

  global drawing
  drawing = False

  paintSeeds(OBJ)
  paintSeeds(BKG)
  return seeds, image


"""
Adapted from boundaryPenalty() function in the original "imagesegmentation.py", but added int() to make sure the capacities are integral.
Large when ip - iq < sigma, and small otherwise
"""


def boundaryPenalty(ip, iq):
  bp = int(SCALE * exp(-pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2))))
  return bp


"""
Given a matrix "seeds" representing where the obj/bkg seeds are, the rows and columns of the image and the radius,
rescale the location of the target seeds into the r * c image and then draw a 2D ball around each of them and make all pixels in the balls seeds as well.
For example, if the shape of matrix "seeds" is 30 *　30 but the size of image is 60 * 60, if seeds[x][y] == 1,
scaled_seeds[2 * x][2 * y] will be 1 and a ball of radius "radius" of pixels around scaled_seeds[2 * x][2 * y] will also be 1.
"""


def ScaleSeeds(seeds, r, c, radius):
  r0, c0 = seeds.shape
  scaled_seeds = np.zeros((r, c))
  for i in range(r0):
    for j in range(c0):
      if seeds[i][j] in [OBJCODE, BKGCODE]:
        x = i * r // r0
        y = j * c // c0
        cv2.circle(scaled_seeds, (y, x), radius, int(seeds[i][j]), thickness)
  return scaled_seeds


"""
A re-implementation of buildGraph() in  to suit our framework.
For an image, if loaded_seeds == None, the function asks the user to select obj/bkg seeds.
Otherwise, it uses loaded_seeds as seeds to construct the graph, and run Ford-Fulkerson algorithm.
"""


def buildGraph(image, loaded_seeds=None, seeds_dir=None):
  # here image.size = width * height, so if 30 * 30 the size is 900
  # + 2 accounts for two terminals: source and sink
  V = image.size + 2

  graph = {i: defaultdict(int) for i in range(V)}
  # graph = np.zeros((V, V), dtype='int32')

  K = makeNLinks(graph, image)
  if loaded_seeds is None:
    seeds, seededImage = plantSeed(image)
    np.savetxt(seeds_dir, seeds, delimiter=",")
    # np.savetxt('./seeds.csv', seeds, delimiter=',')
  else:
    seeds = loaded_seeds
    seededImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    seededImage = cv2.resize(seededImage, (0, 0), fx=SF, fy=SF)

    scale_ratio = seededImage.shape[0] // seeds.shape[0]
    seed_radius = seededImage.shape[0] // default_size
    for i in range(seeds.shape[0]):
      for j in range(seeds.shape[1]):
        if seeds[i][j] in [OBJCODE, BKGCODE]:
          color = {OBJCODE: OBJCOLOR, BKGCODE: BKGCOLOR}[seeds[i][j]]
          cv2.circle(seededImage, (j * scale_ratio, i * scale_ratio),
                     seed_radius, color, thickness)
  seeds = ScaleSeeds(seeds, image.shape[0], image.shape[1],
                     image.shape[0] // default_size)

  # changing this line to make it very big
  makeTLinks(graph, seeds, SCALE * V**2)
  return graph, seededImage


"""
The same as the makeNLinks() function in the "imagesegmentation.py" file.
"""


def makeNLinks(graph, image):
  K = -float("inf")
  r, c = image.shape
  for i in range(r):
    for j in range(c):
      x = i * c + j
      if i + 1 < r:  # pixel below
        y = (i + 1) * c + j
        bp = boundaryPenalty(image[i][j], image[i + 1][j])
        graph[x][y] = graph[y][x] = bp
        K = max(K, bp)
      if j + 1 < c:  # pixel to the right
        y = i * c + j + 1
        bp = boundaryPenalty(image[i][j], image[i][j + 1])
        graph[x][y] = graph[y][x] = bp
        K = max(K, bp)
  return K


"""
The same as the makeTLinks() function in the "imagesegmentation.py" file.
"""


def makeTLinks(graph, seeds, K):
  r, c = seeds.shape

  for i in range(r):
    for j in range(c):
      x = i * c + j
      if seeds[i][j] == OBJCODE:
        graph[SOURCE][x] = K
      elif seeds[i][j] == BKGCODE:
        graph[x][SINK] = K
  return


def displayCut(image, cuts):

  def colorPixel(i, j):
    image[i][j] = CUTCOLOR

  r, c = image.shape
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  for c in cuts:
    if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
      colorPixel(c[0] // r, c[0] % r)
      colorPixel(c[1] // r, c[1] % r)
  return image


"""
Partially adapted from the imageSegmentation() function from imagesegmentation.py.
"""


def imageSegmentation(imagename,
                      folder,
                      group,
                      size=(30, 30),
                      algo="ap",
                      loadseed="yes"):
  # pathname = os.path.splitext(imagefile)[0]
  imagefile = folder + "/" + group + "_cropped" + "/" + imagename
  print("image at: ", imagefile)
  image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
  print("original dimensions: ", image.shape)
  # note that size[0] = width = # of cols, size[1] = height = # of rows
  image = cv2.resize(image, size)

  # stored in sequential_datasets/groupname_resized/imagename_resized.jpg
  # cv2.imwrite(pathname + './' + "resized.jpg", image)
  # seeds_directory = "./seeds.csv"
  cutdir = folder + "/" + group + "_cuts/" + str(size[0])
  if not os.path.exists(cutdir):
    os.makedirs(cutdir)

  seeds_directory = cutdir + "/" + group + "_seeds.csv"
  seeds = np.loadtxt(
      seeds_directory, delimiter=",").astype(int) if loadseed == "yes" else None

  seeded_image_dir = folder + "/" + group + "_seeded/" + str(size[0])
  if not os.path.exists(seeded_image_dir):
    os.makedirs(seeded_image_dir)

  V = size[0] * size[1] + 2
  global SOURCE, SINK
  SOURCE, SINK = -2, -1
  # SOURCE and SINK are the last two nodes
  SOURCE += V
  SINK += V

  graph, seededImage = buildGraph(image, seeds, seeds_directory)

  if seededImage is not None:
    cv2.imwrite(
        seeded_image_dir + "/" + imagename.split(".")[0] + "_seeded.jpg",
        seededImage)

  begin = time.time()

  flows, cuts, path_count, average_path_len = graphCutAlgo[algo](graph, V,
                                                                 SOURCE, SINK)
  end = time.time()
  print("image segmentation time spent in seconds:")
  print(end - begin)

  min_cut = sum([graph[x][y] for x, y in cuts])
  image = displayCut(image, cuts)
  image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
  # show_image(image)

  savename = cutdir + "/" + imagename.split(".")[0] + "_cuts.jpg"
  cv2.imwrite(savename, image)
  print("Saved image as", savename)
  print(min_cut, path_count, average_path_len)
  return flows, min_cut, path_count, average_path_len, graph, end - begin


"""
Adapated from parseArgs() with more input options.
"""


def parseArgs():

  def algorithm(string):
    if string in graphCutAlgo:
      return string
    raise argparse.ArgumentTypeError(
        "Algorithm should be one of the following:", graphCutAlgo.keys())

  def imagegroup(string):
    if string in ALL_GROUPS or string == "all":
      return string
    raise argparse.ArgumentTypeError(
        "Currently only support the following image groups:", ALL_GROUPS)

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", "-i", default="all", type=str)
  parser.add_argument("--folder", "-f", default="./sequential_datasets")
  parser.add_argument("--group", "-g", default="birdhouse", type=imagegroup)
  parser.add_argument("--size", "-s", default="30", type=int)
  parser.add_argument("--algo", "-a", default="ap", type=algorithm)
  parser.add_argument("--loadseed", "-l", default="yes")
  return parser.parse_args()


if __name__ == "__main__":
  args = parseArgs()
  flows, min_cut, path_count, average_path_len, graph, time = imageSegmentation(
      args.image, args.folder, args.group, (args.size, args.size), args.algo,
      args.loadseed)
