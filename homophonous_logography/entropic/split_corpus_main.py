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

"""A tool for splitting the training data.

Given the original training data prepared in the format required by the neural
measure, splits the data into the orthographic, phonological and joint training
and testing components.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string(
    "corpus", "",
    "Path to neural training corpus.")

flags.DEFINE_string(
    "written_train", "/tmp/wtrain.txt",
    "Path to written training.")

flags.DEFINE_string(
    "written_test", "/tmp/wtest.txt",
    "Path to written test.")

flags.DEFINE_string(
    "phoneme_train", "/tmp/ptrain.txt",
    "Path to phoneme training.")

flags.DEFINE_string(
    "phoneme_test", "/tmp/ptest.txt",
    "Path to phoneme test.")

flags.DEFINE_string(
    "joint_train", "/tmp/jtrain.txt",
    "Path to the full train set.")

flags.DEFINE_string(
    "joint_test", "/tmp/jtest.txt",
    "Path to the full test set.")

FLAGS = flags.FLAGS


def _is_null_pron(line):
  return "NULLPRON" in line or "//" in line


def main(unused_argv):
  if not FLAGS.corpus:
    raise ValueError("Specify --corpus!")
  train_joint = []
  train_written = []
  train_phoneme = []
  test_joint = []
  test_written = []
  test_phoneme = []
  nonull = 0
  with open(FLAGS.corpus) as s:
    for line in s:
      if _is_null_pron(line):
        continue
      nonull += 1
      line = line.split()
      if line[0].startswith("train"):
        written = train_written
        phoneme = train_phoneme
        train_joint.append(" ".join(line[1:]))
      else:
        written = test_written
        phoneme = test_phoneme
        test_joint.append(" ".join(line[1:]))
      wline = []
      pline = []
      for w in line[1:]:
        try:
          writt, phone = w.split("/", 1)
        except ValueError:
          continue
        wline.append(writt)
        pline.append(phone)
      written.append(" ".join(wline))
      phoneme.append(" ".join(pline))
  files = [[FLAGS.written_train, train_written],
           [FLAGS.written_test, test_written],
           [FLAGS.phoneme_train, train_phoneme],
           [FLAGS.phoneme_test, test_phoneme],
           [FLAGS.joint_train, train_joint],
           [FLAGS.joint_test, test_joint]]
  for (f, d) in files:
    with open(f, "w") as s:
      for line in d:
        s.write(line + "\n")
  logging.info("Kept %d verses with no null prons.", nonull)


if __name__ == "__main__":
  app.run(main)
