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

"""Postprocessing mnist and cifar10/100 outputs for simple, precond, dropout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v1 as tf


def postprocess_mnist(workingdir):
  """preprocessing mnist and notmnist outputs.

  Args:
    workingdir: path to the working directory
  """
  path = os.path.join(workingdir, 'proba_tab_*.npy')
  if tf.gfile.IsDirectory(os.path.join(workingdir, 'mnist/temp')):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, 'mnist/temp'))
  if tf.gfile.IsDirectory(os.path.join(workingdir, 'notmnist/temp')):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, 'notmnist/temp'))
  tf.gfile.MakeDirs(os.path.join(workingdir, 'mnist/temp'))
  tf.gfile.MakeDirs(os.path.join(workingdir, 'notmnist/temp'))
  files_list = tf.gfile.Glob(path)
  n = len(files_list)
  for i in np.arange(n):
    path = os.path.join(workingdir, 'proba_tab_' + str(i) + '.npy')
    with tf.gfile.Open(path, 'rb') as f:
      p = np.load(f)
      p_mnist = p[:10000, :, :]
      p_notmnist = p[10000:, :, :]
      for k in np.arange(10):
        path = os.path.join(workingdir, 'mnist/temp',
                            'proba_' + str(i) + '_' + str(k) + '.npy')
        with tf.gfile.Open(path, 'wb') as f:
          np.save(f, p_mnist[k*1000:(k+1)*1000, :, :])
        path = os.path.join(workingdir, 'notmnist/temp',
                            'proba_' + str(i) + '_' + str(k) + '.npy')
        with tf.gfile.Open(path, 'wb') as f:
          np.save(f, p_notmnist[k*1000:(k+1)*1000, :, :])
  for dataset in ['mnist', 'notmnist']:
    for k in np.arange(10):
      p_list = []
      for i in np.arange(n):
        path = os.path.join(workingdir, dataset, 'temp',
                            'proba_' + str(i) + '_' + str(k) + '.npy')
        with tf.gfile.Open(path, 'rb') as f:
          p = np.load(f)
          p_list.append(p)
        proba = np.concatenate(tuple(p_list), axis=-1)
        path = os.path.join(workingdir, dataset, 'proba_' + str(k) + '.npy')
        with tf.gfile.Open(path, 'wb') as f:
          np.save(f, proba)
    tf.gfile.DeleteRecursively(os.path.join(workingdir, dataset, 'temp'))


def postprocess_cifar(workingdir, dataset):
  """preprocessing cifar10 outputs.

  Args:
    workingdir: path to the working directory
    dataset: string, 'cifar10' or cifar100'
  """
  path = os.path.join(workingdir, 'proba_tab_*.npy')
  if tf.gfile.IsDirectory(os.path.join(workingdir, dataset)):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, dataset))
  if tf.gfile.IsDirectory(os.path.join(workingdir, 'temp')):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, 'temp'))
  tf.gfile.MakeDirs(os.path.join(workingdir, dataset))
  tf.gfile.MakeDirs(os.path.join(workingdir, 'temp'))
  files_list = tf.gfile.Glob(path)
  n = len(files_list)
  for i in np.arange(n):
    path = os.path.join(workingdir, 'proba_tab_' + str(i) + '.npy')
    with tf.gfile.Open(path, 'rb') as f:
      p = np.load(f)
      for k in np.arange(10):
        path = os.path.join(workingdir, 'temp',
                            'proba_' + str(i) + '_' + str(k) + '.npy')
        with tf.gfile.Open(path, 'wb') as f:
          np.save(f, p[k*1000:(k+1)*1000, :, :])
  for k in np.arange(10):
    p_list = []
    for i in np.arange(n):
      path = os.path.join(workingdir, 'temp',
                          'proba_' + str(i) + '_' + str(k) + '.npy')
      with tf.gfile.Open(path, 'rb') as f:
        p = np.load(f)
        p_list.append(p)
      proba = np.concatenate(tuple(p_list), axis=-1)
      path = os.path.join(workingdir, dataset, 'proba_' + str(k) + '.npy')
      with tf.gfile.Open(path, 'wb') as f:
        np.save(f, proba)
  tf.gfile.DeleteRecursively(os.path.join(workingdir, 'temp'))


def postprocess_bootstrap_mnist(workingdir):
  """preprocessing mnist bootstrap outputs.

  Args:
    workingdir: path to the working directory
  """
  if tf.gfile.IsDirectory(os.path.join(workingdir, 'mnist')):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, 'mnist'))
  if tf.gfile.IsDirectory(os.path.join(workingdir, 'notmnist')):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, 'notmnist'))

  list_tasks = tf.gfile.ListDirectory(workingdir)
  num_samples = len(list_tasks)
  tf.gfile.MakeDirs(os.path.join(workingdir, 'mnist'))
  tf.gfile.MakeDirs(os.path.join(workingdir, 'notmnist'))

  for k in np.arange(10):
    p_mnist_list = []
    p_notmnist_list = []
    for i in np.arange(1, num_samples + 1):
      path_task = os.path.join(workingdir, 'task_' + str(i),
                               'proba_tab_' + str(i-1) + '.npy')
      with tf.gfile.Open(path_task, 'rb') as f:
        p = np.load(f)
        p_mnist = p[:10000, :, :]
        p_notmnist = p[10000:, :, :]
        p_mnist_list.append(p_mnist[k*1000:(k+1)*1000, :, :])
        p_notmnist_list.append(p_notmnist[k*1000:(k+1)*1000, :, :])
    proba_mnist = np.concatenate(tuple(p_mnist_list), axis=-1)
    proba_notmnist = np.concatenate(tuple(p_notmnist_list), axis=-1)
    path = os.path.join(workingdir, 'mnist', 'proba_' + str(k) + '.npy')
    with tf.gfile.Open(path, 'wb') as f:
      np.save(f, proba_mnist)
    path = os.path.join(workingdir, 'notmnist', 'proba_' + str(k) + '.npy')
    with tf.gfile.Open(path, 'wb') as f:
      np.save(f, proba_notmnist)


def postprocess_bootstrap_cifar(workingdir, dataset):
  """preprocessing cifar10 bootstrap outputs.

  Args:
    workingdir: path to the working directory
    dataset: string, 'cifar10' or cifar100'
  """
  if tf.gfile.IsDirectory(os.path.join(workingdir, dataset)):
    tf.gfile.DeleteRecursively(os.path.join(workingdir, dataset))

  list_tasks = tf.gfile.ListDirectory(workingdir)
  num_samples = len(list_tasks)
  tf.gfile.MakeDirs(os.path.join(workingdir, dataset))
  for k in np.arange(10):
    p_list = []
    for i in np.arange(1, num_samples + 1):
      path_task = os.path.join(workingdir, 'task_' + str(i),
                               'proba_tab_' + str(i-1) + '.npy')
      with tf.gfile.Open(path_task, 'rb') as f:
        p = np.load(f)
        p_list.append(p[k*1000:(k+1)*1000, :, :])
    proba = np.concatenate(tuple(p_list), axis=-1)
    path = os.path.join(workingdir, dataset, 'proba_' + str(k) + '.npy')
    with tf.gfile.Open(path, 'wb') as f:
      np.save(f, proba)
