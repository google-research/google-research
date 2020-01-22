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

r"""Evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pdb
import sys
import traceback

from absl import flags
from absl import logging
from monty.collections import AttrDict
import sklearn.cluster
import tensorflow.compat.v1 as tf

from stacked_capsule_autoencoders.capsules.configs import data_config
from stacked_capsule_autoencoders.capsules.configs import model_config
from stacked_capsule_autoencoders.capsules.eval import cluster_classify
from stacked_capsule_autoencoders.capsules.eval import collect_results
from stacked_capsule_autoencoders.capsules.plot import make_tsne_plot
from stacked_capsule_autoencoders.capsules.train import tools

flags.DEFINE_string('snapshot', '', 'Checkpoint file.')
flags.DEFINE_string('tsne_figure_name', 'tsne.png', 'Filename for the TSNE '
                    'figure. It will be saved in the checkpoint folder.')

# These two flags are necessary for model loading. Don't change them!
flags.DEFINE_string('dataset', 'mnist', 'Don\'t change!')
flags.DEFINE_string('model', 'scae', 'Don\'t change!.')


def _collect_results(sess, tensors, dataset, n_batches):
  """Collects some tensors from many batches."""

  to_collect = AttrDict(
      prior_pres=tensors.caps_presence_prob,
      posterior_pres=tensors.posterior_mixing_probs,
      posterior_acc=tensors.posterior_cls_acc,
      prior_acc=tensors.prior_cls_acc,
      label=dataset['label']
  )

  vals = collect_results(sess, to_collect, n_batches)
  vals.posterior_pres = vals.posterior_pres.sum(1)
  return vals


def main(_=None):
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name,redefined-outer-name
  config = FLAGS
  FLAGS.__dict__['config'] = config

  # Build the graph
  with tf.Graph().as_default():

    model_dict = model_config.get(FLAGS)
    data_dict = data_config.get(FLAGS)

    model = model_dict.model
    trainset = data_dict.trainset
    validset = data_dict.validset

    # Optimisation target
    validset = tools.maybe_convert_dataset(validset)
    trainset = tools.maybe_convert_dataset(trainset)

    train_tensors = model(trainset)
    valid_tensors = model(validset)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.snapshot)

  valid_results = _collect_results(sess, valid_tensors, validset,
                                   10000 // FLAGS.batch_size)

  train_results = _collect_results(sess, train_tensors, trainset,
                                   60000 // FLAGS.batch_size)

  results = AttrDict(train=train_results, valid=valid_results)

  # Linear classification
  print('Linear classification accuracy:')
  for k, v in results.items():
    print('\t{}: prior={:.04f}, posterior={:.04f}'.format(
        k, v.prior_acc.mean(), v.posterior_acc.mean()))

  # Unsupervised classification via clustering
  print('Bipartite matching classification accuracy:')
  for field in 'posterior_pres prior_pres'.split():
    kmeans = sklearn.cluster.KMeans(
        n_clusters=10,
        precompute_distances=True,
        n_jobs=-1,
        max_iter=1000,
    ).fit(results.train[field])

    train_acc = cluster_classify(results.train[field], results.train.label, 10,
                                 kmeans)
    valid_acc = cluster_classify(results.valid[field], results.valid.label, 10,
                                 kmeans)

    print('\t{}: train_acc={:.04f}, valid_acc={:.04f}'.format(field, train_acc,
                                                              valid_acc))

  checkpoint_folder = osp.dirname(FLAGS.snapshot)
  figure_filename = osp.join(checkpoint_folder, FLAGS.tsne_figure_name)
  print('Savign TSNE plot at "{}"'.format(figure_filename))
  make_tsne_plot(valid_results.posterior_pres, valid_results.label,
                 figure_filename)


if __name__ == '__main__':
  try:
    logging.set_verbosity(logging.INFO)
    tf.app.run()
  except Exception as err:  # pylint: disable=broad-except
    FLAGS = flags.FLAGS

    last_traceback = sys.exc_info()[2]
    traceback.print_tb(last_traceback)
    print(err)
    pdb.post_mortem(last_traceback)
