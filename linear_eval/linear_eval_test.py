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

"""Tests for linear_eval."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as onp
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection
from linear_eval import linear_eval


# Copied from the FLAX tutorial.
def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[Ellipsis, None] == jnp.arange(num_classes)[None])
  x = jax.lax.select(
      x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels, num_classes):
  return -jnp.sum(onehot(labels, num_classes=num_classes) *
                  jax.nn.log_softmax(logits))


class LinearEvalTest(absltest.TestCase):

  def testTrainAndEvaluate(self):
    iris = datasets.load_iris()

    (train_embeddings, test_embeddings,
     train_labels, test_labels) = model_selection.train_test_split(
         iris.data[:, :2].astype(onp.float32), iris.target, test_size=0.25,
         random_state=0xdeadbeef)

    sklearn_logreg = linear_model.LogisticRegression(
        C=1e5, solver='lbfgs', multi_class='multinomial')
    sklearn_logreg.fit(train_embeddings, train_labels)
    sklearn_y_pred = sklearn_logreg.predict(test_embeddings)

    ((train_embeddings, train_labels),
     train_mask) = linear_eval.reshape_and_pad_data_for_devices(
         (train_embeddings, train_labels))
    ((test_embeddings, test_labels),
     test_mask) = linear_eval.reshape_and_pad_data_for_devices(
         (test_embeddings, test_labels))
    weights, biases, _ = linear_eval.train(
        train_embeddings, train_labels, train_mask, l2_regularization=1e-6)
    accuracy = linear_eval.evaluate(
        test_embeddings, test_labels, test_mask, weights, biases).astype(
            onp.float32)

    self.assertAlmostEqual(accuracy, onp.mean(sklearn_y_pred == test_labels),
                           places=3)

  def testLossFunction(self):
    rng = onp.random.RandomState(1337)
    weights = rng.randn(10, 20)
    biases = rng.randn(20) / 4
    embeddings = rng.randn(100, 10)
    logits = embeddings.dot(weights) + biases[None, :]
    labels = onp.argmax(logits + onp.random.randn(*logits.shape), -1)
    mask = onp.ones((embeddings.shape[0],))

    params = linear_eval.weights_and_biases_to_params(weights, biases)
    loss = linear_eval.multinomial_logistic_loss(
        params, embeddings, labels, mask, 1, 0.0).astype(onp.float32)
    self.assertAlmostEqual(
        loss,
        cross_entropy_loss(logits, labels, weights.shape[-1]),
        places=3)


if __name__ == '__main__':
  absltest.main()
