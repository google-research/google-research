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

# pylint: skip-file
from collections import defaultdict
import sys
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from quantile_regression import qr_lib_gasthaus
from quantile_regression import simulation_lib

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "simulation",
    "",
    "Which simulation to run. Current options are SIN, GRIEWANK, ACKLEY, "
    "MICHALEWICZ, UNIFORM, EXPONENTIAL, GAMMA, GAUSSIAN, LAPLACE, BETA.",
)
flags.DEFINE_string(
    "quantile_type",
    "",
    "Method of generating quantiles. "
    'Options are "FIXED_RANDOM", "BATCH_RANDOM", '
    '"DISCRETE_3_RANDOM", "DISCRETE_9_RANDOM", '
    '"DISCRETE_3_CYCLE", "DISCRETE_9_CYCLE", "UNIFORM_BANDWIDTH, BETA". '
    "*CYCLE options cycle between q in [0.1, 0.5, 0.9] or "
    "range(0.1,1.0,step=0.1), respectively. DISCRETE_RANDOM "
    "randomly draws between the options each batch.",
)
flags.DEFINE_integer("train_size", None, "Number of training examples.")
flags.DEFINE_integer("train_steps", None, "Number of minibatches to train for.")
flags.DEFINE_integer(
    "num_repeats",
    1,
    "Number of models to sequentially train and average metrics over.",
)
flags.DEFINE_string("method", "TFL", "TFL, SAMPLE, HARRELL, DNN, GASTHAUS")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_string("optimizer_type", "ADAM", "Choose ADAM or SGD.")
flags.DEFINE_integer(
    "batch_size",
    32,
    "Batch size. If set above train_size, do full-batch gradient updates "
    "instead.",
)
flags.DEFINE_string(
    "optimize_q_keypoints_type",
    None,
    "One of None, 'ALL', 'P99', 'P50'. Whether to use cross-validation to "
    "select the optimal q_keypoints (None or non-None) and what pinball loss "
    "to perform the cross-validation over (ALL, P99, or P50).",
)

# Flags for specific quantile_types
flags.DEFINE_float(
    "bandwidth",
    None,
    "When quantile_type is UNIFORM_BANDWIDTH, q is sampled from "
    "uniform(0.5-bandwidth, 0.5+bandwidth)",
)
flags.DEFINE_float("mode", None, "Mode parameter for BETA quantile_type.")
flags.DEFINE_float("concentration", None, "Concentration parameter for BETA.")


def create_model(simulation, q_keypoints=None):
  """Returns core lattice or DNN model along with stacked model."""
  if FLAGS.method == "TFL":
    base_model = (
        simulation.create_base_model()
        if q_keypoints is None
        else simulation.create_base_model(q_keypoints=q_keypoints)
    )
  elif FLAGS.method == "DNN":
    base_model = simulation.create_dnn_model()

  model_inputs = base_model.inputs
  model_output = base_model.output
  stacked_output = tf.keras.layers.Lambda(lambda inputs: tf.stack(inputs, 1))(
      [model_output, model_inputs[simulation.num_dims()]])
  stacked_model = tf.keras.Model(inputs=model_inputs, outputs=stacked_output)
  return base_model, stacked_model


@tf.function
def q_loss(y_true, stacked_output):
  """Computes example-varying quantile loss for expected pinball training."""
  y_pred, q = tf.split(stacked_output, 2, axis=1)
  y_pred = tf.squeeze(y_pred, axis=[1, 2])
  q = tf.squeeze(q, axis=[1, 2])
  diff = y_true - y_pred
  return tf.reduce_mean(
      tf.maximum(diff, 0.0) * q + tf.minimum(diff, 0.0) * (q - 1.0))


def CRPS_loss(y_true, output):
  """Computes CRPS loss w.r.t. model output, used by Gasthaus model.

  y_true is a tensor of shape [batch size, 1]. output is a tensor of shape
  [batch size, L, L, 1].
  """
  CRPS_tensor = qr_lib_gasthaus.compute_CRPS_tensor(y_true, output)
  return tf.reduce_mean(CRPS_tensor)


def data_generator(
    x_list, y, num_training_examples, batch_size, quantile_type
):
  """Generates batches of training data, including quantile feature."""

  def looped_batch(arr, pointer, new_pointer):
    return np.concatenate([arr[pointer:], arr[:new_pointer]])

  def next_q(q, quantile_type):
    if quantile_type == "DISCRETE_3_CYCLE":
      # Switch between [0.1, 0.5, 0.9]
      return (q + 0.4) % 1.2
    elif quantile_type == "DISCRETE_9_CYCLE":
      # Switch between range(0.1, 1.0, range=0.1)
      return (q % 0.9) + 0.1

  # Generate fixed training set x, q, y
  if quantile_type in ("FIXED_RANDOM", "BATCH_RANDOM"):
    q = np.random.uniform(size=num_training_examples)
  elif quantile_type in ("DISCRETE_3_CYCLE", "DISCRETE_9_CYCLE"):
    q = np.full(x_list[0].shape, 0.1)
  elif quantile_type == "DISCRETE_3_RANDOM":
    q = np.random.choice(np.linspace(0.1, 0.9, 3), num_training_examples)
  elif quantile_type == "DISCRETE_9_RANDOM":
    q = np.random.choice(np.linspace(0.1, 0.9, 9), num_training_examples)
  elif quantile_type == "UNIFORM_BANDWIDTH":
    q = np.random.uniform(0.5-FLAGS.bandwidth, 0.5+FLAGS.bandwidth,
                          size=num_training_examples)
  elif quantile_type == "BETA":
    q = np.random.beta(FLAGS.mode*(FLAGS.concentration-2) + 1,
                       (1-FLAGS.mode)*(FLAGS.concentration-2) + 1,
                       size=num_training_examples)

  # Deliver data in batches, regenerating q if needed
  pointer = 0
  while True:
    if pointer + batch_size <= y.size:
      new_pointer = pointer + batch_size
      yield [x[pointer:new_pointer] for x in x_list] + [q[pointer:new_pointer]
                                                       ], y[pointer:new_pointer]
      pointer = new_pointer
    else:
      new_pointer = batch_size - (y.size - pointer)
      yield [looped_batch(x, pointer, new_pointer) for x in x_list
            ] + [looped_batch(q, pointer, new_pointer)], looped_batch(
                y, pointer, new_pointer)
      pointer = new_pointer
      if quantile_type == "BATCH_RANDOM":
        q = np.random.uniform(size=num_training_examples)
      elif quantile_type in ("DISCRETE_3_CYCLE", "DISCRETE_9_CYCLE"):
        q = next_q(q, quantile_type)
      elif quantile_type == "DISCRETE_3_RANDOM":
        q = np.random.choice(np.linspace(0.1, 0.9, 3), num_training_examples)
      elif quantile_type == "DISCRETE_9_RANDOM":
        q = np.random.choice(np.linspace(0.1, 0.9, 9), num_training_examples)
      elif quantile_type == "UNIFORM_BANDWIDTH":
        q = np.random.uniform(0.5-FLAGS.bandwidth, 0.5+FLAGS.bandwidth,
                              size=num_training_examples)
      elif quantile_type == "BETA":
        q = np.random.beta(FLAGS.mode*(FLAGS.concentration-2) + 1,
                           (1-FLAGS.mode)*(FLAGS.concentration-2) + 1,
                           size=num_training_examples)


def update_metrics(q_mses, metrics, mono_violations, train_time, sel_kp):
  """Appends metrics after each round of model training."""
  metrics["avg99_mse"].append(np.mean(q_mses))
  metrics["avg9_mse"].append(np.mean([q_mses[i] for i in range(9, 99, 10)]))
  metrics["avg5_mse"].append(np.mean([q_mses[0], q_mses[9], q_mses[49], q_mses[89], q_mses[98]]))
  metrics["avg3_mse"].append(np.mean([q_mses[9], q_mses[49], q_mses[89]]))
  metrics["q1_mse"].append(q_mses[0])
  metrics["q10_mse"].append(q_mses[9])
  metrics["q50_mse"].append(q_mses[49])
  metrics["q90_mse"].append(q_mses[89])
  metrics["q99_mse"].append(q_mses[98])
  if train_time is not None:
    metrics["train_time"].append(train_time)
  if mono_violations is not None:
    metrics["mono_violations"].append(mono_violations)
  if sel_kp is not None:
    metrics["sel_kp"].append(sel_kp)
  return metrics


def train_tfl_dnn_model(simulation, x_list, y, batch_size, q_keypoints=None):
  """Creates and trains lattice or DNN model."""
  base_model, stacked_model = create_model(simulation, q_keypoints)
  if FLAGS.optimizer_type == "ADAM":
    stacked_model.compile(loss=q_loss, optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
  elif FLAGS.optimizer_type == "SGD":
    stacked_model.compile(loss=q_loss, optimizer=tf.keras.optimizers.SGD(FLAGS.learning_rate))
  stacked_model.fit_generator(
      data_generator(
          x_list,
          y,
          FLAGS.train_size,
          batch_size,
          FLAGS.quantile_type,
      ),
      steps_per_epoch=FLAGS.train_size // batch_size,
      epochs=FLAGS.train_steps // (FLAGS.train_size // batch_size),
      verbose=0,
  )
  return base_model


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print("simulation: ", FLAGS.simulation, file=sys.stderr)
  print("quantile_type: ", FLAGS.quantile_type, file=sys.stderr)
  print("train_size: ", FLAGS.train_size, file=sys.stderr)
  print("train_steps: ", FLAGS.train_steps, file=sys.stderr)
  print("method: ", FLAGS.method, file=sys.stderr)
  print("num_repeats: ", FLAGS.num_repeats, file=sys.stderr)
  print("learning_rate: ", FLAGS.learning_rate, file=sys.stderr)
  print("optimizer_type: ", FLAGS.optimizer_type, file=sys.stderr)
  print("batch_size: ", FLAGS.batch_size, file=sys.stderr)

  if FLAGS.simulation == "SIN":
    simulation = simulation_lib.SinTestCase()
  elif FLAGS.simulation == "GRIEWANK":
    simulation = simulation_lib.GriewankTestCase()
  elif FLAGS.simulation == "ACKLEY":
    simulation = simulation_lib.AckleyTestCase()
  elif FLAGS.simulation == "MICHALEWICZ":
    simulation = simulation_lib.MichalewiczTestCase()
  elif FLAGS.simulation == "UNIFORM":
    simulation = simulation_lib.UniformTestCase()
  elif FLAGS.simulation == "EXPONENTIAL":
    simulation = simulation_lib.ExponentialTestCase()
  elif FLAGS.simulation == "GAMMA":
    simulation = simulation_lib.GammaTestCase()
  elif FLAGS.simulation == "GAUSSIAN":
    simulation = simulation_lib.GaussianTestCase()
  elif FLAGS.simulation == "LAPLACE":
    simulation = simulation_lib.LaplaceTestCase()
  elif FLAGS.simulation == "BETA":
    simulation = simulation_lib.BetaTestCase()

  metrics = defaultdict(list)
  batch_size = min(FLAGS.batch_size, FLAGS.train_size)
  for i in range(FLAGS.num_repeats):
    print("Starting repeat ", i, file=sys.stderr, flush=True)
    x_list, y = simulation.generate_training_data(FLAGS.train_size)
    sel_kp = None
    train_time = None
    if FLAGS.method == "TFL" or FLAGS.method == "DNN":
      if FLAGS.optimize_q_keypoints_type is not None:
        if FLAGS.train_size == 51:
          kp_range = range(4, 34, 1)
        else:
          kp_range = range(40, 300, 10)
        num_folds = 10
        val_scores = []
        for q_kp in kp_range:
          val_pinballs = np.zeros((num_folds, 99))
          for j in range(num_folds):
            s, f = FLAGS.train_size, num_folds
            y_train = np.concatenate((y[:(j*s//f)], y[((j+1)*s//f):]))
            y_val = y[(j*s//f):((j+1)*s//f)]
            base_model = train_tfl_dnn_model(simulation, x_list, y_train, batch_size, q_kp)
            val_pinballs[j,:] = simulation.compute_validation_pinballs(base_model, y_val)
          if FLAGS.optimize_q_keypoints_type == "P50":
            val_scores.append(np.mean(val_pinballs[:, 49]))
          elif FLAGS.optimize_q_keypoints_type == "P99":
            val_scores.append(np.mean(val_pinballs[:, 98]))
          elif FLAGS.optimize_q_keypoints_type == "ALL":
            val_scores.append(np.mean(val_pinballs))
        lowest_val_score_index = np.argmin(np.array(val_scores))
        best_q_kp = kp_range[lowest_val_score_index]
        sel_kp = best_q_kp
        base_model = train_tfl_dnn_model(simulation, x_list, y, batch_size, best_q_kp)
        train_time = None
      else:
        start_time = time.time()
        base_model = train_tfl_dnn_model(simulation, x_list, y, batch_size)
        end_time = time.time()
        train_time = end_time - start_time
      q_mses = simulation.compute_quantile_mses(base_model)
      if FLAGS.simulation in ("SIN", "GRIEWANK", "MICHALEWICZ1", "ACKLEY"):
        mono_violations = simulation.compute_monotonicity_violations(base_model)
      else:
        mono_violations = None
    elif FLAGS.method == "GASTHAUS":
      start_time = time.time()
      base_model = simulation.create_gasthaus_model()
      if FLAGS.optimizer_type == "ADAM":
        base_model.compile(
            loss=CRPS_loss, optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
      elif FLAGS.optimizer_type == "SGD":
        base_model.compile(
            loss=CRPS_loss, optimizer=tf.keras.optimizers.SGD(FLAGS.learning_rate))
      base_model.fit(
          x_list, y, epochs=FLAGS.train_steps // (FLAGS.train_size // batch_size), batch_size=batch_size, verbose=0)
      end_time = time.time()
      train_time = end_time - start_time
      q_mses = simulation.compute_quantile_mses(base_model, gasthaus=True)
      mono_violations = None
    elif FLAGS.method == "SAMPLE":
      q_mses = simulation.compute_sample_mses(y)
      mono_violations = None
      train_time = None
    elif FLAGS.method == "HARRELL":
      q_mses = simulation.compute_harrell_davis_mses(y)
      mono_violations = None
      train_time = None
    metrics = update_metrics(q_mses, metrics, mono_violations, train_time, sel_kp)
  for k, v in metrics.items():
    print(k, "::", np.mean(np.array(v)))
    if FLAGS.num_repeats > 1 and k != "mono_violations":
      print(k + "_sd ::", np.std(np.array(v)))


if __name__ == "__main__":
  app.run(main)
