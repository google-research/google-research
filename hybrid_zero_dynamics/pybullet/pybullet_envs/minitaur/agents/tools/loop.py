# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute operations in a loop and coordinate logging and checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

from pybullet_envs.minitaur.agents.tools import streaming_mean


_Phase = collections.namedtuple(
    'Phase',
    'name, writer, op, batch, steps, feed, report_every, log_every,'
    'checkpoint_every')


class Loop(object):
  """Execute operations in a loop and coordinate logging and checkpoints.

  Supports multiple phases, that define their own operations to run, and
  intervals for reporting scores, logging summaries, and storing checkpoints.
  All class state is stored in-graph to properly recover from checkpoints.
  """

  def __init__(self, logdir, step=None, log=None, report=None, reset=None):
    """Execute operations in a loop and coordinate logging and checkpoints.

    The step, log, report, and report arguments will get created if not
    provided. Reset is used to indicate switching to a new phase, so that the
    model can start a new computation in case its computation is split over
    multiple training steps.

    Args:
      logdir: Will contain checkpoints and summaries for each phase.
      step: Variable of the global step (optional).
      log: Tensor indicating to the model to compute summary tensors.
      report: Tensor indicating to the loop to report the current mean score.
      reset: Tensor indicating to the model to start a new computation.
    """
    self._logdir = logdir
    self._step = (
        tf.Variable(0, False, name='global_step') if step is None else step)
    self._log = tf.placeholder(tf.bool) if log is None else log
    self._report = tf.placeholder(tf.bool) if report is None else report
    self._reset = tf.placeholder(tf.bool) if reset is None else reset
    self._phases = []

  def add_phase(
      self, name, done, score, summary, steps,
      report_every=None, log_every=None, checkpoint_every=None, feed=None):
    """Add a phase to the loop protocol.

    If the model breaks long computation into multiple steps, the done tensor
    indicates whether the current score should be added to the mean counter.
    For example, in reinforcement learning we only have a valid score at the
    end of the episode.

    Score and done tensors can either be scalars or vectors, to support
    single and batched computations.

    Args:
      name: Name for the phase, used for the summary writer.
      done: Tensor indicating whether current score can be used.
      score: Tensor holding the current, possibly intermediate, score.
      summary: Tensor holding summary string to write if not an empty string.
      steps: Duration of the phase in steps.
      report_every: Yield mean score every this number of steps.
      log_every: Request summaries via `log` tensor every this number of steps.
      checkpoint_every: Write checkpoint every this number of steps.
      feed: Additional feed dictionary for the session run call.

    Raises:
      ValueError: Unknown rank for done or score tensors.
    """
    done = tf.convert_to_tensor(done, tf.bool)
    score = tf.convert_to_tensor(score, tf.float32)
    summary = tf.convert_to_tensor(summary, tf.string)
    feed = feed or {}
    if done.shape.ndims is None or score.shape.ndims is None:
      raise ValueError("Rank of 'done' and 'score' tensors must be known.")
    writer = self._logdir and tf.summary.FileWriter(
        os.path.join(self._logdir, name), tf.get_default_graph(),
        flush_secs=60)
    op = self._define_step(done, score, summary)
    batch = 1 if score.shape.ndims == 0 else score.shape[0].value
    self._phases.append(_Phase(
        name, writer, op, batch, int(steps), feed, report_every,
        log_every, checkpoint_every))

  def run(self, sess, saver, max_step=None):
    """Run the loop schedule for a specified number of steps.

    Call the operation of the current phase until the global step reaches the
    specified maximum step. Phases are repeated over and over in the order they
    were added.

    Args:
      sess: Session to use to run the phase operation.
      saver: Saver used for checkpointing.
      max_step: Run the operations until the step reaches this limit.

    Yields:
      Reported mean scores.
    """
    global_step = sess.run(self._step)
    steps_made = 1
    while True:
      if max_step and global_step >= max_step:
        break
      phase, epoch, steps_in = self._find_current_phase(global_step)
      phase_step = epoch * phase.steps + steps_in
      if steps_in % phase.steps < steps_made:
        message = '\n' + ('-' * 50) + '\n'
        message += 'Phase {} (phase step {}, global step {}).'
        tf.logging.info(message.format(phase.name, phase_step, global_step))
      # Populate book keeping tensors.
      phase.feed[self._reset] = (steps_in < steps_made)
      phase.feed[self._log] = (
          phase.writer and
          self._is_every_steps(phase_step, phase.batch, phase.log_every))
      phase.feed[self._report] = (
          self._is_every_steps(phase_step, phase.batch, phase.report_every))
      summary, mean_score, global_step, steps_made = sess.run(
          phase.op, phase.feed)
      if self._is_every_steps(phase_step, phase.batch, phase.checkpoint_every):
        self._store_checkpoint(sess, saver, global_step)
      if self._is_every_steps(phase_step, phase.batch, phase.report_every):
        yield mean_score
      if summary and phase.writer:
        # We want smaller phases to catch up at the beginnig of each epoch so
        # that their graphs are aligned.
        longest_phase = max(phase.steps for phase in self._phases)
        summary_step = epoch * longest_phase + steps_in
        phase.writer.add_summary(summary, summary_step)

  def _is_every_steps(self, phase_step, batch, every):
    """Determine whether a periodic event should happen at this step.

    Args:
      phase_step: The incrementing step.
      batch: The number of steps progressed at once.
      every: The interval of the periode.

    Returns:
      Boolean of whether the event should happen.
    """
    if not every:
      return False
    covered_steps = range(phase_step, phase_step + batch)
    return any((step + 1) % every == 0 for step in covered_steps)

  def _find_current_phase(self, global_step):
    """Determine the current phase based on the global step.

    This ensures continuing the correct phase after restoring checkoints.

    Args:
      global_step: The global number of steps performed across all phases.

    Returns:
      Tuple of phase object, epoch number, and phase steps within the epoch.
    """
    epoch_size = sum(phase.steps for phase in self._phases)
    epoch = int(global_step // epoch_size)
    steps_in = global_step % epoch_size
    for phase in self._phases:
      if steps_in < phase.steps:
        return phase, epoch, steps_in
      steps_in -= phase.steps

  def _define_step(self, done, score, summary):
    """Combine operations of a phase.

    Keeps track of the mean score and when to report it.

    Args:
      done: Tensor indicating whether current score can be used.
      score: Tensor holding the current, possibly intermediate, score.
      summary: Tensor holding summary string to write if not an empty string.

    Returns:
      Tuple of summary tensor, mean score, and new global step. The mean score
      is zero for non reporting steps.
    """
    if done.shape.ndims == 0:
      done = done[None]
    if score.shape.ndims == 0:
      score = score[None]
    score_mean = streaming_mean.StreamingMean((), tf.float32)
    with tf.control_dependencies([done, score, summary]):
      done_score = tf.gather(score, tf.where(done)[:, 0])
      submit_score = tf.cond(
          tf.reduce_any(done), lambda: score_mean.submit(done_score), tf.no_op)
    with tf.control_dependencies([submit_score]):
      mean_score = tf.cond(self._report, score_mean.clear, float)
      steps_made = tf.shape(score)[0]
      next_step = self._step.assign_add(steps_made)
    with tf.control_dependencies([mean_score, next_step]):
      return tf.identity(summary), mean_score, next_step, steps_made

  def _store_checkpoint(self, sess, saver, global_step):
    """Store a checkpoint if a log directory was provided to the constructor.

    The directory will be created if needed.

    Args:
      sess: Session containing variables to store.
      saver: Saver used for checkpointing.
      global_step: Step number of the checkpoint name.
    """
    if not self._logdir or not saver:
      return
    tf.gfile.MakeDirs(self._logdir)
    filename = os.path.join(self._logdir, 'model.ckpt')
    saver.save(sess, filename, global_step)
