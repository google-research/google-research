# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for tracing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import six
import tensorflow as tf

import simple_probabilistic_programming as ed

tfe = tf.contrib.eager


@tfe.run_all_tests_in_graph_and_eager_modes
class TraceTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2., "kwargs": {"loc": 0.5, "scale": 1.}},
      {"cls": ed.Bernoulli, "value": 1, "kwargs": {"logits": 0.}},
  )
  def testTrace(self, cls, value, kwargs):
    def tracer(f, *fargs, **fkwargs):
      name = fkwargs.get("name", None)
      if name == "rv2":
        fkwargs["value"] = value
      return f(*fargs, **fkwargs)
    rv1 = cls(value=value, name="rv1", **kwargs)
    with ed.trace(tracer):
      rv2 = cls(name="rv2", **kwargs)
    rv1_value, rv2_value = self.evaluate([rv1.value, rv2.value])
    self.assertEqual(rv1_value, value)
    self.assertEqual(rv2_value, value)

  def testTrivialTracerPreservesLogJoint(self):
    def trivial_tracer(fn, *args, **kwargs):
      # A tracer that does nothing.
      return ed.traceable(fn)(*args, **kwargs)

    def model():
      return ed.Normal(0., 1., name="x")

    def transformed_model():
      with ed.trace(trivial_tracer):
        model()

    log_joint = ed.make_log_joint_fn(model)
    log_joint_transformed = ed.make_log_joint_fn(transformed_model)
    self.assertEqual(self.evaluate(log_joint(x=5.)),
                     self.evaluate(log_joint_transformed(x=5.)))

  def testTraceForwarding(self):
    def double(f, *args, **kwargs):
      return 2. * ed.traceable(f)(*args, **kwargs)

    def set_xy(f, *args, **kwargs):
      if kwargs.get("name") == "x":
        kwargs["value"] = 1.
      if kwargs.get("name") == "y":
        kwargs["value"] = 0.42
      return ed.traceable(f)(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.trace(set_xy):
      with ed.trace(double):
        z = model()

    value = 2. * 1. + 2. * 0.42
    z_value = self.evaluate(z)
    self.assertAlmostEqual(z_value, value, places=5)

  def testTraceNonForwarding(self):
    def double(f, *args, **kwargs):
      self.assertEqual("yes", "no")
      return 2. * f(*args, **kwargs)

    def set_xy(f, *args, **kwargs):
      if kwargs.get("name") == "x":
        kwargs["value"] = 1.
      if kwargs.get("name") == "y":
        kwargs["value"] = 0.42
      return f(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.trace(double):
      with ed.trace(set_xy):
        z = model()

    value = 1. + 0.42
    z_value = self.evaluate(z)
    self.assertAlmostEqual(z_value, value, places=5)

  def testTraceException(self):
    def f():
      raise NotImplementedError()
    def tracer(f, *fargs, **fkwargs):
      return f(*fargs, **fkwargs)

    with ed.get_next_tracer() as top_tracer:
      old_tracer = top_tracer

    with self.assertRaises(NotImplementedError):
      with ed.trace(tracer):
        f()

    with ed.get_next_tracer() as top_tracer:
      new_tracer = top_tracer

    self.assertEqual(old_tracer, new_tracer)

  def testTape(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.tape() as model_tape:
      output = model()

    expected_value, actual_value = self.evaluate([
        model_tape["x"] + model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)

  def testTapeNoName(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1.)
      return x + y

    with ed.tape() as model_tape:
      _ = model()

    self.assertEqual(list(six.iterkeys(model_tape)), ["x"])

  def testTapeOuterForwarding(self):
    def double(f, *args, **kwargs):
      return 2. * ed.traceable(f)(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.tape() as model_tape:
      with ed.trace(double):
        output = model()

    expected_value, actual_value = self.evaluate([
        2. * model_tape["x"] + 2. * model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)

  def testTapeInnerForwarding(self):
    def double(f, *args, **kwargs):
      return 2. * ed.traceable(f)(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.trace(double):
      with ed.tape() as model_tape:
        output = model()

    expected_value, actual_value = self.evaluate([
        model_tape["x"] + model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
  tf.test.main()
