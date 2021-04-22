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

"""Tests for report_utils.py."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
from aqt.utils import report_utils

EventSeries = report_utils.EventSeries


class ReportUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(ReportUtilsTest, self).setUp()
    self.event_series_a = EventSeries(
        name='test_series_a',
        steps=onp.array([0, 2, 4, 6, 8, 10]),
        values=onp.array([1, 2, 3, 4, 5, 6]))
    self.event_series_b = EventSeries(
        name='test_series_b',
        steps=onp.array([0, 2, 4, 6, 8, 10, 12]),
        values=onp.array([1, 2, 3, 0, 0, 3, 6]))
    self.event_series_with_nans = EventSeries(
        name='test_series_with_nans',
        steps=onp.array([0, 2, 4, 6, 8, 10]),
        values=onp.array([1, 2, 3, 4, onp.nan, onp.nan]))
    self.eval_freq = 2

    self.events_dict = {
        'a': self.event_series_a,
        'b': self.event_series_b,
        'with_nans': self.event_series_with_nans
    }

  def assertEventSeriesEqual(self, x, y):
    self.assertEqual(x.name, y.name)
    onp.testing.assert_array_almost_equal(x.steps, y.steps)
    onp.testing.assert_array_almost_equal(x.values, y.values)
    if x.wall_times is None:
      self.assertEqual(x.wall_times, y.wall_times)
    else:
      onp.testing.assert_array_almost_equal(x.wall_times, y.wall_times)

  @parameterized.named_parameters(
      dict(
          testcase_name='wdsz_1_st_0',
          window_size_in_steps=1,
          step=0,
          exp=1,
      ),
      dict(
          testcase_name='wdsz_2_st_1',
          window_size_in_steps=2,
          step=1,
          exp=(1 + 2) / 2,
      ),
      dict(
          testcase_name='wdsz_2_st_2',
          window_size_in_steps=2,
          step=2,
          exp=2,
      ),
      dict(
          testcase_name='wdsz_4_st_0',
          window_size_in_steps=4,
          step=0,
          exp=(1 + 2) / 2,
      ),
      dict(
          testcase_name='wdsz_4_st_2',
          window_size_in_steps=4,
          step=2,
          exp=(1 + 2 + 3) / 3,
      ),
      dict(
          testcase_name='wdsz_12_st_6',
          window_size_in_steps=12,
          step=6,
          exp=(1 + 2 + 3 + 4 + 5 + 6) / 6,
      ),
      dict(
          testcase_name='wdsz_12_st_8',
          window_size_in_steps=12,
          step=8,
          exp=(2 + 3 + 4 + 5 + 6) / 5,
      ),
  )
  def test_apply_smoothing_about_step_with_rectangular_kernel(
      self, window_size_in_steps, step, exp):
    smoothing_kernel = report_utils.SmoothingKernel.RECTANGULAR
    rectangular_kernel_fn = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    res = report_utils.apply_smoothing_about_step(self.event_series_a, step,
                                                  rectangular_kernel_fn)
    self.assertAlmostEqual(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='wdsz_1_st_0',
          window_size_in_steps=1,
          step=0,
          exp=1,
      ),
      dict(
          testcase_name='wdsz_3_st_1',
          window_size_in_steps=3,
          step=1,
          exp=(1 + 2) / 2,
      ),
      dict(
          testcase_name='wdsz_2_st_2',
          window_size_in_steps=2,
          step=2,
          exp=2,
      ),
      dict(
          testcase_name='wdsz_4_st_0',
          window_size_in_steps=4,
          step=0,
          exp=1,
      ),
      dict(
          testcase_name='wdsz_4_st_2',
          window_size_in_steps=4,
          step=2,
          exp=2,
      ),
      dict(
          testcase_name='wdsz_12_st_6',
          window_size_in_steps=12,
          step=6,
          exp=(2 * 2 + 3 * 4 + 4 * 6 + 5 * 4 + 6 * 2) / 18,
      ),
      dict(
          testcase_name='wdsz_12_st_8',
          window_size_in_steps=12,
          step=8,
          exp=(3 * 2 + 4 * 4 + 5 * 6 + 6 * 4) / 16,
      ),
  )
  def test_apply_smoothing_about_step_with_triangular_kernel(
      self, window_size_in_steps, step, exp):
    smoothing_kernel = report_utils.SmoothingKernel.TRIANGULAR
    kernel_fn = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    res = report_utils.apply_smoothing_about_step(self.event_series_a, step,
                                                  kernel_fn)
    self.assertAlmostEqual(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='wdsz_0',
          event_series_key='a',
          window_size_in_steps=0,
          step=0,
      ),
      dict(
          testcase_name='with_nans',
          event_series_key='with_nans',
          window_size_in_steps=5,
          step=0,
      ),
  )
  def test_apply_smoothing_about_step_raise_value_error(self, event_series_key,
                                                        window_size_in_steps,
                                                        step):
    smoothing_kernel = report_utils.SmoothingKernel.RECTANGULAR
    rectangular_kernel_fn = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    with self.assertRaises(ValueError):
      report_utils.apply_smoothing_about_step(
          self.events_dict[event_series_key], step, rectangular_kernel_fn)

  @parameterized.named_parameters(
      dict(
          testcase_name='wdsz_2',
          window_size_in_steps=2,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([1, 2, 3, 4, 5, 6])),
      ),
      dict(
          testcase_name='wdsz_4',
          window_size_in_steps=4,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([(1 + 2) / 2, (1 + 2 + 3) / 3, (2 + 3 + 4) / 3,
                                (3 + 4 + 5) / 3, (4 + 5 + 6) / 3,
                                (5 + 6) / 2])),
      ),
      dict(
          testcase_name='wdsz_6',
          window_size_in_steps=6,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([(1 + 2) / 2, (1 + 2 + 3) / 3, (2 + 3 + 4) / 3,
                                (3 + 4 + 5) / 3, (4 + 5 + 6) / 3,
                                (5 + 6) / 2])),
      ),
      dict(
          testcase_name='wdsz_12',
          window_size_in_steps=12,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([(1 + 2 + 3 + 4) / 4, (1 + 2 + 3 + 4 + 5) / 5,
                                (1 + 2 + 3 + 4 + 5 + 6) / 6,
                                (1 + 2 + 3 + 4 + 5 + 6) / 6,
                                (2 + 3 + 4 + 5 + 6) / 5, (3 + 4 + 5 + 6) / 4])),
      ),
  )
  def test_apply_smoothing_with_rectangular_kernel(self, window_size_in_steps,
                                                   exp):
    smoothing_kernel = report_utils.SmoothingKernel.RECTANGULAR
    rectangular_kernel_fn = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    res = report_utils.apply_smoothing(self.event_series_a,
                                       rectangular_kernel_fn)
    self.assertEventSeriesEqual(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='wdsz_2',
          window_size_in_steps=2,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([1, 2, 3, 4, 5, 6])),
      ),
      dict(
          testcase_name='wdsz_4',
          window_size_in_steps=4,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([1, 2, 3, 4, 5, 6])),
      ),
      dict(
          testcase_name='wdsz_6',
          window_size_in_steps=6,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([(1 * 3 + 2) / 4,
                                (1 + 2 * 3 + 3) / 5,
                                (2 + 3 * 3 + 4) / 5,
                                (3 + 4 * 3 + 5) / 5,
                                (4 + 5 * 3 + 6) / 5,
                                (5 + 6 * 3) / 4])),
      ),
      dict(
          testcase_name='wdsz_12',
          window_size_in_steps=12,
          exp=EventSeries(
              name='test_series_a',
              steps=onp.array([0, 2, 4, 6, 8, 10]),
              values=onp.array([(1 * 6 + 2 * 4 + 3 * 2) / 12,
                                (1 * 4 + 2 * 6 + 3 * 4 + 4 * 2) / 16,
                                (1 * 2 + 2 * 4 + 3 * 6 + 4 * 4 + 5 * 2) / 18,
                                (2 * 2 + 3 * 4 + 4 * 6 + 5 * 4 + 6 * 2) / 18,
                                (3 * 2 + 4 * 4 + 5 * 6 + 6 * 4) / 16,
                                (4 * 2 + 5 * 4 + 6 * 6) / 12])),
      ),
  )
  def test_apply_smoothing_with_triangular_kernel(self, window_size_in_steps,
                                                  exp):
    smoothing_kernel = report_utils.SmoothingKernel.TRIANGULAR
    kernel_fn = smoothing_kernel.get_func(
        window_size_in_steps=window_size_in_steps)
    res = report_utils.apply_smoothing(self.event_series_a,
                                       kernel_fn)
    self.assertEventSeriesEqual(res, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='two_mins',
          event_series_key='b',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=0,
          exp=6),
      dict(
          testcase_name='min_start_step_4',
          event_series_key='b',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=4,
          exp=6),
      dict(
          testcase_name='min_start_step_10',
          event_series_key='b',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=10,
          exp=10),
      dict(
          testcase_name='max_start_step_0',
          event_series_key='b',
          early_stop_agg=report_utils.MinOrMax.MAX,
          start_step=0,
          exp=12),
      dict(
          testcase_name='with_nans_min_start_step_0',
          event_series_key='with_nans',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=0,
          exp=0),
      dict(
          testcase_name='with_nans_min_start_step_2',
          event_series_key='with_nans',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=2,
          exp=2),
      dict(
          testcase_name='with_nans_min_start_step_after_nans',
          event_series_key='with_nans',
          early_stop_agg=report_utils.MinOrMax.MIN,
          start_step=8,
          exp=8),
      dict(
          testcase_name='with_nans_max_start_step_0',
          event_series_key='with_nans',
          early_stop_agg=report_utils.MinOrMax.MAX,
          start_step=0,
          exp=6),
  )
  def test_find_early_stop_step(self, event_series_key, early_stop_agg,
                                start_step, exp):
    early_stop_func = early_stop_agg.get_func()
    res = report_utils.find_early_stop_step(
        self.events_dict[event_series_key],
        early_stop_func=early_stop_func,
        start_step=start_step)
    self.assertEqual(res, exp)

  def test_find_early_stop_step_raises_value_error_on_too_large_start_step(
      self):
    start_step = 20
    with self.assertRaisesRegex(
        ValueError, 'event_series does not have events after start_step.'):
      report_utils.find_early_stop_step(
          self.event_series_b,
          early_stop_func=onp.argmin,
          start_step=start_step)

  @parameterized.named_parameters(
      dict(
          testcase_name='min', early_stop_agg=report_utils.MinOrMax.MIN, exp=0),
      dict(
          testcase_name='max', early_stop_agg=report_utils.MinOrMax.MAX, exp=1),
  )
  def test_get_early_stop_func(self, early_stop_agg, exp):
    early_stop_func = early_stop_agg.get_func()
    res = early_stop_func(onp.array([1, 2]))
    self.assertEqual(res, exp)

  def test_get_early_stop_func_raises_error_on_unknown_agg(self):
    with self.assertRaises(AttributeError):
      early_stop_agg = 'bad input'
      early_stop_agg.get_func()

  def test_get_smoothing_kernel_func(self):
    smoothing_kernel = report_utils.SmoothingKernel.RECTANGULAR
    rect_kernel_func = smoothing_kernel.get_func(window_size_in_steps=3)
    self.assertEqual(rect_kernel_func(1), 1.)
    self.assertEqual(rect_kernel_func(2), 0.)
    self.assertEqual(rect_kernel_func(3), 0.)

  def test_get_smoothing_kernel_func_raises_error_on_unknown_agg(self):
    smoothing_kernel = 'bad input'
    with self.assertRaises(AttributeError):
      smoothing_kernel.get_func(window_size_in_steps=3)

  def test_get_smoothing_kernel_func_raises_error_on_rect_kernel_without_window_size(
      self):
    smoothing_kernel = report_utils.SmoothingKernel.RECTANGULAR
    with self.assertRaises(ValueError):
      smoothing_kernel.get_func(window_size_in_steps=None)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_nans',
          event_series_key='a',
          start_step=0,
          exp=None),
      dict(
          testcase_name='with_nans',
          event_series_key='with_nans',
          start_step=0,
          exp=8),
      dict(
          testcase_name='with_nans_start_step_after_nan',
          event_series_key='with_nans',
          start_step=10,
          exp=10),
  )
  def test_check_for_nans(self, event_series_key, start_step, exp):
    first_nan_step = report_utils.check_for_nans(
        self.events_dict[event_series_key], start_step=start_step)
    self.assertEqual(first_nan_step, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_nans',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 2, 3, 4])),
              }
          },
          exp=None),
      dict(
          testcase_name='with_nans',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 2, onp.nan, onp.nan])),
              }
          },
          exp=6),
      dict(
          testcase_name='with_nan_in_multiple_series',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, onp.nan, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 2, onp.nan, onp.nan])),
              }
          },
          exp=4),
  )
  def test_check_all_events_for_nans(self, all_events, exp):
    first_nan_step = report_utils.check_all_events_for_nans(all_events)
    self.assertEqual(first_nan_step, exp)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_nans_no_smoothing',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 1, 4, 4])),
              }
          },
          early_stop_step=4,
          smoothing=False,
          exp_agg_metrics={
              'train': {
                  'acc': 2,
              },
              'eval': {
                  'loss': 1,
              },
          }),
      dict(
          testcase_name='no_nans_with_smoothing',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 1, 4, 4])),
              }
          },
          early_stop_step=4,
          smoothing=True,
          exp_agg_metrics={
              'train': {
                  'acc': 2,
              },
              'eval': {
                  'loss': 3,
              },
          }),
      dict(
          testcase_name='with_nans_no_smoothing',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 1, onp.nan, onp.nan])),
              }
          },
          early_stop_step=4,
          smoothing=False,
          exp_agg_metrics={
              'train': {
                  'acc': 2,
              },
              'eval': {
                  'loss': 1,
              },
          }),
  )
  def test_get_agg_metrics_at_step(self, all_events, early_stop_step, smoothing,
                                   exp_agg_metrics):
    if smoothing:
      smoothing_kernel_fn = report_utils.SmoothingKernel.RECTANGULAR.get_func(
          window_size_in_steps=5)
    else:
      smoothing_kernel_fn = None

    agg_metrics = report_utils.get_agg_metrics_at_step(
        all_events, early_stop_step, smoothing_kernel_fn)
    self.assertEqual(agg_metrics, exp_agg_metrics)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_nans',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 2, 1, 3])),
              }
          },
          exp_stop_step=6,
          exp_first_nan_step=None,
          exp_agg_metrics={
              'train': {
                  'acc': 3,
              },
              'eval': {
                  'loss': 2,
              },
          },
          exp_agg_metrics_unsmoothed={
              'train': {
                  'acc': 3,
              },
              'eval': {
                  'loss': 1,
              }
          }),
      dict(
          testcase_name='with_nans',
          all_events={
              'train': {
                  'acc':
                      EventSeries(
                          name='acc',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([0, 1, 2, 3, 4])),
              },
              'eval': {
                  'loss':
                      EventSeries(
                          name='loss',
                          steps=onp.array([0, 2, 4, 6, 8]),
                          values=onp.array([2, 4, 1, onp.nan, onp.nan])),
              }
          },
          exp_stop_step=4,
          exp_first_nan_step=6,
          exp_agg_metrics=None,
          exp_agg_metrics_unsmoothed={
              'train': {
                  'acc': 2,
              },
              'eval': {
                  'loss': 1,
              }
          }),
      )
  def test_compute_agg_metrics_from_events(self, all_events, exp_stop_step,
                                           exp_first_nan_step, exp_agg_metrics,
                                           exp_agg_metrics_unsmoothed):

    agg_metrics_unsmoothed, agg_metrics, stop_step, first_nan_step = report_utils.compute_agg_metrics_from_events(
        all_events=all_events,
        early_stop_component='eval',
        early_stop_attr='loss',
        early_stop_agg=report_utils.MinOrMax.MIN,
        smoothing_kernel=report_utils.SmoothingKernel.RECTANGULAR,
        window_size_in_steps=5,
        start_step=2,
    )

    self.assertEqual(stop_step, exp_stop_step)
    self.assertEqual(first_nan_step, exp_first_nan_step)
    self.assertEqual(agg_metrics, exp_agg_metrics)
    self.assertEqual(agg_metrics_unsmoothed, exp_agg_metrics_unsmoothed)


if __name__ == '__main__':
  absltest.main()
