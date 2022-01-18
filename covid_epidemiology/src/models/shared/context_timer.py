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

"""Context timer."""

import logging
import time
from typing import Any
from typing import Sequence

from absl import app
import pandas as pd


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")


if __name__ == "__main__":
  app.run(main)


class Timer:
  """Time blocks of code.

  Example:
    with Timer('Important Code'):
      very_important = True
      with Timer('Outside Loop'):
        for i in range(100):
          with Timer('Inside Loop'):
            time.sleep(0.01)

    profile = Timer.get_profile()
    print(profile)

  Example Output:
                    duration      self  dur_per_call  calls     start       end
    name
    Important Code  1.011003  0.000000      1.011003      1  0.000000  1.011003
    Outside Loop    1.010994  0.000000      1.010994      1  0.000007  1.011001
    Inside Loop     1.009758  1.009758      0.010098    100  0.000010  1.010994

  """
  # Internal data structures for keeping timing values
  _run_timings = []
  _active_profilers = []

  # Specifies how the timing values will be output. See `_log_msg method.
  output_method: str = "print"
  # If true will output the max and min durations in addition to the average
  output_max_min: bool = True

  def __init__(self, name: str = __name__, display_output: bool = True):
    """Time and identify blocks of code using this name.

    Args:
      name: The name of the timer. Used for identification and grouping.
      display_output: If true the specific timings will be output.
    """
    self.name = name
    self.display_output = display_output

  def __enter__(self):
    self._active_profilers.append(self)
    self.depth = len(self._active_profilers)
    self.start_time = time.perf_counter()
    self._log_msg_filtered(f"{self.start_name}")

  def __exit__(self, *args):
    end_time = time.perf_counter()
    self._active_profilers.remove(self)
    self._run_timings.append({
        "name": self.start_name,
        "start": self.start_time,
        "end": end_time
    })
    if self.display_output:
      duration = end_time - self.start_time
      self._log_msg_filtered(f"{duration:.03f}s {self.end_name}")

  def _log_msg_filtered(self, msg: str) -> None:
    """Log a message depending on the output method.

    Args:
      msg: The message to be logged
    """
    if not self.display_output:
      return

    self._log_msg(msg)

  @property
  def start_name(self) -> str:
    return f"{'>' * self.depth} {self.name}"

  @property
  def end_name(self) -> str:
    return f"{self.name} {'<' * self.depth}"

  @classmethod
  def _log_msg(cls, msg: str) -> None:
    if cls.output_method.lower().startswith("log"):
      logging.debug(msg)
    elif cls.output_method.lower() in ["tf", "tf.print", "tf_print"]:
      # So we don't have to import tensorflow if we aren't using it.
      import tensorflow as tf  # pylint: disable=g-import-not-at-top
      tf.print(msg)
    else:
      print(msg)

  @classmethod
  def reset_timings(cls) -> None:
    """Clear the current timings."""
    cls._run_timings = []

  @classmethod
  def get_timings(cls, calculate_self_time: bool = True) -> pd.DataFrame:
    """Get all the raw timing information."""
    timing_df = pd.DataFrame(cls._run_timings)
    first_start = timing_df["start"].min()
    timing_df["start"] -= first_start
    timing_df["end"] -= first_start
    timing_df["duration"] = timing_df["end"] - timing_df["start"]
    timing_df = timing_df.sort_values(["start", "end"],
                                      ascending=True,
                                      ignore_index=True)
    if calculate_self_time:
      timing_df["self"] = pd.NA
      for idx, row in timing_df.iterrows():
        _calculate_self_time(idx, row, timing_df)
    return timing_df

  @classmethod
  def get_profile(cls, calculate_self_time: bool = True) -> pd.DataFrame:
    """Get the profiling results."""
    timings = cls.get_timings(calculate_self_time=calculate_self_time)
    name_gb = timings.groupby(["name"])
    if cls.output_max_min:
      profile = name_gb.agg({
          "start": ["min", "count"],
          "end": "max",
          "duration": ["min", "mean", "max", "sum"],
          "self": "sum"
      })
      output = profile[[
          ("self", "sum"),
          ("duration", "min"),
          ("duration", "mean"),
          ("duration", "max"),
          ("duration", "sum"),
          ("start", "count"),
          ("start", "min"),
          ("end", "max"),
      ]].copy()
      output.columns = [
          "self", "min", "mean", "max", "total", "calls", "start", "end"
      ]
    else:
      profile = name_gb.agg({
          "start": "min",
          "end": "max",
          "duration": "sum",
          "self": "sum"
      })
      # Count amount of times each timer was run.
      profile["calls"] = name_gb[("start", "count")].count()
      profile["mean"] = profile["duration"] / profile["calls"]
      profile = profile.rename(columns={"duration": "total"})
      cols = ["self", "mean", "total", "calls", "start", "end"]
      output = profile[cols]
    return output.sort_values("start")

  @classmethod
  def print_profile(cls, calculate_self_time: bool = True) -> None:
    """Logs the full profile via the method selected.

    Args:
      calculate_self_time: If you want to calculate self time
    """

    profile = cls.get_profile(calculate_self_time)
    print("ContextTimer Profile:")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        -1,
    ):
      cls._log_msg(str(profile))


def _calculate_self_time(current_index: Any, current_row: pd.Series,
                         input_df: pd.DataFrame):
  """Calculate the amount of time spent in a block excluding sub-blocks."""
  # Already calculated the time for this row
  if pd.notnull(current_row["self"]):
    return

  # Get overlapping timers so I can parse out my own time
  overlapping_timers = input_df[(input_df["start"] >= current_row["start"])
                                & (input_df["end"] <= current_row["end"])].drop(
                                    current_index)

  if overlapping_timers.empty:
    input_df.loc[current_index, "self"] = current_row["duration"]
  else:
    for overlapping_index, overlapping_timer in overlapping_timers[
        overlapping_timers["self"].isnull()].iterrows():
      _calculate_self_time(overlapping_index, overlapping_timer,
                           overlapping_timers)

    overlapping_timers.loc[
        current_index,
        "self"] = current_row["duration"] - overlapping_timers["self"].sum()
