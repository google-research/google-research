"""Script for generating plots like the ones used in the paper."""

import dataclasses
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags
from tensorboard.backend.event_processing import event_accumulator

FLAGS = flags.FLAGS

flags.DEFINE_string("plot_dir", "/tmp/xirl/plots", "Directory wherein to store plots.")
flags.DEFINE_integer("size", 15, "Size multiplier.")
flags.DEFINE_float("std_dev_frac", 0.5, "+/- standard deviation fraction.")
flags.DEFINE_integer("dpi", 100, "Dots per inch for PNG plots.")


def update_plotting_params(size: int) -> None:
  plt.rcParams.update({
    "legend.fontsize": "large",
    "axes.titlesize": size,
    "axes.labelsize": size,
    "xtick.labelsize": size,
    "ytick.labelsize": size,
  })


def minimum_truncate_array_list(arrs: List[np.ndarray]) -> List[np.ndarray]:
  min_len = arrs[0].shape[0]
  for arr in arrs[1:]:
    if arr.shape[0] < min_len:
      min_len = arr.shape[0]
  return [arr[:min_len] for arr in arrs]


@dataclasses.dataclass
class Experiment:
  path: str
  name: str
  color: str
  linestyle: str

  def __post_init__(self) -> None:
    self.path = Path(self.path)
    assert self.path.exists()
    subdirs = [f for f in Path(self.path).iterdir() if f.is_dir()]
    subdirs = sorted(subdirs, key=lambda x: int(x.stem))
    logdirs = [subdir / "tb" for subdir in subdirs]
    logfiles = [list(logdir.iterdir())[0] for logdir in logdirs]
    data = []
    for logfile in logfiles:
      ea = event_accumulator.EventAccumulator(str(logfile))
      ea.Reload()
      df = pd.DataFrame(ea.Scalars("evaluation/average_eval_scores"))
      arr = df[["step", "value"]].to_numpy()
      data.append(arr)
    self.data = minimum_truncate_array_list(data)

  @property
  def mean(self) -> np.ndarray:
    return np.mean(self.data, axis=0)

  @property
  def std_dev(self) -> np.ndarray:
    return np.std(self.data, axis=0)


def cross_shortstick(savename: str) -> None:
  # Note: Append baselines or other methods to this list.
  experiments = [
    Experiment(
      path="/tmp/xirl/rl_runs/env_name=SweepToTop-Shortstick-State-Allo-TestLayout-v0_reward=learned_reward_type=distance_to_goal_mode=cross_algo=xirl_uid=2d6818b7-075e-4dae-895f-e59f4a70ea3d",
      name="XIRL",
      color="tab:red",
      linestyle="dashdot",
    ),
  ]

  _, ax = plt.subplots(1, constrained_layout=True)
  for experiment in experiments:
    return_mean = experiment.mean
    return_stddev = experiment.std_dev

    return_mean_x = return_mean[:, 0] / 1_000
    return_mean_y = return_mean[:, 1]

    ax.plot(
      return_mean_x,
      return_mean_y,
      lw=2,
      label=experiment.name,
      color=experiment.color,
      linestyle=experiment.linestyle,
    )
    ax.fill_between(
      return_mean_x,
      return_mean_y + FLAGS.std_dev_frac * return_stddev[:, 1],
      return_mean_y - FLAGS.std_dev_frac * return_stddev[:, 1],
      alpha=0.2,
      color=experiment.color,
    )

  ax.set_xlabel("Steps (thousands)")
  ax.set_ylabel("Success Rate")
  ax.set_title("short-stick")

  ax.legend(loc="lower right")
  ax.grid(linestyle="--", linewidth=0.5)

  plt.savefig(f"{FLAGS.plot_dir}/{savename}.pdf", format="pdf")
  plt.savefig(f"{FLAGS.plot_dir}/{savename}.png", format="png", dpi=FLAGS.dpi)
  plt.close()


def main(_):
  if not os.path.exists(FLAGS.plot_dir):
    os.makedirs(FLAGS.plot_dir)

  # Update matplotlib settings.
  update_plotting_params(FLAGS.size)

  cross_shortstick("cross_embodiment_shortstick")


if __name__ == "__main__":
  app.run(main)
