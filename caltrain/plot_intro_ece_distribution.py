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

"""Distribution of ECE scores for intro."""
import os

from absl import app
from absl import flags
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from caltrain.run_calibration import calibrate
from caltrain.run_calibration import get_true_dataset

matplotlib.use("Agg")
font = {"size": 40}
matplotlib.rc("font", **font)

FLAGS = flags.FLAGS
flags.DEFINE_string("plot_dir", "./plots", "location to write plots")


def plot_ece_distribution(alpha=1.0, beta=1.0):
  """Plot ece distribution."""
  config = {}
  # For paper quality figure, change num_reps to 1000000
  config["num_reps"] = 100
  config["num_bins"] = 15
  config["split"] = ""
  config["norm"] = 2
  config["calibration_method"] = "no_calibration"
  config["bin_method"] = ""
  config["d"] = 1
  config["alpha"] = alpha
  config["beta"] = beta
  config["a"] = alpha
  config["b"] = beta
  config["dataset"] = "polynomial"
  config["ce_type"] = "ew_ece_bin"
  config["num_samples"] = 200

  true_dataset = get_true_dataset(config)

  eces = []
  for _ in range(config["num_reps"]):
    ece = calibrate(config, true_dataset)
    eces.append(ece)

  fig, ax = plt.subplots(figsize=(10, 10.8))
  linewidth = 3
  hist, bins = np.histogram(eces, bins=20, density=True)
  bin_centers = (bins[1:] + bins[:-1]) * 0.5
  plt.plot(bin_centers, hist, linewidth=linewidth)
  print(np.mean(eces))

  ax.axvline(
      np.mean(eces),
      color="k",
      linestyle="--",
      linewidth=linewidth,
      label="Mean")
  plt.title(r"$\mathrm{ECE}_\mathrm{BIN}$ Distribution")

  # Left panel
  ax.axvline(
      5.08, color="r", linestyle="--", linewidth=linewidth, label="Sample A")

  ax.legend(loc="center right")
  ax.set_ylim([0, 0.22])
  ax.set_xlim([0, 20])
  ax.set_xlabel(r"$\mathrm{ECE}_\mathrm{BIN}$ (%)")
  ax.set_ylabel("PDF")
  save_dir = os.path.join(FLAGS.plot_dir, "intro")
  os.makedirs(save_dir, exist_ok=True)
  fig.savefig(
      os.path.join(
          save_dir,
          "{}.pdf".format("ece_pdf_Beta_alpha={}_beta={}").format(alpha, beta)),
      dpi="figure",
      bbox_inches="tight",
  )


def main(_):
  plot_ece_distribution(alpha=2.8, beta=0.05)


if __name__ == "__main__":
  app.run(main)
