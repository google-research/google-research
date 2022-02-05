"""Utilities for plotting.
"""
import os
import csv
import pdb
import pickle as pkl
from collections import defaultdict

import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')


def reformat_large_tick_values(tick_val, pos):
  """
  Turns large tick values (in the billions, millions and thousands) such as
  4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the
  decimal).
  """
  if tick_val >= 1000000000:
    val = round(tick_val/1000000000, 1)
    new_tick_format = '{:}B'.format(val)
  elif tick_val >= 1000000:
    val = round(tick_val/1000000, 1)
    new_tick_format = '{:}M'.format(val)
  elif tick_val >= 1000:
    val = round(tick_val/1000, 1)
    new_tick_format = '{:}K'.format(val)
  elif tick_val < 1000:
    new_tick_format = round(tick_val, 1)
  else:
    new_tick_format = tick_val

  # make new_tick_format into a string value
  new_tick_format = str(new_tick_format)

  # code below will keep 4.5M as is but change values such as 4.0M to 4M since
  # that zero after the decimal isn't needed
  index_of_decimal = new_tick_format.find(".")

  if index_of_decimal != -1:
    value_after_decimal = new_tick_format[index_of_decimal+1]
    if value_after_decimal == "0":
      # remove the 0 after the decimal point since it's not needed
      new_tick_format = new_tick_format[0:index_of_decimal] + \
                        new_tick_format[index_of_decimal+2:]

  return new_tick_format


def load_log(exp_dir, fname='train_log.csv'):
  result_dict = defaultdict(list)
  with open(os.path.join(exp_dir, fname), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      for key in row:
        try:
          if key in ['global_iteration', 'iteration', 'epoch']:
            result_dict[key].append(int(row[key]))
          else:
            result_dict[key].append(float(row[key]))
        except:
          pass
  return result_dict


def plot_heatmap(pkl_path,
                 xlabel,
                 ylabel,
                 title='',
                 key='train_sum_loss',
                 cmap=plt.cm.gray,
                 levels=10,
                 sigma=1.0,
                 use_smoothing=True,
                 show_contours=False,
                 contour_alpha=0.2,
                 figsize=(10,8)):
  with open(pkl_path, 'rb') as f:
    result = pkl.load(f)

  side_length = int(np.sqrt(len(result['thetas'])))
  grid_data = result[key].reshape(side_length, side_length)
  vmin = np.nanmin(grid_data)
  vmax = np.nanmax(grid_data)

  xv, yv = result['xv'], result['yv']
  # xv = np.log10(np.exp(xv))  # Optional depending on how you want to plot

  if key in ['F_grid_train_loss', 'F_grid_val_loss'] and (vmax > 3):
    vmax = 3
  elif key == 'train_sum_loss' or key == 'unroll_obj':
    vmax = 1e5

  grid_data[np.isnan(grid_data)] = vmax
  grid_data[grid_data > vmax] = vmax

  if use_smoothing:
    F_grid = scipy.ndimage.gaussian_filter(grid_data, sigma=sigma)
  else:
    F_grid = grid_data

  plt.figure(figsize=figsize)

  if 'acc' in key:
    # Good for val_acc
    levels = [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94,
              0.95, 0.96, 0.97, 0.98]

    contour_cmap = plt.cm.get_cmap(cmap, len(levels)+1)
    CS = plt.contourf(yv, xv, F_grid.T, levels, cmap=contour_cmap)
    cbar = plt.colorbar(CS, boundaries=levels)
    cbar.ax.tick_params(labelsize=16)

    if show_contours:
      plt.contour(yv, xv, F_grid.T, levels, colors='white', alpha=contour_alpha)

  else:
    contour_cmap = plt.cm.get_cmap(cmap, levels+1)
    CS = plt.contourf(yv, xv, np.log(F_grid).T, levels, cmap=contour_cmap)
    cbar = plt.colorbar(CS)
    cbar.ax.tick_params(labelsize=16)

    if show_contours:
      plt.contour(
          yv, xv, np.log(F_grid).T, levels, colors='white', alpha=contour_alpha
      )

  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel(xlabel, fontsize=20)
  plt.ylabel(ylabel, fontsize=20)
  plt.title(title, fontsize=24)
