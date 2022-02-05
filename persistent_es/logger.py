"""A simple CSV logger.
"""
import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CSVLogger():
  def __init__(self, fieldnames, filename='log.csv'):

    self.fieldnames = fieldnames
    self.filename = filename
    self.csv_file = open(filename, 'w')

    self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
    self.writer.writeheader()

    self.csv_file.flush()

  def writerow(self, row):
    self.writer.writerow(row)
    self.csv_file.flush()

  def close(self):
    self.csv_file.close()


def plot_csv(fpath, xkey='global_iteration'):
  with open(fpath, 'r') as f:
    reader = csv.reader(f)
    dict_of_lists = {}
    ks = None
    for i, r in enumerate(reader):
      if i == 0:
        for k in r:
          dict_of_lists[k] = []
        ks = r
      else:
        for _i, v in enumerate(r):
          dict_of_lists[ks[_i]].append(float(v))

  for k in dict_of_lists:
    if k == xkey:
      continue

    fig = plt.figure()
    plt.plot(dict_of_lists[xkey], dict_of_lists[k], linewidth=2)
    plt.title(k, fontsize=20)
    plt.grid()
    plt.xlabel(xkey, fontsize=18)
    plt.ylabel(k, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if 'loss' in k:
      plt.yscale('log')
    plt.savefig(os.path.join(os.path.dirname(fpath), f'_{k}.png'),
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
