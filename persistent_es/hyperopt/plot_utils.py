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

"""Hyperopt plotting utilities.
"""
import os
import csv
import ipdb
import json
import pandas as pd
import pickle as pkl
from collections import defaultdict

import numpy as np
import scipy.ndimage

from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator

import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
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

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

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


def print_best_thetas(log,
                      key='train_loss',
                      print_keys=['train_sum_loss', 'train_loss', 'val_loss', 'train_acc', 'val_acc'],
                      top_k=1,
                      reverse=False,
                      constrained=True):
    sorted_idxs = np.argsort(log[key])

    if reverse:
        sorted_idxs = list(reversed(sorted_idxs))

    if constrained:
        theta_key = 'thetas_constrained'
    else:
        theta_key = 'thetas'

    for k, idx in enumerate(sorted_idxs[:top_k]):
        print('{} Top {}: {}'.format(key, k, log[key][idx]))
        for print_key in print_keys:
            print('{}: {:5.3f}'.format(print_key, log[print_key][idx]))
        print(' ')

    print('\n')


def print_best_thetas_for_keys(log,
                               keys=['train_sum_loss', 'train_loss', 'val_loss', 'train_acc', 'val_acc'],
                               top_k=1,
                               constrained=True):
    for key in keys:
        print_best_thetas(log, key=key, print_keys=keys,
                          top_k=top_k, reverse=True if 'acc' in key else False, constrained=constrained)


def plot_hist(directory, key, ax=None, bins=60, min_value=None, max_value=None, yscale='linear'):
    rs_values = get_all_random_search_values(directory, key=key)
    if not max_value:
        max_value = np.nanmax(rs_values)
    if not min_value:
        min_value = np.nanmin(rs_values)

    if ax:
        plt.sca(ax)
    else:
        plt.figure()

    plt.hist([value for value in np.sort(rs_values) if (value >= min_value) and (value <= max_value)], bins=bins)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale(yscale)
    plt.xlabel('Objective Value', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title(key, fontsize=20)


def get_best_theta(log, key='train_losses', top_k=1, reverse=False, constrained=True):
    sorted_idxs = np.argsort(log[key])

    if reverse:
        sorted_idxs = list(reversed(sorted_idxs))

    if constrained:
        theta_key = 'thetas_constrained'
    else:
        theta_key = 'thetas'

    # for idx in sorted_idxs[:top_k]:
    #     for name, value in zip(log['hparam_fieldnames'], log[theta_key][idx]):
    #         print('\t{}: {:6.3e}'.format(name, value))

    best_theta = log[theta_key][sorted_idxs[0]]
    value_at_best_theta = log[key][sorted_idxs[0]]
    # print('Best theta: {} | Value at theta: {}'.format(best_theta, value_at_best_theta))
    return best_theta, value_at_best_theta, sorted_idxs


def make_dataframe(result):
    subset_result = {}
    subset_result['thetas'] = [', '.join(['{:5.2f}'.format(item) for item in theta]) for theta in result['thetas']]
    subset_result['train_accs'] = result['train_accs']
    subset_result['val_accs'] = result['val_accs']
    subset_result['train_losses'] = result['train_losses']
    subset_result['val_losses'] = result['val_losses']
    subset_result['trajectory_sums'] = result['trajectory_sums']
    # subset_result['train_cost'] = result['train_cost']
    df = pd.DataFrame.from_dict(subset_result, orient='index').transpose()
    return df


def make_theta_dicts(result):
    best_theta_F_sum, value_at_best_theta_F_sum = get_best_theta(result, key='trajectory_sums', reverse=False)
    best_theta_train_loss, value_at_best_theta_train_loss = get_best_theta(result, key='train_losses', reverse=False)
    best_theta_val_loss, value_at_best_theta_val_loss = get_best_theta(result, key='val_losses', reverse=False)
    best_theta_train_acc, value_at_best_theta_train_acc = get_best_theta(result, key='train_accs', reverse=True)
    best_theta_val_acc, value_at_best_theta_val_acc = get_best_theta(result, key='val_accs', reverse=True)

    best_theta_dict = {'F_sum': best_theta_F_sum,
                       'full_train_loss': best_theta_train_loss,
                       'full_val_loss': best_theta_val_loss,
                       'full_train_acc': best_theta_train_acc,
                       'full_val_acc': best_theta_val_acc,
                      }

    value_at_best_theta_dict = {'F_sum': value_at_best_theta_F_sum,
                                'full_train_loss': value_at_best_theta_train_loss,
                                'full_val_loss': value_at_best_theta_val_loss,
                                'full_train_acc': value_at_best_theta_train_acc,
                                'full_val_acc': value_at_best_theta_val_acc,
                               }

    return best_theta_dict, value_at_best_theta_dict


def plot_heatmap(pkl_path, xlabel, ylabel, title='', key='train_sum_loss', cmap=plt.cm.gray, levels=10, figsize=(10,8)):
    with open(pkl_path, 'rb') as f:
        result = pkl.load(f)

    side_length = int(np.sqrt(len(result['thetas'])))

    # grid_data = result[key]
    grid_data = result[key].reshape(side_length, side_length)
    vmin = np.nanmin(grid_data)
    vmax = np.nanmax(grid_data)

    xv, yv = result['xv'], result['yv']

    if key in ['F_grid_train_loss', 'F_grid_val_loss'] and (vmax > 3):
        vmax = 3
    elif key == 'train_sum_loss':
        vmax = 1e4

    grid_data[np.isnan(grid_data)] = vmax
    grid_data[grid_data > vmax] = vmax

    smoothed_F_grid = scipy.ndimage.gaussian_filter(grid_data, sigma=1.0)

    plt.figure(figsize=figsize)
    if key in ['F_grid_train_loss', 'F_grid_val_loss', 'train_sum_loss']:
        smoothed_F_grid = np.log(smoothed_F_grid)

    if 'acc' in key:
        # levels = [0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
        levels = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        contour_cmap = plt.cm.get_cmap(cmap, len(levels)+1)
        CS = plt.contourf(yv, xv, smoothed_F_grid.T, levels=levels, cmap=contour_cmap)
        print(CS.levels)
        cbar = plt.colorbar(CS, boundaries=levels)
        cbar.ax.tick_params(labelsize=16)
    else:
        contour_cmap = plt.cm.get_cmap(cmap, levels+1)
        CS = plt.contourf(yv, xv, smoothed_F_grid.T, levels=levels, cmap=contour_cmap)
        cbar = plt.colorbar(CS)
        cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=24)


def plot_hparams(log,
                 keys,
                 inner_problem_len=200,
                 xkey='inner_problem_steps',
                 plot_inner_problem_ticks=False,
                 xlim=None,
                 xtick_locs=None,
                 xtick_labels=None,
                 xscale='linear',
                 yscale='linear',
                 xlabel='Inner Iteration',
                 ylabel='Hyperparameter Value',
                 show_legend=True,
                 legend_outside=True):
    plt.figure(figsize=(8,5))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, key in enumerate(keys):
        # plt.plot(log['result_dict'][xkey], np.log10(np.exp(np.array(log['result_dict'][key]))),
        #          label='{}'.format(key), color=colors[i % len(colors)], marker='o', linewidth=2, linestyle='-')
        plt.plot(log[xkey], np.log10(np.exp(np.array(log[key]))),
                 label='{}'.format(key), color=colors[i % len(colors)], linewidth=2, linestyle='-')

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    if xlim:
        plt.xlim(xlim)
    plt.xscale(xscale)
    plt.yscale(yscale)

    if plot_inner_problem_ticks:
        ml = MultipleLocator(inner_problem_len)
        plt.gca().xaxis.set_minor_locator(ml)
        plt.gca().xaxis.set_major_locator(ml)
        plt.gca().xaxis.grid(which='both', color='k', linestyle='-.', linewidth=0.5, alpha=0.5)

    if xtick_locs:
        plt.xticks(xtick_locs, xtick_labels, fontsize=18)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if show_legend:
        if legend_outside:
            plt.legend(fontsize=16, fancybox=True, framealpha=0.3, bbox_to_anchor=(1.04,1), loc='upper left')
        else:
            plt.legend(fontsize=16, fancybox=True, framealpha=0.3)

    plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    sns.despine()


def plot_es_pes_hparams(es_log, pes_log, keys, inner_problem_len=200, xlim=None,
                        xtick_locs=None, xtick_labels=None, yscale='linear',
                        xlabel='Inner Iteration', ylabel='Hyperparameter Value',
                        legend_outside=False):
    plt.figure(figsize=(8,5))
    # plt.axhline(y=best_theta_dict['F_sum'], color='k')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, key in enumerate(keys):
        plt.plot(es_log['inner_problem_steps'], es_log[key],
                 label='ES {}'.format(key), color=colors[i % len(colors)], linewidth=2, linestyle='-', alpha=0.4)

    for i, key in enumerate(keys):
        plt.plot(pes_log['inner_problem_steps'], pes_log[key],
                 label='PES {}'.format(key), color=colors[i % len(colors)], linewidth=2, alpha=1)

    ml = MultipleLocator(inner_problem_len)
    plt.gca().xaxis.set_minor_locator(ml)
    plt.gca().xaxis.set_major_locator(ml)
    plt.gca().xaxis.grid(which='both', color='k', linestyle='-.', linewidth=0.5, alpha=0.5)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    if xtick_locs:
        plt.xticks(xtick_locs, xtick_labels, fontsize=18)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if xlim:
        plt.xlim(xlim)
    plt.yscale(yscale)
    if legend_outside:
        plt.legend(fontsize=16, fancybox=True, framealpha=0.3, bbox_to_anchor=(1.04,1), loc='upper left')
    else:
        plt.legend(fontsize=16, fancybox=True, framealpha=0.3)

    plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    sns.despine()


def plot_performance(log, show_key,
                     xkey='inner_problem_steps',
                     inner_problem_len=200,
                     plot_inner_problem_ticks=False,
                     xlim=None,
                     xtick_locs=None, xtick_labels=None,
                     xscale='linear', yscale='linear'):
    plt.figure(figsize=(8,5))

    plt.plot(log[xkey], log[show_key], linewidth=2, marker='o')

    plt.xlabel('Inner Iteration', fontsize=20)
    plt.ylabel(y_label_dict[show_key], fontsize=20)

    if plot_inner_problem_ticks:
        ml = MultipleLocator(inner_problem_len)
        plt.gca().xaxis.set_minor_locator(ml)
        plt.gca().xaxis.set_major_locator(ml)
        plt.gca().xaxis.grid(which='both', color='k', linestyle='-.', linewidth=0.5, alpha=0.5)

    plt.xticks(xtick_locs, xtick_labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xlim)
#     plt.xscale(xscale)
    plt.yscale(yscale)

    plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    sns.despine()


def plot_es_pes_performance(es_log, pes_log, show_key, inner_problem_len=200, xlim=None,
                            xtick_locs=None, xtick_labels=None, yscale='linear'):
    plt.figure(figsize=(10,5))
    plt.figure(figsize=(8,5))

    # Plot value at the best theta found by the random search
    # plt.axhline(y=value_at_best_theta_dict[show_key], color='k', linestyle='-')

    plt.plot(es_log['inner_problem_steps'], es_log[show_key], label='ES', linewidth=2)
    plt.plot(pes_log['inner_problem_steps'], pes_log[show_key], label='PES', linewidth=2)

    ml = MultipleLocator(inner_problem_len)
    plt.gca().xaxis.set_minor_locator(ml)
    plt.gca().xaxis.set_major_locator(ml)
    plt.gca().xaxis.grid(which='both', color='k', linestyle='-.', linewidth=0.5, alpha=0.5)

    plt.xlabel('Inner Iteration', fontsize=20)
    plt.ylabel(y_label_dict[show_key], fontsize=20)
    plt.xticks(xtick_locs, xtick_labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xlim)
    plt.yscale(yscale)
    plt.legend(fontsize=16, fancybox=True, framealpha=0.3)
    plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    sns.despine()


def plot_piecewise_schedule_knots_vals(knots, values, T=2500, color=None, alpha=1.0, create_figure=True):
    ts = jax.nn.softmax(jnp.array(knots))
    ts = jnp.cumsum(ts)
    ts = jnp.concatenate([jnp.array([0.0]), ts])  # Explicitly add the 0 mark --> [0, 0.25, 0.5, 0.75, 1]
    ts = ts * T

    if create_figure:
        plt.figure()

    if color:
        plt.plot(ts, values, marker='o', linewidth=2, color=color, alpha=alpha)
    else:
        plt.plot(ts, values, marker='o', linewidth=2, alpha=alpha)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Hparam Schedule', fontsize=20)


def plot_piecewise_schedule(log, T=2500):
    # log = log['result_dict']

    knots = [log[key][-1] for key in log.keys() if 'knot' in key and 'grad' not in key]
    values = [log[key][-1] for key in log.keys() if 'value' in key and 'grad' not in key]

    plot_piecewise_schedule_knots_vals(knots, values, T)


def plot_individual_layer_grid(log_dir, xkey='inner_problem_steps', key_identifier='lr'):
    log = load_log(log_dir, fname='frequent.csv')
    keys = [key for key in log.keys() if key_identifier in key and 'grad' not in key]

    param_name_hparam_dict = defaultdict(list)
    xvalues = log[xkey]

    ymin = 10
    ymax = -10

    for key in keys:
        if '/' in key:
            values = log[key]
            param_name, hparam_name = key.rsplit('/', 1)
            param_name_hparam_dict[param_name].append((hparam_name, values))

            if min(values) < ymin:
                ymin = min(values)
            if max(values) > ymax:
                ymax = max(values)

    nrows = 5
    ncols = 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))

    i = 0
    for param_name in param_name_hparam_dict:
        ax_row, ax_col = i // nrows, i % ncols
        for (hparam_name, values) in param_name_hparam_dict[param_name]:
            axs[ax_row, ax_col].plot(xvalues, values, label=hparam_name)
        axs[ax_row, ax_col].set_ylim(ymin, ymax)
        axs[ax_row, ax_col].set_title(param_name[6:], fontsize=10)
        axs[ax_row, ax_col].legend(fontsize=14, fancybox=True, framealpha=0.3)
        i += 1

    plt.tight_layout()


def plot_stuff(log_dir,
               xkey='inner_problem_steps',
               key_identifiers=['net'],
               perf_keys=['F_sum', 'full_val_acc'],
               what_to_show=['uncons_hparams', 'cons_hparams', 'performance', 'online'],
               plot_grads=False,
               inner_problem_len=200,
               plot_inner_problem_ticks=False,
               show_legend=True,
               xlim=None,
               xtick_locs=None,
               xtick_labels=None,
               xscale='linear',
               yscale='log',
               optimal_hparams=None,
               optimal_obj=None):
    log = load_log(log_dir, fname='frequent.csv')

    keys = [key for key in log.keys() if any(key_id in key for key_id in key_identifiers) and ('grad' not in key) and ('cons' not in key)]
    cons_keys = [key for key in log.keys() if any(key_id in key for key_id in key_identifiers) and ('grad' not in key) and ('cons' in key)]
    grad_keys = [key for key in log.keys() if any(key_id in key for key_id in key_identifiers) and 'grad' in key]

    if 'uncons_hparams' in what_to_show:
        plot_hparams(log,
                     keys=keys,
                     xkey=xkey,
                     xlim=xlim,
                     xtick_locs=xtick_locs,
                     xtick_labels=xtick_labels,
                     xscale=xscale,
                     inner_problem_len=inner_problem_len,
                     plot_inner_problem_ticks=plot_inner_problem_ticks,
                     show_legend=show_legend)

    if 'cons_hparams' in what_to_show:
        plot_hparams(log,
                     keys=cons_keys,
                     xkey=xkey,
                     xlim=xlim,
                     xtick_locs=xtick_locs,
                     xtick_labels=xtick_labels,
                     xscale=xscale,
                     yscale='log',
                     inner_problem_len=inner_problem_len,
                     plot_inner_problem_ticks=plot_inner_problem_ticks,
                     show_legend=show_legend)

        if optimal_hparams:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i in range(len(optimal_hparams)):
                plt.axhline(y=optimal_hparams[i], color=colors[i % len(colors)], linestyle='--')

    if plot_grads:
        plot_hparams(log, keys=grad_keys,
                     xkey=xkey,
                     xlim=xlim,
                     xtick_locs=xtick_locs,
                     xtick_labels=xtick_labels,
                     xscale=xscale,
                     inner_problem_len=inner_problem_len,
                     plot_inner_problem_ticks=plot_inner_problem_ticks,
                     show_legend=show_legend,
                     yscale='symlog',
                     ylabel='Hyperparameter Gradient')

    # ---------------------------------------------------

    if 'performance' in what_to_show:
        log = load_log(log_dir, fname='iteration.csv')

        for key in perf_keys:
            plot_performance(log, key,
                             xkey=xkey,
                             xlim=xlim,
                             xtick_locs=xtick_locs,
                             xtick_labels=xtick_labels,
                             xscale=xscale,
                             yscale=yscale,
                             inner_problem_len=inner_problem_len,
                             plot_inner_problem_ticks=plot_inner_problem_ticks)

        if optimal_obj:
            plt.axhline(y=optimal_obj, color='k', linestyle='--')

    if 'print_performance' in what_to_show:
        for metric in ['train_sum_loss', 'val_sum_loss', 'train_mean_loss', 'val_mean_loss']:
            if 'acc' in metric:
                best_value = max(log[metric])
                best_idx = np.argmax(log[metric])
            else:
                best_value = min(log[metric])
                best_idx = np.argmin(log[metric])

            print('Best {}: {}'.format(metric, best_value))
            print('Final {}: {}'.format(metric, log[metric][-1]))

            best_params = {curr_key: log[curr_key][best_idx] for curr_key in cons_keys}
            for curr_key in best_params:
                print('\t{}: {:6.3e}'.format(curr_key, best_params[curr_key]))

            print('\n')

    # ---------------------------------------------------

    if 'online' in what_to_show:
        try:
            log = load_log(log_dir, fname='online.csv')
        except:
            return

        for key in ['full_train_loss', 'full_val_loss']:
        # for key in ['full_train_loss']:
            plot_performance(log, key,
                             xkey=xkey,
                             xlim=xlim,
                             xtick_locs=xtick_locs,
                             xtick_labels=xtick_labels,
                             xscale=xscale,
                             yscale=yscale,
                             inner_problem_len=inner_problem_len,
                             plot_inner_problem_ticks=plot_inner_problem_ticks)


y_label_dict = { 'F_sum': 'Sum of Train Losses',
                 'train_sum_loss': 'Sum of Train Losses',
                 'train_mean_loss': 'Mean of Train Losses',
                 'val_sum_loss': 'Sum of Val Losses',
                 'val_mean_loss': 'Mean of Val Losses',
                 'train_sum_acc': 'Sum of Train Accs',
                 'val_sum_acc': 'Sum of Val Accs',
                 'full_train_loss': 'Train Loss',
                 'full_val_loss': 'Val Loss',
                 'full_train_acc': 'Train Accuracy',
                 'full_val_acc': 'Val Accuracy' }


def get_all_random_search_values(rs_dir, key='train_sum_loss', return_thetas=False, constrained=True):
    all_values = []
    all_thetas = []

    for subdir in os.listdir(rs_dir):
        pkl_path = os.path.join(rs_dir, subdir, 'result.pkl')
        with open(pkl_path, 'rb') as f:
            result = pkl.load(f)
        values = result[key].reshape(-1)
        if constrained:
            thetas = result['thetas_constrained']
        else:
            thetas = result['thetas']
        all_values.append(values)
        all_thetas.append(thetas)

    values = np.concatenate(all_values)
    thetas = np.concatenate(all_thetas)

    if return_thetas:
        return values, thetas
    else:
        return values


def sim_rs_runs(values, num_parallel=2, inner_problem_len=2500, max_loss=1e8, seeds=[5,7,11]):
    all_best_value_lists = []

    # for seed in [5, 7, 11, 13, 17, 23]:
    # for seed in [5, 7, 11, 13, 17]:
    for seed in seeds:
        np.random.seed(seed)
        best_value = max_loss
        total_compute = 0
        total_compute_list = []
        best_value_list = []

        for i in range(100000):
            random_indexes = np.random.randint(low=0, high=len(values), size=num_parallel)
            values_in_current_run = values[random_indexes]
            best_value_in_run = np.nanmin(values_in_current_run)
            if best_value_in_run < best_value:
                best_value = best_value_in_run
            total_compute += inner_problem_len * num_parallel
            total_compute_list.append(total_compute)
            best_value_list.append(best_value)

        all_best_value_lists.append(best_value_list)

    best_values_np = np.array(all_best_value_lists)
    best_values_min = np.nanmin(best_values_np, axis=0)
    best_values_max = np.nanmax(best_values_np, axis=0)
    best_values_mean = np.nanmean(best_values_np, axis=0)

    return total_compute_list, best_values_mean, best_values_min, best_values_max


def get_mean_min_max(exp_dir, N):
    running_value_list = []
    total_compute_list = []

    for log_fname in os.listdir(exp_dir):
        log_path = os.path.join(exp_dir, log_fname)
        log = load_log(log_path, fname='iteration.csv')
        total_compute = np.array(log['inner_problem_steps']) * N  # N is the number of particles
        running_values = np.minimum.accumulate(log['F_sum'])
        running_value_list.append(running_values)
        total_compute_list.append(total_compute)

    result_array = np.array(running_value_list)
    result_mean = np.mean(result_array, axis=0)
    result_min = np.min(result_array, axis=0)
    result_max = np.max(result_array, axis=0)

    return total_compute_list[0], result_mean, result_min, result_max


def get_mean_min_max_bo(log_paths, inner_problem_len, min_or_max='min'):
    running_value_list = []
    total_compute_list = []

    shortest_list_len = 1e9

    for log_path in log_paths:
        # with tf.io.gfile.GFile(os.path.join(log_path, 'logs.json'), 'r') as f:
        with open(os.path.join(log_path, 'logs.json'), 'r') as f:
            text = f.read()

        all_values = []
        total_compute = []

        for (i,line) in enumerate(text.splitlines()):
            result = json.loads(line)
            all_values.append(-result['target'])
            total_compute.append(i * inner_problem_len)

        if min_or_max == 'min':
            running_values = np.minimum.accumulate(all_values)
        elif min_or_max == 'max':
            running_values = np.maximum.accumulate(all_values)

        running_value_list.append(running_values)
        total_compute_list.append(total_compute)
        if len(total_compute) < shortest_list_len:
            shortest_list_len = len(total_compute)

    running_value_list = [lst[:shortest_list_len] for lst in running_value_list]
    result_array = np.array(running_value_list)
    result_mean = np.mean(result_array, axis=0)
    result_min = np.min(result_array, axis=0)
    result_max = np.max(result_array, axis=0)

    return total_compute_list[0][:shortest_list_len], result_mean, result_min, result_max
