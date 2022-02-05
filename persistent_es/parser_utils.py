"""Creates a command-line argument parser for use with hyperopt.py
"""
import argparse


def create_parser():
  parser = argparse.ArgumentParser(description='Hyperopt with ES/PES')

  # Dataset and model hyperparameters
  parser.add_argument('--dataset', type=str, default='mnist',
                      help='Dataset')
  parser.add_argument('--model', type=str, default='mlp',
                      choices=['mlp', 'resnet'],
                      help='Model')
  parser.add_argument('--model_size', type=str, default='small',
                      help='Model size, that affects #channels in the ResNet '
                           '(tiny, small, med, or large)')
  parser.add_argument('--nlayers', type=int, default=2,
                      help='Number of layers in the MLP')
  parser.add_argument('--nhid', type=int, default=100,
                      help='Number of hidden units in each layer of the MLP')
  parser.add_argument('--activation', type=str, default='relu',
                      help='Activation function')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Batch size')

  # Outer optimization hyperparameters
  parser.add_argument('--outer_iterations', type=int, default=5000,
                      help='Number of meta-optimization iterations')
  parser.add_argument('--outer_optimizer', type=str, default='adam',
                      help='Outer optimizer')
  parser.add_argument('--outer_lr', type=float, default=1e-2,
                      help='Outer learning rate')
  parser.add_argument('--outer_b1', type=float, default=0.9,
                      help='Outer optimizer Adam b1 hyperparameter')
  parser.add_argument('--outer_b2', type=float, default=0.999,
                      help='Outer optimizer Adam b2 hyperparameter')
  parser.add_argument('--outer_eps', type=float, default=1e-8,
                      help='Outer optimizer epsilon hyperparameter')

  # Inner optimization hyperparameters
  parser.add_argument('--inner_optimizer', type=str, default='sgdm',
                      help='Inner optimizer')
  parser.add_argument('--tune_params', type=str, default='lr:itd',
                      help='A comma-separated string of hyperparameters to '
                           'search over')
  parser.add_argument('--objective', type=str, default='train_sum_loss',
                      help='The objective for meta-optimization')
  parser.add_argument('--objective_batch_size', type=int, default=100,
                      help='The batch size for computing the meta-objective'
                           'after each unroll')
  parser.add_argument('--lr', type=float, default=1e-5,
                      help='Learning rate')
  parser.add_argument('--lr0', type=float, default=1e-5,
                      help='Initial LR for a linear schedule')
  parser.add_argument('--lr1', type=float, default=1e-5,
                      help='Final LR for a linear schedule')
  parser.add_argument('--b1', type=float, default=0.99,
                      help='Adam b1 hyperparameter')
  parser.add_argument('--b2', type=float, default=0.999,
                      help='Adam b2 hyperparameter')
  parser.add_argument('--eps', type=float, default=1e-8,
                      help='Adam epsilon hyperparameter')
  parser.add_argument('--mom', type=float, default=0.9,
                      help='Momentum')
  parser.add_argument('--wd', type=float, default=1e-10,
                      help='Weight decay')
  parser.add_argument('--random_hparam_init', action='store_true', default=False,
                      help='Whether to initialize the hparams to random values')

  # ES/PES gradient estimator hyperparameters
  parser.add_argument('--estimate', type=str, default='pes',
                      help='Type of gradient estimate (es or pes)')
  parser.add_argument('--T', type=int, default=5000,
                      help='Maximum number of iterations of the inner loop')
  parser.add_argument('--K', type=int, default=10,
                      help='Number of steps to unroll (== truncation length)')
  parser.add_argument('--sigma', type=float, default=0.1,
                      help='Variance for ES/PES perturbations')
  parser.add_argument('--n_chunks', type=int, default=1,
                      help='Number of particle chunks for ES/PES')
  parser.add_argument('--n_per_chunk', type=int, default=10,
                      help='Number of particles per chunk for ES/PES')
  parser.add_argument('--telescoping', action='store_true', default=False,
                      help='Whether to use telescoping sums')

  # Clipping
  parser.add_argument('--outer_clip', type=float, default=-1,
                      help='Outer gradient clipping (-1 means no clipping)')
  parser.add_argument('--inner_clip', type=float, default=-1,
                      help='Gradient clipping for each step of the inner unroll'
                           '(-1 means no grad clipping)')

  # Logging hyperparameters
  parser.add_argument('--num_eval_runs', type=int, default=10,
                      help='Number of runs to average over when evaluating')
  parser.add_argument('--print_every', type=int, default=10,
                      help='Print theta every N iterations')
  parser.add_argument('--log_every', type=int, default=10,
                      help='Log the full training and val losses to the CSV '
                           'log every N iterations')
  parser.add_argument('--eval_every', type=int, default=500,
                      help='Log the full training and val losses to the CSV '
                           'log every N iterations')

  # Saving hyperparameters
  parser.add_argument('--save_dir', type=str, default='saves',
                      help='Save directory')
  parser.add_argument('--seed', type=int, default=3,
                      help='Random seed')
  return parser
