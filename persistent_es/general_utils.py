"""General utility functions.
"""
from jax.tree_util import tree_flatten


def recursive_keys(dictionary, upper_key=''):
  all_keys = []
  for key, value in dictionary.items():
    try:
      value.keys()  # Try to see if value is a dictionary with keys
      all_keys += recursive_keys(value, key)
    except:
      all_keys += ['{}/{}'.format(upper_key, key)]
  return all_keys


def count_params(params):
    value_flat, value_tree = tree_flatten(params)
    return sum([v.size for v in value_flat])


def save_command(argv, fname):
  with open(fname, 'w') as f:
    f.write(
        '\n'.join(['python {} \\'.format(argv[0])] +
                  ["    {} \\".format(line) for line in argv[1:-1]] +
                  ['    {}'.format(argv[-1])])
    )
