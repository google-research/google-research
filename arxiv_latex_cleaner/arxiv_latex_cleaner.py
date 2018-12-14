# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Cleans the LaTeX code of your paper to submit to arXiv."""
import argparse
import collections
import json
import os
import re
import shutil

from PIL import Image


def _create_dir_erase_if_exists(path):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)


def _create_dir_if_not_exists(path):
  if not os.path.exists(path):
    os.makedirs(path)


def _keep_pattern(haystack, patterns_to_keep):
  """Keeps the strings that match 'patterns_to_keep'."""
  out = []
  for item in haystack:
    if any((re.findall(rem, item) for rem in patterns_to_keep)):
      out.append(item)
  return out


def _remove_pattern(haystack, patterns_to_remove):
  """Removes the strings that match 'patterns_to_remove'."""
  return [
      item for item in haystack
      if item not in _keep_pattern(haystack, patterns_to_remove)
  ]


def _list_all_files(in_folder, ignore_dirs=None):
  if ignore_dirs is None:
    ignore_dirs = []
  to_consider = [
      os.path.join(os.path.relpath(path, in_folder), name)
      for path, _, files in os.walk(in_folder)
      for name in files
  ]
  return _remove_pattern(to_consider, ignore_dirs)


def _copy_file(filename, params):
  _create_dir_if_not_exists(
      os.path.join(params['output_folder'], os.path.dirname(filename)))
  shutil.copy(
      os.path.join(params['input_folder'], filename),
      os.path.join(params['output_folder'], filename))


def _remove_comments(text):
  """Removes the comments from the string 'text'."""
  if 'auto-ignore' in text:
    return text
  if text.lstrip(' ').startswith('%'):
    return ''
  match = re.search(r'(?<!\\)%', text)
  if match:
    return text[:match.end()] + '\n'
  else:
    return text


def _read_file_content(filename):
  with open(filename, 'r') as fp:
    return fp.readlines()


def _write_file_content(content, filename):
  with open(filename, 'w') as fp:
    return fp.write(content)


def _read_remove_comments_and_write_file(filename, parameters):
  """Reads a file, erases all LaTeX comments in the content, and writes it."""
  _create_dir_if_not_exists(
      os.path.join(parameters['output_folder'], os.path.dirname(filename)))
  content = _read_file_content(
      os.path.join(parameters['input_folder'], filename))
  content_out = [_remove_comments(line) for line in content]
  _write_file_content(''.join(content_out),
                      os.path.join(parameters['output_folder'], filename))


def _resize_and_copy_figure(filename,
                            origin_folder,
                            destination_folder,
                            dest_size=600):
  """Copies a file while erasing all the LaTeX comments in its content."""
  _create_dir_if_not_exists(
      os.path.join(destination_folder, os.path.dirname(filename)))

  if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
    im = Image.open(os.path.join(origin_folder, filename))
    if max(im.size) > dest_size:
      im = im.resize(
          tuple([int(x * float(dest_size) / max(im.size)) for x in im.size]),
          Image.ANTIALIAS)
    if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg']:
      im.save(os.path.join(destination_folder, filename), 'JPEG', quality=90)
    elif os.path.splitext(filename)[1].lower() in ['.png']:
      im.save(os.path.join(destination_folder, filename), 'PNG')
  else:
    shutil.copy(
        os.path.join(origin_folder, filename),
        os.path.join(destination_folder, filename))


def _resize_and_copy_figures(parameters, splits):
  out_size = collections.defaultdict(lambda: parameters['im_size'])
  out_size.update(parameters['images_whitelist'])
  for image_file in _keep_only_referenced(
      splits['figures'],
      [os.path.join(parameters['output_folder'], fn) for fn in splits['tex']]):
    _resize_and_copy_figure(
        image_file,
        parameters['input_folder'],
        parameters['output_folder'],
        dest_size=out_size[image_file])


def _keep_only_referenced(filenames, container_files):
  """Returns the filenames referenced from the content of container_files."""
  referenced = set()
  for container in container_files:
    with open(container, 'r') as fp:
      data = fp.read()

    for fn in filenames:
      if os.path.splitext(fn)[0] in data:
        referenced.add(fn)

  return referenced


def _split_all_files(parameters):
  """Splits the files into types or location to know what to do with them."""
  file_splits = {
      'all':
          _list_all_files(
              parameters['input_folder'], ignore_dirs=['.git' + os.sep]),
      'in_root': [
          f for f in os.listdir(parameters['input_folder'])
          if os.path.isfile(os.path.join(parameters['input_folder'], f))
      ]
  }
  file_splits['not_in_root'] = [
      f for f in file_splits['all'] if f not in file_splits['in_root']
  ]
  file_splits['to_copy_in_root'] = _remove_pattern(
      file_splits['in_root'], parameters['to_delete_in_root'] +
      parameters['figures_to_copy_if_referenced'])
  file_splits['to_copy_not_in_root'] = _remove_pattern(
      file_splits['not_in_root'], parameters['to_delete_in_root'] +
      parameters['figures_to_copy_if_referenced'])
  file_splits['figures'] = _keep_pattern(
      file_splits['all'], parameters['figures_to_copy_if_referenced'])

  file_splits['tex'] = _keep_pattern(
      file_splits['to_copy_in_root'] + file_splits['to_copy_not_in_root'],
      ['.tex$'])
  file_splits['non_tex'] = _remove_pattern(
      file_splits['to_copy_in_root'] + file_splits['to_copy_not_in_root'],
      ['.tex$'])

  return file_splits


def _create_out_folder(input_folder):
  """Creates the output folder, erasing it if existed."""
  out_folder = input_folder.rstrip(os.sep) + '_arXiv'
  _create_dir_erase_if_exists(out_folder)

  return out_folder


def _handle_arguments():
  """Defines and returns the arguments."""
  parser = argparse.ArgumentParser(
      description=('Clean the LaTeX code of your paper to submit to arXiv. '
                   'Check the README for more information on the use.'))
  parser.add_argument(
      'input_folder', type=str, help='Input folder containing the LaTeX code.')
  parser.add_argument(
      '--im_size',
      default=500,
      type=int,
      help=('Size of the output images (in pixels, longest side). Fine tune '
            'this to get as close to 10MB as possible.'))
  parser.add_argument(
      '--images_whitelist',
      default={},
      type=json.loads,
      help=('Images that won\'t be resized to the default resolution, but the '
            'one provided here in a dictionary as follows '
            '\'{"path/to/im.jpg": 1000}\''))

  return vars(parser.parse_args())


def _run_arxiv_cleaner(parameters):
  """Core of the code, runs the actual arXiv cleaner."""
  parameters.update({
      'to_delete_in_root': [
          '.aux$', '.sh$', '.bib$', '.blg$', '.log$', '.out$', '.ps$', '.dvi$',
          '.synctex.gz$', '~$', '.backup$', '.gitignore$', '.DS_Store$', '.svg$'
      ],
      'to_delete_not_in_root': ['.DS_Store$', '.gitignore$', '.svg$'],
      'figures_to_copy_if_referenced': ['.png$', '.jpg$', '.jpeg$', '.pdf$']
  })

  splits = _split_all_files(parameters)

  parameters['output_folder'] = _create_out_folder(parameters['input_folder'])

  for non_tex_file in splits['non_tex']:
    _copy_file(non_tex_file, parameters)
  for tex_file in splits['tex']:
    _read_remove_comments_and_write_file(tex_file, parameters)

  _resize_and_copy_figures(parameters, splits)


def main():
  command_line_parameters = _handle_arguments()
  _run_arxiv_cleaner(command_line_parameters)


if __name__ == '__main__':
  main()
