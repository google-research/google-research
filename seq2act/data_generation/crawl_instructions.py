# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Crawl instruction from html page."""

import collections
import glob
import gzip
import json
import os
import re

from absl import app
from absl import flags
from absl import logging
from bs4 import BeautifulSoup

FLAGS = flags.FLAGS

flags.DEFINE_string('input_warc_dir', None,
                    'The directory of the downloaded WARC files.')
flags.DEFINE_string('output_instruction_json', None,
                    'The path of the generated instruction json file.')

# pylint: disable=line-too-long
URL_WHITE_LIST = [
    'support.google.com',
    'www.wikihow.com',
    'www.androidauthority.com',
    'www.androidcentral.com',
    'www.cnet.com',
    'joyofandroid.com',
    'www.bt.com',

    'www.t-mobile.com',
    'www.samsung.com',
    'www.guidingtech.com',
    'www.htc.com',
    'support.bell.ca',
    'consumer.huawei.com',
    'www.helpforsmartphone.com',
    'www.makeuseof.com',

    'steps.app',
    'support.walkertracker.com',
    'support.apple.com',

    'www.lifewire.com',
    'www.techbone.net',
    'support.microsoft.com',
]


flags.DEFINE_boolean('filter_domain', True, 'Whether to filter domains.')


def process_html(url, content):
  """Processes single html webpage and extracts instructions as tasks."""

  returnme = []

  domain = url.split('://')[1].split('/')[0]
  soup = BeautifulSoup(content, 'html.parser')

  # Remove unnecessary tags which could exist in <ol>
  for s in soup.select('script'):
    s.extract()
  for s in soup.select('noscript'):
    s.extract()
  for s in soup.select('table'):
    s.extract()
  for s in soup.select('figure'):
    s.extract()

  if domain == 'www.lifewire.com':
    for s in soup.find_all('div', {'class': 'theme-experttiptip'}):
      s.extract()
    for s in soup.find_all('div', {'class': 'theme-experttipimportant'}):
      s.extract()

  # For specific websites, need fine tune the parser to remove (.extract()) some
  # unnecessary tags to clean up the result got from ol.get_text()
  if domain == 'www.wikihow.com':
    for s in soup.select('span'):
      s.extract()

  ols = soup.find_all('ol')
  for _, ol in enumerate(ols):

    if domain == 'support.google.com':
      for s in ol.find_all('img'):
        # In Google support web, the 'alt' text are duplicated with text ahead
        # But the arrow image should be replaced with its alt, see both example:
        # https://support.google.com/pixelphone/answer/7444033
        if s['alt'].lower().strip() == 'and then':
          s.replace_with('and then')
    else:
      for s in ol.find_all('img'):
        s.replace_with(s['alt'])

    if domain in ['steps.app', 'www.techbone.net']:
      # This website has no separater between steps, if call get_text(), the
      # words between steps will mess up.
      instruction_got = ol.get_text('. ', strip=True)
    else:
      # Replace any HTML tag with a space, especially between steps of instruction
      # See https://www.crummy.com/software/BeautifulSoup/bs4/doc/#get-text
      instruction_got = ol.get_text(' ', strip=True)

    processed_str = _replace_unicode_with_space(instruction_got)
    # Decide whether the instruction is Android-related by URL/instruction.
    # Sometimes instruction does not contain "android" but it's indeed valid, so
    # add url as part of the text.
    if _is_valid(url.split('?')[0], processed_str):
      returnme.append(processed_str)

  return returnme


def _replace_unicode_with_space(text):
  """Replaces all unwanted unicode chars with single space."""
  returnme = ''.join([i if ord(i) < 128 else ' ' for i in text])
  returnme = ' '.join(returnme.split())  # Change all space/newline to one space
  return returnme


def _is_valid(url, inst):
  url_words = re.compile(r'\w+').findall(url.lower())
  instruction_words = re.compile(r'\w+').findall(inst.lower())

  phone_set = {'android', 'phone', 'iphone'}
  click_set = {'tap', 'click'}

  return (set(url_words + instruction_words).intersection(phone_set) and
          set(instruction_words).intersection(click_set))


# DomainStatsIdx
COUNT_IN_WARC = 0
COUNT_IS_RESPONSE = 1
COUNT_HTML = 2
COUNT_HTML_HAS_INST = 3
COUNT_INST = 4


def _parse_one_page(lines, stats, domain_stats):
  """Parses one page in warc file.

  Args:
    lines: the lines of WARC content to parse, which should contain single web
      interaction info, such as a request or a response
    stats: dict of {string, int}, for reason of failure and count
    domain_stats: dict of {domain: [a, b, c, d, e]} which are the counts of
      different DomainStatsIdx items for each domain
  Returns:
    list of triple (url, instruction, html_content) for each instruction found.
  """
  if not lines:
    return []
  if lines[0].strip() != 'WARC/1.0':
    stats['Error_no_WARC/1.0_in_head'] += 1
    return []

  url = None
  warc_type = None
  section = 1
  html_lines = []
  for _, line in enumerate(lines):
    line = line.strip()
    if section < 3:
      if not line:
        section += 1
    if section == 1:
      if line.startswith('WARC-Type: '):
        warc_type = line[len('WARC-Type: '):].strip()
      if line.startswith('WARC-Target-URI: '):
        url = line[len('WARC-Target-URI: '):].strip()
        # Extract support.google.com from
        # https://support.google.com/news/publisher-center/answer/9603942
        domain = url.split('://')[1].split('/')[0]
        if FLAGS.filter_domain:
          if domain not in URL_WHITE_LIST:
            stats['NotFound_Domain_mismatch'] += 1
            return []
        domain_stats['DOMAIN_' + domain][COUNT_IN_WARC] += 1
        if warc_type == 'response':
          domain_stats['DOMAIN_' + domain][COUNT_IS_RESPONSE] += 1

    if section == 3 and line:  # section 3 is html:
      html_lines.append(line)

  if not url or not html_lines:
    stats['No_HTML'] += 1
    return []

  domain_stats['DOMAIN_' + domain][COUNT_HTML] += 1

  try:
    html_content = '\n'.join(html_lines)
    instructions = process_html(url, html_content)
  except Exception:  # pylint: disable=broad-except
    stats['Error_parse_html'] += 1
    return []

  if not instructions:
    stats['No_instruction'] += 1
    return []

  stats['Got'] += 1
  domain_stats['DOMAIN_' + domain][COUNT_HTML_HAS_INST] += 1
  domain_stats['DOMAIN_' + domain][COUNT_INST] += len(
      instructions)
  return [(url, index, instruction)
          for index, instruction in enumerate(instructions)]


def extract_instructions_from_warc_file(warc_file_path, file_handler):
  """Reads instruction from WARC file.

  Args:
    warc_file_path: warc file path.
    file_handler: file handler of the warc file.
  Yields:
    triple(url, index, instruction)
  """
  lines_of_one_page = []
  stats = collections.defaultdict(int)
  domain_stats = collections.defaultdict(lambda: [0, 0, 0, 0, 0])

  for line in file_handler:
    if line.strip().startswith('WARC/1.0'):
      stats['Total'] += 1
      urls_and_instructions = _parse_one_page(lines_of_one_page,
                                              stats, domain_stats)
      for triple in urls_and_instructions:
        yield triple
      lines_of_one_page = [line]
    else:
      lines_of_one_page.append(line)

  urls_and_instructions = _parse_one_page(lines_of_one_page,
                                          stats, domain_stats)
  stats['file_name'] = warc_file_path

  if FLAGS.filter_domain:  # without filter, the log will be too long
    logging.info(json.dumps({**stats, **domain_stats}))
  for triple in urls_and_instructions:
    yield triple


def main(_):
  # This is for the downloaded WARC files if they are stored in local device.
  # If the downloaded WARC files are stored in your own remote file system,
  # please costomize this part.
  result = []
  for warc_file in glob.glob(os.path.join(FLAGS.input_warc_dir, '*.warc.gz')):
    with open(warc_file, 'rb') as f1:
      with gzip.open(f1, mode='rt', encoding='latin1') as f2:
        for (url, index, instruction
             ) in extract_instructions_from_warc_file(warc_file, f2):
          output = {
              'file_name': warc_file,
              'instructions': instruction,
              'url': url,
              'index': index,
          }
          result.append(json.dumps(output))

  with open(FLAGS.output_instruction_json, 'w+') as f:
    for json_str in result:
      f.write(json_str + '\n')


if __name__ == '__main__':
  FLAGS.set_default('logtostderr', True)
  app.run(main)

