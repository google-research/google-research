# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Make necessary changes in MiniWoB classes to integrate gMiniWoB."""

if __name__ == '__main__':
  lines = open('gwob/miniwob-plusplus/python/miniwob/instance.py').readlines()
  lines = lines[0:84] + [
      """
        elif subdomain.startswith('gminiwob.') or subdomain.startswith('gwob.'):
          self.url = urllib.parse.urljoin(base_url, '{}/{}.html'.format(subdomain[0:subdomain.index('.')], subdomain[subdomain.index('.')+1:]))
          self.window_width = self.FLIGHT_WINDOW_WIDTH
          self.window_height = self.FLIGHT_WINDOW_HEIGHT
          self.task_width = self.FLIGHT_TASK_WIDTH
          self.task_height = self.FLIGHT_TASK_HEIGHT\n"""
  ] + lines[84:]

  with open('gwob/miniwob-plusplus/python/miniwob/instance.py', 'w') as fout:
    fout.writelines(lines)

  lines = open('gwob/miniwob-plusplus/python/miniwob/fields.py').readlines()
  lines = lines[0:11] + [
      """
def default_field_extractor(task_name):
  def extractor(utterance):
    try:
      fields = json.loads(utterance)
      return Fields({str(x): str(y) for (x, y) in fields.items()})
    except:
      print(utterance)
      raise ValueError('{} does not have a field extractor.'.format(task_name))
  return extractor


def get_field_extractor(task_name):
  try:
    return FIELD_EXTRACTORS[task_name]
  except KeyError:
    return default_field_extractor(task_name)
                        """
  ] + lines[19:]

  with open('gwob/miniwob-plusplus/python/miniwob/fields.py', 'w') as fout:
    fout.writelines(lines)

  lines = open('gwob/miniwob-plusplus/python/miniwob/state.py').readlines()
  lines = lines[0:144] + [
      "        self._placeholder = raw_dom.get('placeholder')\n"
  ] + lines[144:242] + [
      """
    @property
    def placeholder(self):
        return self._placeholder\n\n"""
  ] + lines[242:388] + ['    def diff(self, other_dom, interactive=False):\n'
                       ] + lines[389:436] + [
                           '                        or (interactive and '+
                           'first.tampered != second.tampered)\n'
                       ] + lines[437:]

  with open('gwob/miniwob-plusplus/python/miniwob/state.py', 'w') as fout:
    fout.writelines(lines)

  with open('gwob/miniwob-plusplus/python/setup.py', 'w') as fout:
    fout.write("""

from setuptools import setup

# Copied from the main miniwob_plusplus repository.
setup(
    name="miniwob_plusplus",
    version="0.0.1",
    python_requires=">=3.7, <3.11",
    packages=["miniwob"],
    install_requires=[
        "Gymnasium==0.26.3",
        "Pillow>=9.0.0",
        "selenium>=4.5.0",
        "numpy>=1.18.0",
    ],
    extras_require={
        "testing": [
            "pytest>=7.0.0",
            "pytest-timeout>=2.1.0",
        ]
    },
)
""")
