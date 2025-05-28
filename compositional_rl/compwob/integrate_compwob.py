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

"""Make necessary changes in MiniWoB classes to integrate CompWoB."""

if __name__ == '__main__':
  lines = open('miniwob-plusplus/python/miniwob/instance.py').readlines()
  lines = lines[0:84] + [
      """
        elif subdomain.startswith('compositional.'):
          subdomain = subdomain[subdomain.index('.')+1:].strip()
          self.url = urllib.parse.urljoin(base_url, 'compositional/{}.html'.format(subdomain))
          self.window_width = self.WINDOW_WIDTH
          self.window_height = self.WINDOW_HEIGHT
          self.task_width = self.TASK_WIDTH
          self.task_height = self.TASK_HEIGHT\n"""
  ] + lines[84:]

  with open('miniwob-plusplus/python/miniwob/instance.py', 'w') as fout:
    fout.writelines(lines)

  with open('miniwob-plusplus/python/setup.py', 'w') as fout:
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
