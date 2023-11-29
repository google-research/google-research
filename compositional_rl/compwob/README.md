# Language Model Agents Suffer from Compositional Decision Making

## Overview
This repository includes the implementation of CompWoB (i.e. Compositional MiniWoB) presented in `Language Model Agents Suffer from Compositional Decision Making by Hiroki Furuta, Yutaka Matsuo, Aleksandra Faust, and Izzeddin Gur`.

## Getting Started

* Download this repo:

```bash
svn export https://github.com/google-research/google-research/trunk/compositional_rl/compwob
```

### Install Dependencies
#### Install MiniWoB

* Clone original MiniWoB project (from legacy branch):

```bash
git clone -b legacy https://github.com/Farama-Foundation/miniwob-plusplus.git
```

* Convert all python files from Python 2 to Python 3:

```bash
pip install 2to3
2to3 miniwob-plusplus/python/miniwob -w
```

* Integrate MiniWoB by making necessary changes:

```bash
python3 integrate_compwob.py
```

* Copy necessary files for CompWoB to original MiniWoB:
```bash
# replace fields.py
cp compwob/fields.py miniwob-plusplus/python/miniwob/fields.py

# copy HTML files
cp -r compwob/compositional miniwob-plusplus/html/compositional
```

* Install MiniWoB:

```bash
pip install gwob/miniwob-plusplus/python/
```

* Install the [ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/) with the version that is matching your Chrome browser:

```bash
export PATH=$PATH:/path/to/chromedriver
```

### Examples
You can load CompWoB tasks the same as base MiniWoB tasks. Please see original paper for the list of 50 compositional tasks.

```python
import os

from miniwob_plusplus.environment import MiniWoBEnvironment

# task name should be start with `compositional.`
env = MiniWoBEnvironment('compositional.click-button_click-link')
base_url = os.environ.get('MINIWOB_BASE_URL')
print('BASE URL:', base_url)
env.configure(num_instances=2, seeds=[0, 1], base_url=base_url)
```
