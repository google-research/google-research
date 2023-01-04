# Environment Generation for Zero-Shot Compositional Reinforcement Learning

## Overview
This repository includes the implementation of `Environment Generation for Zero-Shot Compositional Reinforcement Learning by Izzeddin Gur, Natasha Jaques, Yingjie Miao, Jongwook Choi,Manoj Tiwari, Honglak Lee, and Aleksandra Faust. NeurIPS'21`.

## Getting Started

* Download this repo

```
svn export https://github.com/google-research/google-research/trunk/compositional_rl
```

### Install Dependencies
#### Install Bootstrap

* Download bootstrap files:

```
mkdir gwob/bootstrap/ && cd gwob/bootstrap && wget https://github.com/twbs/bootstrap/releases/download/v4.3.1/bootstrap-4.3.1-dist.zip
```

* Unzip and extract:

```
unzip bootstrap-4.3.1-dist.zip && cp bootstrap-4.3.1-dist/css/bootstrap.min.css . && cp bootstrap-4.3.1-dist/js/bootstrap.min.js . && rm -r bootstrap-4.3.1-dist* && cd ../../
```

#### Install MiniWoB

* Clone the MiniWoB project:

```
git clone https://github.com/stanfordnlp/miniwob-plusplus gwob/miniwob-plusplus
```

* Checkout the version that we used in our project:

```
cd gwob/miniwob-plusplus && git checkout 833a477a8fbfbd2497e95fee019f76df2b9bd75e
```

* Convert all python files from Python 2 to Python 3:

```
pip install 2to3 && cd ../../ && 2to3 gwob/miniwob-plusplus/python/miniwob -w
```

* Integrate miniwob by making necessary changes:

```
python3 integrate_miniwob.py
```

* Install miniwob:

```
pip install gwob/miniwob-plusplus/python/
```

* Install the [ChromeDriver](https://chromedriver.chromium.org/downloads) with the version that is matching your Chrome browser:

```
export PATH=$PATH:/path/to/chromedriver
```

### Install gMiniWoB
* Install gminiwob:

```
pip install gwob/
```

## Examples

### Generating random websites in gMiniWoB

* Open ```file:///path/to/compositional_rl/gwob/gminiwob/sample_random_website.html``` in a browser and click "START".
* Each time the "START" button is clicked, this will create a random gMiniWoB website using a subset of primitives available in gMiniWoB.

### Running a rule-based policy with a fixed test environment
* Run `python3 gwob/examples/web_environment_example.py --data_dep_path='/path/to/compositional_rl/gwob/` to run a rule-based policy for a simulated shopping website. If you get any errors related to non-headless browsing, make sure to pass `--run_headless_mode=True`.

### Environment design and Q-value network
* The following is a simple tutorial for randomly designing an environment and
using an LSTM-based DQN to generate logits and values.

```
import gin
import numpy as np

from CoDE import test_websites
from CoDE import utils
from CoDE import vocabulary_node
from CoDE import web_environment
from CoDE import web_primitives
from CoDE import q_networks

gin.parse_config_files_and_bindings(["/path/to/compositional_rl/gwob/configs/envdesign.gin"], None)

# Create an empty environment.
env = web_environment.GMiniWoBWebEnvironment(
  base_url="file:///path/to/compositional_rl/gwob/",
  global_vocabulary=vocabulary_node.LockedVocabulary())

# Create a q network.
q_net = q_networks.DQNWebLSTM(vocab_size=env.local_vocab.max_vocabulary_size, return_state_value=True)

# Sample a new design of the form {'number_of_pages': Integer, 'action': List[Integer], 'action_page': List[Integer]}.
# `action` denotes primitive indices and `action_page` denotes their corresponding page indices.
# Each item in the `action_page` should be less than `number_of_pages`.
# For this tutorial, we will randomly sample a design.
number_of_pages = np.random.randint(4) + 1
design =  {'number_of_pages': number_of_pages,
            'action': np.random.choice(np.arange(len(web_primitives.CONCEPTS)), 5),
            'action_page': np.random.choice(np.arange(number_of_pages), 5)}

# Design the actual environment.
env.design_environment(
    design, auto_num_pages=True)

# Reset the environment.
state = env.reset()

# Add batch dimension.
state = {key: np.expand_dims(tensor, axis=0) for key, tensor in state.items()}

# Get flattened logits and values.
logits, values = q_net(state)

# Get greedy action.
action = np.argmax(logits)

# Run the action.
new_state, reward, done, info = env.step(action)

```
