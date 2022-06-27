# Environment Generation for Zero-Shot Compositional Reinforcement Learning

## Overview
This repository includes the implementation of `Environment Generation for Zero-Shot Compositional Reinforcement Learning' by Izzeddin Gur, Natasha Jaques, Yingjie Miao, Jongwook Choi,Manoj Tiwari, Honglak Lee, and Aleksandra Faust. NeurIPS'21.

There are two core components: (i) CoDE, Compositional Design of Environments, and (ii) the gMiniWoB framework.
For now, it only hosts the main gMiniWoB implementation. The rest will be added soon.

## Installation

* Download contents: ```svn export https://github.com/google-research/google-research/trunk/compositional_rl```

## Install Dependencies
### Install Bootstrap

* Create a new directory: ```mkdir /path/to/compositional_rl/gwob/bootstrap/```
* Download bootstrap files from https://github.com/twbs/bootstrap/releases/download/v4.3.1/bootstrap-4.3.1-dist.zip
* Extract ```/path/to/bootstrap/js/bootstrap.min.js``` and ```/path/to/bootstrap/css/bootstrap.min.css``` to ```/path/to/compositional_rl/gwob/bootstrap/```.

### Install MiniWoB

* Clone the MiniWoB project: ```git clone https://github.com/stanfordnlp/miniwob-plusplus```
* Put the whole directory ```/path/to/miniwob-plusplus``` under ```/path/to/compositional_rl/gwob/```

You should now have three folders ```/path/to/compositional_rl/gwob/gminiwob/```, ```/path/to/compositional_rl/gwob/bootstrap/```, and ```/path/to/compositional_rl/gwob/miniwob-plusplus/```.


## Generating random websites in gMiniWoB

* Open ```file:///path/to/compositional_rl/gwob/gminiwob/sample_random_website.html``` in a browser and click "START".
* Each time the "START" button is clicked, this will create a random gMiniWoB website using a subset of primitives available in gMiniWoB.
