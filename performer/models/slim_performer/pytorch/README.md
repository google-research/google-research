# SLiM-Performer (PyTorch)

Evaluated under Python 3.6.9 and PyTorch 1.7.0.

To reduce repository memory, the following data can be downloaded:

* [Penn Treebank](https://github.com/wojzaremba/lstm/tree/master/data)
* [Enwik8](http://prize.hutter1.net/)

The copy task data is generated randomly.

The following are the commands used to produce the experimental results in the paper.

Penn Treebank commands:

* `python3 train.py --arg_code=0` for the full setup.
* `python3 train.py --arg_code=1` for the C = 512 setup.
* `python3 train.py --arg_code=2` for the C = 256 setup.
* `python3 train.py --arg_code=3` for the C = 512 fine-tuning setup.
* `python3 train.py --arg_code=4` for the C = 256 fine-tuning setup.

Enwik8 commands:

* `python3 train.py --arg_code=5` for the full setup.
* `python3 train.py --arg_code=6` for the C = 2048 setup.
* `python3 train.py --arg_code=7` for the C = 1366 setup.
* `python3 train.py --arg_code=8` for the C = 2048 fine-tuning setup.
* `python3 train.py --arg_code=9` for the C = 1366 fine-tuning setup.

Copying task commands:

* `python3 train.py --arg_code 10` for the full setup.
* `python3 train.py --arg_code 11` for the C = 128 setup.
* `python3 train.py --arg_code 12` for the C = 64 setup.
* `python3 train.py --arg_code 13` for the C = 128 fine-tuning setup.
* `python3 train.py --arg_code 14` for the C = 64 fine-tuning setup.
