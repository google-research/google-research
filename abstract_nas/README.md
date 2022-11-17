# αNAS: Neural Architecture Search using Property Guided Synthesis

This repository contains the code to reproduce the results for the paper
αNAS: Neural Architecture Search using Property Guided Synthesis

Link to paper https://arxiv.org/abs/2205.03960

To run a toy example locally, from the parent directory:

```shell
./abstract_nas/run.sh
```

This will start an instance of evolutionary search for a 2-layer CNN on
CIFAR-10 for 50 generations. Depending on the resources available on your local
machine this might take a very long time, so instead we suggest using a free
Google Colab instance with TPUs to reproduce our results.

To run all tests, from the parent directory:

```shell
python -m unittest discover -s abstract_nas -p "*_test.py" -v
```

The tests should take about an hour to complete.

## Inquiry

If you have questions about this particular project, please contact
`ccj[at]csail.mit.edu` instead of using GitHub issues (as we don't get
notifications).
