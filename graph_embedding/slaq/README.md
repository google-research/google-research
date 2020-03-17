Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs
===============================

This is the implementation accompanying our WWW'2020 paper, Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs.

Example usage:
---
First, make sure you have the requirements `numpy` and `scipy`. You can always install the with `pip`: `pip3 install -r graph_embedding/slaq/requirements.txt`.

Then, to try the example code on the [Karate club graph](https://en.wikipedia.org/wiki/Zachary%27s_karate_club), run

```python
# From google-research/
python3 -m graph_embedding.slaq.example
```

The output shows approximation errors with the parameters used in the paper.

Citing
---
If you find SLaQ useful in your research, we ask that you cite the following paper:

> Tsitsulin, A., Munkhoeva, M., Perozzi, B., (2020).
> Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs.
> In _The Web Conference_.
```
@inproceedings{tsitsulin2020,
     author={Tsitsulin, Anton and Munkhoeva, Marina and Perozzi, Bryan}
     title={Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs},
     booktitle = {The Web Conference},
     year = {2020},
    }
```

Contact us
---
For questions or comments about the implementation, please contact [anton@tsitsul.in](mailto:anton@tsitsul.in) or [bperozzi@google.com](mailto:bperozzi@google.com).
