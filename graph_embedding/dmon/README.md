Graph Clustering with Graph Neural Networks
===============================

This is the implementation accompanying our paper, [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904).

Example usage
---
First, make sure you have the requirements `numpy`, `scipy`, `tensorflow 2`. You can always install them with `pip`: `pip3 install -r graph_embedding/dmon/requirements.txt`.

Then, to reproduce the paper results on the [Cora graph](https://relational.fit.cvut.cz/dataset/CORA), run

```python
# From google-research/
python3 -m graph_embedding.dmon.train --graph_path=graph_embedding/dmon/data/cora.npz --dropout_rate=0.5
```

Citing
---
If you find DMoN useful in your research, we ask that you cite the following paper:

> Tsitsulin, A., Palowitch, J., Perozzi, B., Mueller, E., (2020).
> Graph Clustering with Graph Neural Networks.
```
@inproceedings{tsitsulin2020clustering,
     author={Tsitsulin, Anton and Palowitch, John and Perozzi, Bryan and M\"uller, Emmanuel}
     title={Graph Clustering with Graph Neural Networks},
     year = {2020},
    }
```

Contact us
---
For questions or comments about the implementation, please contact [anton@tsitsul.in](mailto:anton@tsitsul.in) or [bperozzi@google.com](mailto:bperozzi@google.com).
