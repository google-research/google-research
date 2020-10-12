# Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

A generalized cross-entropy loss function with bounded loss value per sample and a heavy-tail softmax probability function.

Bi-tempered loss generalizes (with a bias correction term):

- Zhang & Sabuncu. "Generalized cross entropy loss for training deep neural networks with noisy labels." In NeurIPS 2018.

which is recovered when 0.0 <= t1 <= 1.0 and t2 = 1.0. It also includes:

- Ding & Vishwanathan. "t-Logistic regression." In NeurIPS 2010.

for t1 = 1.0 and t2 >= 1.0.

Bi-tempered loss is equal to the cross entropy loss when t1 = t2 = 1.0. For 0.0 <= t1 < 1.0 and t2 > 1.0, bi-tempered loss provides a more robust alternative to the cross entropy loss for handling label noise and outliers.




Source: https://papers.nips.cc/paper/9638-robust-bi-tempered-logistic-loss-based-on-bregman-divergences.pdf

Google AI blog post: https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html


Reference:

@inproceedings{amid2019robust,
  title={Robust bi-tempered logistic loss based on bregman divergences},
  author={Amid, Ehsan and Warmuth, Manfred KK and Anil, Rohan and Koren, Tomer},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15013--15022},
  year={2019}
}
