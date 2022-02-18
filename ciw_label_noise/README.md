# Constrained Instance and Class reWeighting (CIW)
Code for Constrained Instance and Class reWeighting (CIW / CICW) for learning with noisy labels as proposed in [Constrained Instance and Class Reweighting for Robust Learning under Label Noise
](https://arxiv.org/abs/2111.05428) by Abhishek Kumar and Ehsan Amid:
```
@article{kumar2021constrained,
  title={Constrained Instance and Class Reweighting for Robust Learning under Label Noise},
  author={Kumar, Abhishek and Amid, Ehsan},
  journal={arXiv preprint arXiv:2111.05428},
  year={2021}
}
```
The performance of deep neural nets is largely dependent on the quality of the training data and often degrades in the presence of noise. We propose a principled approach for tackling label noise with the aim of assigning importance weights to individual instances and class labels. Our method works by formulating a class of constrained optimization problems that yield simple closed form updates for these importance weights. The proposed optimization problems are solved per mini-batch which obviates the need of storing and updating the weights over the full dataset. Our optimization framework also provides a theoretical perspective on existing label smoothing heuristics for addressing label noise (such as label bootstrapping [1]). We also propose the use of our instance weights to do mixup [2] which results in further performance gains in the presence of label noise.

## Example Usage

Training CIW on CIFAR-10 with 40\% symmetric label noise:

```
python -m ciw_label_noise.train --logtostderr --dataset cifar10 --num_classes 10 \
  --noise_type random_flip --noisy_frac 0.4 --loss dcl --lambda_hyp 0.1 
```

Training CICW with Mixup (CICW-M) on CIFAR-10 with 40\% symmetric label noise:
```
python -m ciw_label_noise.train --logtostderr --dataset cifar10 --num_classes 10 \ 
  --noise_type random_flip --noisy_frac 0.4 --loss dcl --lambda_hyp 2.5 --div_type_cls kl \
  --gamma 0.01 --dcl_w_mixup 1 --mixup_type sample_weight_w
```
Use `--loss ce` for training and evaluating the model with cross-entropy loss.

Train CIW with Mixup (CIW-M) on CIFAR-10 with 40\% asymmetric label noise:
```
python -m ciw_label_noise.train --logtostderr --dataset cifar10 --num_classes 10 \
  --noise_type random_flip_asym --noisy_frac 0.4 --loss dcl --lambda_hyp 10 --mixup_type \
  label_smoothing_w --mixup_alpha 0.2
```

Use `--dataset cifar100 --num_classes 100` for training on CIFAR. The hyperparmeters to reproduce the results of the paper are reported in the Appendix of the paper.

## References
[1] Reed, S., Lee, H., Anguelov, D., Szegedy, C., Erhan, D., and Rabinovich, A. Training deep neural networks on noisy labels with bootstrapping. In ICLR, 2015.
[2] Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D.  mixup: Beyond empirical risk minimization. In ICLR, 2018.
