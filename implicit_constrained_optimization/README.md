# Implicit Constrained Optimization
Code for Implicit Constrained Optimization (ICO) as proposed in [Implicit rate-constrained optimization of non-decomposable objectives](https://arxiv.org/abs/2107.10960) by Abhishek Kumar, Harikrishna Narasimhan, and Andrew Cotter:
```
@inproceedings{kumar2021implicit,
  title={Implicit rate-constrained optimization of non-decomposable objectives},
  author={Kumar, Abhishek and Narasimhan, Harikrishna and Cotter, Andrew},
  booktitle={International Conference on Machine Learning},
  pages={5861--5871},
  year={2021},
  organization={PMLR}
}
```

The paper proposes a method for optimizing non-decomposable rate metrics with a certain thresholded form, while constraining another metric of interest. Such problems include optimizing the false negative rate (FNR) at a fixed false positive rate (FPR), optimizing precision at a fixed recall, optimizing the area under the precision-recall (AUC-PR) or ROC curves (AUC-ROC), etc. The key idea in the method is to formulate a rate-constrained optimization that expresses the threshold parameter (that acts on the model predictions to classify positive vs negative class) as a function of the model parameters via the Implicit Function theorem.

## Example Usage

### Optimizing FNR at a fixed FPR

To optimize for false negative rate (FNR) at a fixed false positive rate (FPR) for a CelebA attribute (binary classification problem) using the proposed method, run

```
python -m implicit_constrained_optimization.train_celeba_fnr_at_fpr --logtostderr --method ico \
  --attr $ATTR --sigmoid_temp $SIG_TEMP --target_fpr $TARGET_FPR
```
For example, to reproduce the results in the paper for `Black_Hair` attribute at FPR of 1\%, run
```
python -m implicit_constrained_optimization.train_celeba_fnr_at_fpr --logtostderr --method ico \
  --attr Black_Hair --sigmoid_temp 0.001 --target_fpr 0.01
```
Use `--method ce` for training and evaluating the model with cross-entropy loss.


### Optimizing (partial) area under the ROC curve

To optimize for partial area under the ROC curve in the FPR range `[0,$FPR_HIGH]` for a CelebA attribute (binary classification problem) using the proposed method, run

```
python -m implicit_constrained_optimization.train_celeba_aucroc_joint --logtostderr --method ico \
  --attr "[$ATTR]" --sigmoid_temp $SIG_TEMP --fpr_high $FPR_HIGH
```
For training a joint model over multiple attributes, use `--attr "[$ATTR1,$ATTR2,...]"`.

For example, to reproduce the results in the paper for `High_Cheekbones` attribute in the FPR range `[0,0.01]`, run
```
python -m implicit_constrained_optimization.train_celeba_aucroc_joint --logtostderr --method ico \
  --attr "['High_Cheekbones']" --sigmoid_temp 1. --fpr_high 0.01
```
Use `--method ce` or `--method pairwise` for training and evaluating the model with cross-entropy loss or with pairwise proxy loss for partial AUC, respectively.
