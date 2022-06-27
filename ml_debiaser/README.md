# Near-Optimal Debiasing Algorithms

[*Near-optimal algorithm for debiasing machine learning models*](https://arxiv.org/abs/2106.12887).
Ibrahim Alabdulmohsin and Mario Lucic, NeurIPS 2021.

*A Reduction to Binary Approach for Debiasing Multiclass Datasets*
Ibrahim Alabdulmohsin, Jessica Schrouff, and Oluwasanmi Koyejo, 2022 (under reivew)

This is the directory for algorithms that we develop to debias machine learing
models while treating them as black-box classifiers. We develop a near-optimal
post-processing method for binary classifiers and a pre-processing method for
multiclass datasets. The multiclass method does not require accesss to the
sensitive attribute at test time and can handle an arbitrary number of classes.

Both algorithms return solutions that satisfy the bias requirement on the
training data. The guarantees on test data hold with a high probability provided
that certain conditions are met (e.g. sufficient sample size per group). Please
check out the papers to understand the conditions required for bias mitigation
to generalize to test data with a high probability.

## Running the code

### 1. Setting up the Python environment
The code has been tested with Python 3.7.10 and Numpy 1.21.5.

### 2. Debiasing binary classifiers via post-processing.
Here is a walk-through example.

Import the necessary packages.
```
import numpy as np
import sklearn.ensemble
from ml_debiaser import randomized_threshold
from ml_debiaser import reduce_to_binary
```

Fetch the data (synthetic in the example below). The data should have three
splits: (1) *train*, (2) *hold-out*, and (3) *test*. The *train* split is used
to train the original classifier. The *hold-out* split is used to train the
debiaser. The *test* split is used to evaluate metrics (e.g. bias and accuracy).

```
num_examples_train = 10000
num_examples_val = 1000
num_examples_test = 1000

num_features = 100
num_classes = 2
num_groups = 10

x_train = np.random.randn(num_examples_train, num_features)
y_train = np.random.randint(0, num_classes, size=num_examples_train)
s_train = np.random.randint(0, num_groups, size=num_examples_train)

x_test = np.random.randn(num_examples_test, num_features)
y_test = np.random.randint(0, num_classes, size=num_examples_test)
s_test = np.random.randint(0, num_groups, size=num_examples_test)

x_val = np.random.randn(num_examples_val, num_features)
y_val = np.random.randint(0, num_classes, size=num_examples_val)
s_val = np.random.randint(0, num_groups, size=num_examples_val)
```

Train the original classifier:
```
clf = sklearn.ensemble.RandomForestClassifier(max_depth=10)
clf.fit(x_train, y_train)
```

Train the debiasing rule. It is important to rescale the predictions to the
interval [-1, +1] before training the debiaser. We do not do the normalization
automatically to keep the sovler general (since it can be used to solve some
other optimization problems). Below, ```eps``` controls the level of bias.
```
# note that the debiaser should be trained on a fresh sample
y_pred = clf.predict(x_val)
y_pred = 2 * y_pred - 1
rto = randomized_threshold.RandomizedThreshold(gamma=0.05, eps=0)
rto.fit(y_pred, s_val, sgd_steps=10_000, full_gradient_epochs=1000)
```

You can test your code by applying the debiaser on the hold-out split that was
used to train it.

```
ydeb_val = rto.predict(y_pred, s_val)

mean_scores_before = [np.mean(y_pred[s_val == k]) for k in range(num_groups)]
mean_scores_after = [np.mean(ydeb_val[s_val == k]) for k in range(num_groups)]

print("DP before: ", max(mean_scores_before) - min(mean_scores_before))
print("DP after: ", max(mean_scores_after) - min(mean_scores_after))
```
Your bias should now equal ```eps```; e.g. the code above should generate
something like the following:
<center>
  DP before:  0.26624737945492666
  DP after:  0.003612479474548458
</center>


Finally, apply the debiaser on the test data:
```
y_pred_test = clf.predict(x_test)
y_pred_test = 2 * y_pred_test - 1
ydeb_test = rto.predict(y_pred_test, s_test)
print("DP = ", max(mean_scores) - min(mean_scores))
```

### 2. Debiasing Multi-class Datasets
If you have a multiclass dataset, you can debias it according to multi-class
demographic parity. First, fetch the data (synthetic in the example below):

```
num_examples_train = 10000
num_examples_test = 1000

num_features = 100
num_classes = 5
num_groups = 10

x_train = np.random.randn(num_examples_train, num_features)
s_train = np.random.randint(0, num_groups, size=num_examples_train)

y_train_1d = np.random.randint(0, num_classes, size=num_examples_train)
# convert to one_hot encoding
y_train = np.zeros((num_examples_train, num_classes))
y_train[np.arange(num_examples_train), y_train_1d] = 1.0
```

And apply the method:
```
r2b = reduce_to_binary.Reduce2Binary(num_classes=num_classes)
ydeb_train = r2b.fit(y_train, s_train, sgd_steps=10_000,
                     full_gradient_epochs=1000, max_admm_iter=100)
```
Then, you can train your classifier on the new label matrix ```ydeb_train```.
