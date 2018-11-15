# Overview

This directory contains a Python/NumPy implementation of the Sufficient Input
Subsets (SIS) procedure for interpreting black-box functions [[1]](#references).
SIS was developed in the [Gifford Lab](http://cgs.csail.mit.edu/research/SIS/)
at MIT CSAIL.

SIS is a local explanation framework for interpreting black-box functions.
A sufficient input subset is a minimal set of input features whose observed
values alone suffice for the same decision to be reached, even with all other
input values missing. These subsets can be understood as rationales for a
model's decision-making.

Presuming the model's decision is based on *f(x)* exceeding a pre-specified
threshold, the goal of SIS is to find a sparse subset S of the input features
such that *f(x_s) >= threshold* (in other words, the prediction on just the
subset alone, with all other features masked, also exceeds the threshold).
Each SIS represents a justification for the decision by the model.

For a given input *x*, function *f*, and *threshold*, the SIS procedure
identifies a disjoint collection of such subsets (a "SIS-collection"), each of
which satisfies the above sufficiency criterion.

See the SIS paper [[1]](#references) for more information.


# Using SIS in Practice

SIS can be applied to any black-box function *f*, does not require access to
gradients, requires no additional training, and does not use any auxiliary
explanation model. However, applying the SIS procedure in practice requires some
consideration of how to choose an appropriate threshold and how to mask input
features. This section discusses some practical usage tips for running SIS.


### Choosing the Threshold

In classification settings, *f* can represent the predicted probability of a
class *C*, where *x* is considered to belong to class *C* if
*f(x) >= threshold*. In this case, the threshold may be chosen with
precision/recall considerations in mind (as discussed in [[1]](#references)).

For example, if *f* represents a probability, a threshold of 0.9 may be suitable
so that each SIS intuitively represents a feature pattern which *f* assigns to
the class with probability >= 0.9.

In other cases, the threshold may be chosen based on some known values relevant
to the task. For example, consider a regression task in which *f* maps an input
molecule into a solubility score (where larger score represents greater
solubility). There may be a chemically-meaningful value of the score that
corresponds to a boundary between soluble and insoluble. In such a setting, the
SIS would represent minimal feature subsets that the model needs to predict a
molecule to be soluble.

Without any such insights, it may be helpful to plot the predictive
distribution and select a threshold that selects a set of inputs for which the
model makes the most confident decisions.
For example, in the SIS paper [[1]](#references), the authors define the
threshold in a DNA-protein binding classification task as the 90th percentile of
predicted binding probabilities over all sequences.


#####  What happens to the SIS as the threshold varies?

In general, smaller threshold values result in a SIS-collection (for a given
input) that contains more disjoint SIS, each of which contains fewer elements.
As the threshold increases, the SIS-collection will tend to contain fewer SIS,
each of which is larger. Intuitively, each SIS becomes larger as additional
features are needed in order to make the model more confident in the decision.


##### How do I apply SIS to interpret strong negative predictions?

In some applications, small values of *f* may likewise correspond to confident
predictions. Consider a sentiment analysis task in which the model's output is
passed through sigmoid activation (so *0 <= f(x) <= 1*).
In this case, when *f(x)* is close to 1, the model predicts strong positive
sentiment, while *f(x)* close to 0 indicates strong predicted negative
sentiment.

Since the SIS procedure interprets decisions where *f(x) __>=__ threshold*,
how can we apply SIS in cases where we want to interpret
*f(x) __<=__ threshold*? All that is needed is to negate the values of *f* and
*threshold* so that the SIS procedure is applied to *-f(x)* with *-threshold*.
(Note that *f(x) <= threshold* is equivalent to *-f(x) >= -threshold*.)


### Masking

This implementation of SIS requires specification of the fully-masked input.
In many cases, it may be suitable to select a mask for a feature as the mean
value of that feature over the training data. Some options include:

* In natural language applications, if words in a sequence are represented by
  embeddings, a word may be masked using the mean embedding taken over the
  entire vocabulary (the authors favor this approach in the SIS paper
  [[1]](#references)).
  Other possible options in this domain include masking using an \<UNK\> token,
  replacement with an all-zeros representation (if words are represented as
  one-hots), or replacement with a trained mask embedding.
* In image tasks, a pixel may be masked by replacement with a mean pixel value
  taken over the training images or by replacement with an all-zero (black)
  value.
* In biological tasks (e.g. inputs are DNA or protein sequences), there are
  often standard IUPAC representations corresponding to unknown bases.
  In DNA sequences, "N" represents any nucleotide, while in protein
  sequences, "X" represents any amino acid. These values are typically taken as
  a uniform distribution over possible values (nucleotides or amino acids,
  respectively).
* In tasks where inputs contain fixed-length representations of mixed features,
  a mask value for a feature can be taken as the mean value of that feature
  across the training data. For categorical features, masks may be mean
  values, all-zero values, or uniform values across the categories.

Whatever mask is chosen, it is a good idea to check that:

* The prediction on the fully-masked input is well below the threshold.
* The prediction on the fully-masked input is uninformative, to suggest that
  the mask is destroying information (e.g. prediction on the fully-masked input
  tends to the mean or prior over the training data).


# Usage Examples

To compute the SIS-collection for an input using this library, the
`sis_collection` function takes as input a function `f_batch` (in this
implementation, the function takes a batch of examples and returns a batch
of scalars), `threshold` (scalar), `initial_input` (NumPy array),
`fully_masked_input` (NumPy array), and (optionally) an `initial_mask`.
See the docstring of `sis_collection` for details of each parameter.

Usage examples for 1-D and 2-D inputs are given below. In both cases, suppose
the function *f(x)* computes the sum of the components of *x* and the threshold
is 1.

```python
# Function that computes the sum for each array in the batch.
# For each element x of the batch, f(x) = sum(x) (sums over all axes, giving a
# scalar regardless of input dimensionality).
F_SUM_BATCH = lambda batch: np.array([np.sum(arr) for arr in batch])

threshold = 1.0
```


Other Notes:

* `f_batch` should handle any necessary batching for inference. `f_batch` may be
  called with a large number of arrays.
* Masks are defined as boolean masks, where True indicates the presence of a
  feature, and False indicates the feature is masked.
* Masks that are not the same shape as the input (but which are broadcastable
  to the shape of the input) are supported. See the 2-D (Sequence) usage example
  below, as well as the docstring of
  `make_empty_boolean_mask_broadcast_over_axis` for details.
* The following examples are for 1-D and 2-D inputs, but this library can handle
  higher-dimensional inputs as well.
* If an application warrants masking a position using a more complex approach
  (e.g. hot-deck imputation using repeated sampling from a marginal
  distribution), you can specify `fully_masked_input` in a way that allows
  `f_batch` to determine which positions are masked, and then perform any
  necessary replacement/sampling within `f_batch`.


#### 1-D Input

```python
initial_input = np.array([0.1, 0.5, 0.8, 0.1])
fully_masked_input = np.array([0.0, 0.0, 0.0, 0.0])

collection = sis.sis_collection(F_SUM_BATCH, threshold, initial_input,
                                fully_masked_input)
```

In this example, `collection` contains only a single SIS (with the elements
0.8 and 0.5). No secondary SIS exists because after 0.8 and 0.5 are masked
(here with zeros), the function on the remaining values only evaluates to 0.2
(less than the threshold of 1).

```python
>>> len(collection)
1
>>> collection[0]
SISResult(sis=array([[2], [1]]),
          ordering_over_entire_backselect=array([[0], [3], [1], [2]]),
          values_over_entire_backselect=array([1.4, 1.3, 0.8, 0. ]),
          mask=array([False,  True,  True, False]))
```

See the docstring for `SISResult` for more details on the returned object.
Next, we can produce a version of the input where all features not in the SIS
are masked:

```python
sis_masked_input = sis.produce_masked_inputs(initial_input, fully_masked_input,
                                             [collection[0].mask])

>>> sis_masked_input
array([[0. , 0.5, 0.8, 0. ]])
```


#### 2-D (Sequence) Input

Suppose that `initial_input` here is 2-dimensional of shape
`(feature encoding, # timesteps)`, where each column of the input is a vector
encoding of the value at that timestep (for example, each column could be a word
embedding in a natural language task):

```python
initial_input = np.array([[0.1, 0.5, 0.4], [0.2, 0.6, 0.3]])
>>> initial_input
array([[0.1, 0.5, 0.4],
       [0.2, 0.6, 0.3]])

fully_masked_input = np.zeros(initial_input.shape)

initial_mask = sis.make_empty_boolean_mask_broadcast_over_axis(
    initial_input.shape, 0)
>>> initial_mask
array([[ True,  True,  True]])  # Note that shape of mask is (1, 3).

collection = sis.sis_collection(F_SUM_BATCH, threshold, initial_input,
                                fully_masked_input,
                                initial_mask=initial_mask)
```

In this example, we specify a value for `initial_mask` so SIS masks entire
columns at a time by broadcasting the mask over the columns of the input (rather
than considering each position independently). Here, there are two SIS in the
SIS-collection:

```python
>>> collection
[SISResult(sis=array([[0, 1]]),
           ordering_over_entire_backselect=array([[0, 0], [0, 2], [0, 1]]),
           values_over_entire_backselect=array([1.8, 1.1, 0. ]),
           mask=array([[False,  True, False]])),
 SISResult(sis=array([[0, 2], [0, 0]]),
           ordering_over_entire_backselect=array([[0, 0], [0, 2]]),
           values_over_entire_backselect=array([0.7, 0. ]),
           mask=array([[ True, False,  True]]))]
```

In this example, the first SIS contains the 2nd column of the input, while
the second SIS contains columns 1 and 3. (In this case, because of how the
initial_mask was specified, SIS operates __not__ on individual entries of the
2-D array, but instead considers __each__ column in the array as a feature.)
We can produce the corresponding masked version of each SIS (masking all non-SIS
elements) by:

```python
sis_masked_inputs = sis.produce_masked_inputs(initial_input, fully_masked_input,
                                              [sr.mask for sr in collection])

>>> sis_masked_inputs
array([[[0. , 0.5, 0. ],
        [0. , 0.6, 0. ]],

       [[0.1, 0. , 0.4],
        [0.2, 0. , 0.3]]])
```


# Applications of SIS

These SIS can be used in a number of downstream applications, including
clustering the SIS produced over a large number of examples to understand global
model behavior or to contrast features learned by two models. See Section 4.4 in
the SIS paper [[1]](#references) for details and examples.


# References

[1] Carter, B., Mueller, J., Jain, S., & Gifford, D. (2018). What made you do
    this? Understanding black-box decisions with sufficient input subsets.
    arXiv preprint arXiv:1810.03805. https://arxiv.org/abs/1810.03805
