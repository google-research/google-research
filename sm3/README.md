# Memory-Efficient Adaptive Optimization

Source: https://arxiv.org/abs/1901.11150

Adaptive gradient-based optimizers such as AdaGrad and Adam are among the
defacto methods of choice in modern machine learning.These methods tune the learning rate for each parameter during the optimization process using cumulative second-order statistics. These methods provide superior convergence properties and are very attractive in large scale applications due to their moderate time and space requirements which are linear in the number of parameters.


However, the recent advances in natural language processing such as BERT and GPT2 show that models with 10<sup>8</sup> to 10<sup>10</sup> parameters, trained with adaptive optimization methods, achieve state-of-the-art results. In such cases, the memory overhead of the optimizer can restrict the size of the model that can be used as well as the batch size, both of which can have a dramatic effect on the quality of the final model.


Here we construct a new adaptive optimization method that retains most of the benefits of standard per-parameter adaptivity while significantly reducing memory overhead.


We observe that in standard neural networks that certain entries of the stochastic gradients have (on average) similar values, and exhibit what we refer to as an activation pattern. For example, in gradients of embedding layers of deep networks, an entire row (or column) is either zero or non-zero. Similarly, in intermediate layers we often observe that gradients associated with the same unit are of similar order of magnitude. In these cases, a similar phenomenon is observed in the second-order statistics maintained by adaptive methods. With this key observation, to reduce the memory overhead of the optimizer our method takes in a cover set of the parameters. Cover sets are typically selected in practice such that parameters in each of the sets have second order statistics of similar magnitude. Our method is general enough that it can easily be extended to arbitrary cover sets. For parameters of deep networks that are organized as a collection of tensors, we form a cover consisting of slices of codimension one for each tensor. Thus, for an m x n parameter matrix, the cover consists of rows and columns of the matrix. The memory requirements therefore drop from mxn to merely m+n. For a parameter tensor of rank p, with dimensions n<sub>1</sub>  ...   n<sub>p</sub>, the reduction in memory consumption is even more pronounced, dropping from product of all the dimensions to the sum of all dimensions. This virtually eliminates the memory overhead associated with maintaining the adaptive learning rates!

Another practical aspect worthy of note is that our method does not require an external hand engineered learning rate decay schedule but instead relies on the per parameter adaptivity that is natural to its update rule which makes it easier to tune. We provide details in the supplementary section of the paper.

## Advice on using SM3 on your model

### Learning rate warm-up:

```python
learning_rate = lr_constant * tf.minimum(1.0, (warm_up_step / global_step) ** p)
```

* p = 1, linear ramp up of learning rate.
* p = 2, quadratic ramp up of learning rate [preferred].

We typically set `warm_up_step` as 5% of overall steps. Initially, the norm of the preconditioned gradient is much larger than norm of the weights. Learning rate warmup allows us to heuristically fix this scale mismatch.

### Learning rate decay:

We make use accumulated gradient squares for the decay. This means that each coordinate gets its own natural decay based on the scales of the gradients over time. Hence, users need not put in an external learning rate decay schedule. Moreover, we found in our experiments with translation and language models that this approach is superior to a hand-tuned learning rate decay schedules which is typically combined with exponential moving averages of the gradient squares.

Having said that if users want to add exponential moving averages instead of the standard accumulated gradient squares - It's easy to modify the optimizer implementation to switch to exponential moving averages.

For rank > 1:

|            from                     |                  to                 |
|-------------------------------------|-------------------------------------|
|  current_accumulator += grad * grad |  current_accumulator = beta * current_accumulator + (1-beta) * grad * grad |


For rank <= 1:


|            from                     |                  to                 |
|-------------------------------------|-------------------------------------|
|  current_accumulator = tf.assign_add(accumulator, grad * grad) |   current_accumulator = tf.assign(accumulator, beta * accumulator + (1-beta) * (grad * grad)) |


### Polyak averaging of parameters: 
It's useful to run [polyak averaging](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage) of the parameters. These parameters are then used in inference / serving. Using the averaged parameters instead of the last iterate typically improves the overall performance of the model.

An **alternative** to polyak averaging which does not make use of extra memory is to decay the learning rate from the constant to zero for the last 10% of the steps of your entiring training run, we term the phase a **cool-down** phase of the model. As training makes smaller and smaller steps the final iterate can be thought of as an average iterate.
