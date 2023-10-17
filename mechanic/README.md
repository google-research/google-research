# Mechanic - Black Box Learning Rate Tuner Implementation

<p align="center"> Ashok Cutkosky, Aaron Defazio and Harsh Mehta </p>

We introduce a technique for tuning the learning rate scale factor of any base optimization algorithm and schedule automatically, which we call mechanic. Our method provides a practical realization of recent theoretical reductions for accomplishing a similar goal in online convex optimization. We rigorously evaluate mechanic on a range of large scale deep learning tasks with varying batch sizes, schedules, and base optimization algorithms. These experiments demonstrate that depending on the problem, mechanic either comes very close to, matches or even improves upon manual tuning of learning rates.

This directory contains the exact jax/optax based implementation used to produce results of the Mechanic paper.

Preprint: https://arxiv.org/pdf/2306.00144.pdf

```
@article{Cutkosky2023MechanicAL,
  title={Mechanic: A Learning Rate Tuner},
  author={Ashok Cutkosky and Aaron Defazio and Harsh Mehta},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.00144}
}
```

## How to use?
Change from

```
    lr = lr_schedule(base_learning_rate=1e-4, step=step)
    optimizer = optax.adamw(learning_rate=lr)
```

To

```
    lr = lr_schedule(base_learning_rate=1.0, step=step)
    adamw = optax.adamw(learning_rate=lr)
    optimizer = mechanic.mechanize(adamw)
```