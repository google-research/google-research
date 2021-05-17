This code runs an experiment with the "after kernel".  The after kernel is
like the neural tangent kernel, but after training.  For example, to compare
translation invariance of the after kernel to the neural tangent kernel after
training for 10 epochs on a binary classification problem with MNIST classes
3 and 8, run

```
python -m run_experiment.py --dataset=mnist \
                            --plus_class=8 \
                            --minus_class=3 \
                            --num_translations=1000 \
                            --num_epochs=10
```

This code depends on tensorflow-addons.  As of this writing, tensorflow-addons
is incompatible with the most recent version of python.  I worked around this 
by using pyenv to install an earlier version of python before installing 
tensorflow-addons.

