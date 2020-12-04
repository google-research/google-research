# Notebooks

This directory contains two Colab notebooks describing how to use the GFSA layer.

- ["Interactive demo: Learning static analyses with the GFSA layer"][notebook_demo] (`demo_learning_static_analyses.ipynb`): This notebook walks through the process of generating random Python programs, encoding them into JAX arrays, training the GFSA layer from scratch, and visualizing the results.
- ["How to use the GFSA layer for new tasks"][notebook_new_task_guide] (`guide_for_new_tasks.ipynb`): This notebook describes the necessary steps for using the GFSA layer with a different type of graph domain, including converting the graph into an MDP and specifying the set of possible actions and observations at each state.

[notebook_demo]: https://colab.research.google.com/github/google-research/google-research/blob/master/gfsa/notebooks/demo_learning_static_analyses.ipynb
[notebook_new_task_guide]: https://colab.research.google.com/github/google-research/google-research/blob/master/gfsa/notebooks/guide_for_new_tasks.ipynb
