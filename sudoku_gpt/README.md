# Library for training a GPT model on Sudoku puzzles.

This document describes the functionality and experiments in this project.
We train decoder-only Transformer language models on Sudoku puzzles and their
solutions and study how well the model learns to solve unseen Sudoku puzzles
at test time. 
In addition, we perform an interpretability analysis on the internal embeddings
of the model to study whether it has  developed an in-depth understanding of the
rules and strategies for solving Sudoku.

## From google-research/
To run the experiment, run
```
python -m sudoku_gpt/main.py
```

## License
Sudoku-GPT is licensed under the Apache License, Version 2.0.
This is not an officially supported Google product.
Contact nishanthd@google.com for questions about the repo.