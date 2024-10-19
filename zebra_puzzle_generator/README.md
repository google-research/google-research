# Library for generating random Zebra puzzles and their solutions.

This document describes the functionality in this project.
We generate random zebra puzzles which are a type of logic puzzles. One aspect
that is important to us is to generate puzzles solvable by a human-like solver
which can only make certain types of deductions. This ensures that all puzzles
generated are solvable in polynomial time. Moreover, we generate the step-by-step
solution to the puzzles as well.

## From google-research/
To run the experiment, run
```
python -m zebra_puzzle_generator/main.py
```

## License
Zebra Puzzle Generator is licensed under the Apache License, Version 2.0.
This is not an officially supported Google product.
Contact nishanthd@google.com for questions about the repo.