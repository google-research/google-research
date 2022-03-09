This folder holds code for the following paper: "Predicting the utility of search spaces for black-box optimization: a simple, budget-aware approach" (https://arxiv.org/abs/2112.08250), accepted to AISTATS 2022.
The paper describes a general method to score and rank the search spaces over which we perform black-box optimization.

Currently, there is code that shows how to score search spaces, but, in order to run it, the user must supply various functions to sample from a GP (or other probabilistic model of the response surface). We plan to release example implementations of the necessary functions in the future.
