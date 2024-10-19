ELO-based pairwise comparison system with rater modeling
===

This folder contains a program that will compute ELO scores
from a set of pairwise comparisons of some methods.

The comparisons are expected to be provided as a `csv` file,
with (at least) the following columns:

 - `methodA`: the first method being compared
 - `methodB`: the second method being compared
 - `isGolden`: `1` if the question has a known correct answer
 - `modelProbability`: `1` if `isGolden` and the given answer matches the
   correct answer
 - `answerValue`: `A` if `methodA` was preferred, `B` if `methodB` was
   preferred, `draw` if neither was.
 - `answerer`: a string uniquely identifying the rater that answered the
   question.

The script will output the ELO scores of each method (with 99% credible
intervals), followed by the inferred probability of each rater of answering
questions randomly, followed by the expected information to be gained by
asking each specific question to raters next.
