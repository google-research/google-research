This repository contains code supporting "Reach Implementation Best Practices in
the Privacy Sandbox Private Aggregation API" white paper.

**Note**: This is not an officially supported Google product.

## Installation

You can install the dependencies using: `pip install -r requirements.txt`

## Running the Scripts

1.  To generate the error estimates for cumulative queries, run `python -m
    reach_whitepaper.compute_cumulative_error`
2.  To generate the error estimates of the direct method for fixed windows, run
    `python -m reach_whitepaper.compute_direct_error`
3.  To generate the error estimates of the sketching method for fixed windows,
    run `python -m reach_whitepaper.compute_sketches_error`
4.  To generate the estimates of the errors due to difference between
    observations and true values run run `python -m
    reach_whitepaper.compute_observation_error`
