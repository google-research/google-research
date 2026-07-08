# JAX-RTM Examples

This directory contains examples demonstrating how to run the JAX-RTM simulator.

### Usage
Ensure you have installed the package and downloaded the required ice
properties database:

```bash
# Download the 155MB ice database (required)
python3 download_data.py

# Run the example
python3 examples/example_simulation.py
```

This will run a simulation over a sample 85x85 weather grid and save a
composed **Ash RGB** visualization as `examples/simulated_ash_rgb_85x85.png`.