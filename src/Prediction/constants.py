"""Monte Carlo simulation constants for the prediction system."""
import numpy as np

# Number of Monte Carlo simulation runs per prediction
# More runs = more stable confidence intervals, but slower
MONTE_CARLO_RUNS = 1000

# Reproducible random number generator for Monte Carlo sampling
# Using default_rng (modern numpy) instead of legacy np.random.seed
RNG = np.random.default_rng(42)
