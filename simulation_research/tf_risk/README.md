# TF-RISK

Tf-risk is a library aimed at leveraging tensorflow and TPUs for risk
assessment, optimization and dynamic programming for stochastic systems such as
financial markets.

For now, tf-risk is mostly tailored to Financial Monte Carlo and comprises
different modules to price financial derivatives and to compute risk metrics.

## Modules:

### dynamics.py

Implements multiple discretization schemes for Geometric Brownian in 1d and Nd.

### monte_carlo_manager.py

Provides a suite of tools to run Monte Carlo methods to convergence.

### risk_metrics.py

Provides elementary functions to estimate Value-At-Risk and
Conditional-Value-at-Risk.

#### payoffs.py

Library of payoffs for financial derivatives (e.g., calls, puts, exotics).

### controllers.py

Provides primitives to hedge a European option based on an estimator for its
"delta".

### util.py

Entails different commonly used functions such as the analytical price for a
European Call under the Black-Scholes and Merton model.
