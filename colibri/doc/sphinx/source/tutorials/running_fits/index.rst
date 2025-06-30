.. _in_running_fits:

============
Running Fits
============

Colibri supports three inference methods for fitting models:

1. **Analytical Fit**:
   Computes the posterior distribution of model parameters by solving the linear regression analytically.

2. **Bayesian Fit**:
   Employs Markov Chain Monte Carlo (MCMC) sampling to explore the posterior distribution.

3. **Monte Carlo Replica Fit**:
   Uses a parametric-bootstrap approach to approximate the posterior distribution via repeated resampling.

In the sections that follow, we'll explore the use cases and workflows for each method.


.. toctree::
   :maxdepth: 1

   ./running_analytic

   ./bayesian/index

   ./running_mc_replica
