.. _in_running_fits:

============
Running Fits
============

Colibri supports three inference methods for PDF fitting:

1. **Bayesian Fit**:
   Employs Markov Chain Monte Carlo (MCMC) sampling to explore the posterior distribution.

2. **Monte Carlo Replica Fit**:
   Uses a parametric-bootstrap approach to approximate the posterior distribution via repeated resampling.

3. **Analytic Fit**:
   Computes the posterior distribution of model parameters by solving the linear regression analytically.

In the sections that follow, we'll explore the use cases and workflows for each method.


.. toctree::
   :maxdepth: 1

   ./bayesian/index

   ./running_mc_replica

   ./running_analytic

   ./running_hessian

   ./running_positivity
