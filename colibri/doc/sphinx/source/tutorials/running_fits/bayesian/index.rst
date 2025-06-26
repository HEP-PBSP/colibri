.. _running_bayesian:

=============
Bayesian Fits
=============

In this section, we discuss how to run a bayesian fit in Colibri.

In bayesian statistics, the parameters :math:`\theta` that describe the theory are
treated as random variables. They have a `prior probability density` (or `prior`),
:math:`P(\theta)`, which encodes what is known or assumed about the parameters 
prior to experimental observation.

The `posterior probability distribution`, i.e. the outcome of the fit, is determined
from `Bayes' theorem`:

.. math::
    P(\theta \mid \mathrm{data}) = \frac{P(\mathrm{data} \mid \theta) \times P(\theta)}{P(\mathrm{data})}

In a bayesian fit, the posterior distribution of the PDF model parameters is sampled
using a sampling method. Colibri currently supports bayesian sampling with the
following packages:

.. toctree::
    :maxdepth: 1

    ./ultranest

Following the links above will take you to a tutorial on how to run a
bayesian fit with the respective package.