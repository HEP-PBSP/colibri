.. _prior_distributions:

===================
Prior Distributions 
===================

To sample the posterior of the PDF model parameters :math:`\boldsymbol{\theta}`,
we must specify a prior distribution :math:`\pi(\boldsymbol{\theta})`.

Colibri's release supports two prior types:

- **Uniform priors**, with individually configurable bounds for each parameter.  
- **Gaussian priors**, defined by a mean vector and covariance matrix derived from the posterior samples of a previous fit.

The latter implements a Bayesian-update (posterior-factorisation): when you believe
the earlier fit is a valid approximation (or exactly Gaussian), and the two datasets
are uncorrelated, you can use the posterior of the first fit as a prior for the second fit.

.. _extra_parameters_prior:

Extra parameters of the forward map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Besides the PDF parameters, the forward map can have *extra* fit parameters,
e.g. a normalisation or SMEFT Wilson coefficients. Their prior is independent
of the PDF one:

- ``prior_settings`` sets the prior of the PDF parameters only. PDF model
  apps can replace it with their own prior (the ``bayesian_prior`` provider).
- ``extra_prior_settings`` sets the prior of the extra parameters. It has the
  same structure as ``prior_settings``, but only ``uniform_parameter_prior``
  is supported, with per-parameter ``bounds`` or a global ``min_val`` and
  ``max_val``. It is required as soon as the forward map has extra parameters.

.. code-block:: yaml

    prior_settings:                     # PDF parameters
      prior_distribution: uniform_parameter_prior
      prior_distribution_specs: {min_val: -1, max_val: 1}

    extra_prior_settings:               # extra parameters
      prior_distribution: uniform_parameter_prior
      prior_distribution_specs:
        bounds: {norm: [0.5, 1.5]}      # or: {min_val: ..., max_val: ...}

The fit uses the product of the two priors, so any PDF prior, including the
one of a PDF model app, can be combined with any extra parameters. The
analytic fit takes its uniform prior bounds from the same two places.

Likewise, for the fits that need a starting point (Monte Carlo and Hessian),
``param_initialiser_settings`` initialises the PDF parameters, and
``extra_param_initialiser_settings`` the extra ones, with the same options
(``zeros``, ``uniform``, ``normal``). The default for the extra parameters is
``zeros``:

.. code-block:: yaml

    extra_param_initialiser_settings:
      type: uniform
      min_val: 0.9
      max_val: 1.1

In the next section we outline the basic idea and domain of validity of this
posterior-factorisation approach. 

Bayesian update
^^^^^^^^^^^^^^^

Let us suppose that experimental data comprising :math:`N_{\rm data}` datapoints is 
distributed according to a multivariate normal distribution,

.. math::

    \begin{align}
    \mathbf{D} \sim N(FK(\boldsymbol{\theta}), \Sigma),
    \end{align}

where :math:`\Sigma` is an :math:`N_{\rm data}\times N_{\rm data}` experimental
covariance matrix.

Since in Bayesian statistics, :math:`\boldsymbol{\theta}` itself is assumed to
be a random variable, it has some associated prior probability density
:math:`\pi(\boldsymbol{\theta})` which, in what follows, we will assume to be a
“sufficiently” wide uniform probability density. Bayes' theorem then tells us
that after an observation :math:`\mathbf{D}_0` of :math:`\mathbf{D}`, the
probability density of :math:`\boldsymbol{\theta}` is:

.. math::
   :label: eq:bayes-theorem

   p(\boldsymbol{\theta}\mid\mathbf{D}_0)
   =
   \frac{\pi(\boldsymbol{\theta})\,L(\mathbf{D}_0\mid\boldsymbol{\theta})}{Z}
   =
   \frac{\pi(\boldsymbol{\theta})
         \exp\!\bigl(-\tfrac12 \|\mathbf{D}_0 - FK(\boldsymbol{\theta})\|^2_{\Sigma}\bigr)}
        {\displaystyle
         \int d\boldsymbol{\theta}\;\pi(\boldsymbol{\theta})
         \exp\!\bigl(-\tfrac12 \|\mathbf{D}_0 - FK(\boldsymbol{\theta})\|^2_{\Sigma}\bigr)} ,

where we wrote the generalised :math:`L_2` norm as:

.. math::
   \|\vec{x}\|^2_{\Sigma} = \vec{x}^T\,\Sigma^{-1}\,\vec{x}
   \quad\text{for}\quad\vec{x}\in\mathbb{R}^{N_{\rm data}}.

Now let's assume that :math:`\mathbf{D}_0 = (\mathbf{D}_1, \mathbf{D}_2)^T`,
with :math:`\mathbf{D}_1\in\mathbb{R}^{n_1}`, :math:`\mathbf{D}_2\in\mathbb{R}^{n_2}`
and :math:`n_1+n_2 = N_{\rm data}`, and that the two measurements are uncorrelated.  
That is, the covariance matrix factorises;

.. math::
   \Sigma = \Sigma_1 \oplus \Sigma_2,
   \quad
   \Sigma_1\in\mathbb{R}^{n_1\times n_1},
   \;\;
   \Sigma_2\in\mathbb{R}^{n_2\times n_2}.

In this case, since the likelihood :math:`L(\mathbf{D}_0\mid\boldsymbol{\theta})`
factorises (block-diagonal :math:`\Sigma`), we can write :eq:`eq:bayes-theorem` as:

.. math::
   :label: eq:bayes-uncorr

   p(\boldsymbol{\theta}\mid\mathbf{D}_0)
   =
   \frac{
     \pi(\boldsymbol{\theta})
     \exp\!\bigl(-\tfrac12\|\mathbf{D}_1-FK_1(\boldsymbol{\theta})\|^2_{\Sigma_1}\bigr)
     \exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   }
   {
     \displaystyle
     \int d\boldsymbol{\theta}\;\pi(\boldsymbol{\theta})
     \exp\!\bigl(-\tfrac12\|\mathbf{D}_1-FK_1(\boldsymbol{\theta})\|^2_{\Sigma_1}\bigr)
     \exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   } ,

where we write :math:`FK(\boldsymbol{\theta}) = (FK_1(\boldsymbol{\theta}), FK_2(\boldsymbol{\theta}))^T`.

Now, by noticing that the posterior for parameters :math:`\boldsymbol{\theta}` given
only :math:`\mathbf{D}_1` is:

.. math::
   :label: eq:pd1-post

   p_{\mathbf{D}_1}(\boldsymbol{\theta}\mid\mathbf{D}_1)
   =
   \frac{\pi(\boldsymbol{\theta})
         \exp\!\bigl(-\tfrac12\|\mathbf{D}_1-FK_1(\boldsymbol{\theta})\|^2_{\Sigma_1}\bigr)}
        {\displaystyle
         \int d\boldsymbol{\theta}\;\pi(\boldsymbol{\theta})
         \exp\!\bigl(-\tfrac12\|\mathbf{D}_1-FK_1(\boldsymbol{\theta})\|^2_{\Sigma_1}\bigr)}
   = \frac{\pi(\boldsymbol{\theta})
           \exp\!\bigl(-\tfrac12\|\mathbf{D}_1-FK_1(\boldsymbol{\theta})\|^2_{\Sigma_1}\bigr)}
          {Z_1} ,

we can rewrite :eq:`eq:bayes-uncorr` as:

.. math::

   p(\boldsymbol{\theta}\mid\mathbf{D}_0)
   =
   \frac{
     Z_1\,p_{\mathbf{D}_1}(\boldsymbol{\theta}\mid\mathbf{D}_1)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   }
   {
     \displaystyle
     \int d\boldsymbol{\theta}\;Z_1\,p_{\mathbf{D}_1}(\boldsymbol{\theta}\mid\mathbf{D}_1)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   }
   =
   \frac{
     p_{\mathbf{D}_1}(\boldsymbol{\theta}\mid\mathbf{D}_1)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   }
   {
     \displaystyle
     \int d\boldsymbol{\theta}\;p_{\mathbf{D}_1}(\boldsymbol{\theta}\mid\mathbf{D}_1)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_2-FK_2(\boldsymbol{\theta})\|^2_{\Sigma_2}\bigr)
   }.

Note that, if we have a measurement :math:`\mathbf{D}\sim N(FK(\boldsymbol{\theta}),\Sigma)`
with covariance matrix given by the direct sum of individual contributions,

.. math::
   \Sigma = \Sigma_1 \oplus \Sigma_2 \oplus \dots \oplus \Sigma_n,

we can apply this recursively. The posterior after all :math:`n` uncorrelated blocks is:

.. math::

   p(\boldsymbol{\theta}\mid\mathbf{D}_0)
   =
   \frac{
     \prod_{i=1}^{n-1} p_{\mathbf{D}_i}(\boldsymbol{\theta}\mid\mathbf{D}_i)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_n-FK_n(\boldsymbol{\theta})\|^2_{\Sigma_n}\bigr)
   }
   {
     \displaystyle
     \int d\boldsymbol{\theta}\;\prod_{i=1}^{n-1} p_{\mathbf{D}_i}(\boldsymbol{\theta}\mid\mathbf{D}_i)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_n-FK_n(\boldsymbol{\theta})\|^2_{\Sigma_n}\bigr)
   } ,

with each intermediate posterior for :math:`k>1` defined by:

.. math::
   p_{\mathbf{D}_k}(\boldsymbol{\theta}\mid\mathbf{D}_k)
   =
   \frac{
     \displaystyle
     \prod_{i=1}^{k-1} p_{\mathbf{D}_i}(\boldsymbol{\theta}\mid\mathbf{D}_i)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_k-FK_k(\boldsymbol{\theta})\|^2_{\Sigma_k}\bigr)
   }
   {
     \displaystyle
     \int d\boldsymbol{\theta}\;\prod_{i=1}^{k-1} p_{\mathbf{D}_i}(\boldsymbol{\theta}\mid\mathbf{D}_i)
     \,\exp\!\bigl(-\tfrac12\|\mathbf{D}_k-FK_k(\boldsymbol{\theta})\|^2_{\Sigma_k}\bigr)
   }.
