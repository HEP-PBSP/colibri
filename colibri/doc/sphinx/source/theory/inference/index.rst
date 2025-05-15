.. _inference:

=================
Inference Methods
=================

In its release version Colibri supports three types of inference methods:

- **Analytic fit**: the PDF model posteriors mean and covariance are determined as analytic solution to a linear regression problem.

- **Bayesian inference**: the posterior distribution of the PDF model parameters is sampled using a Bayesian sampling method. 

- **Monte Carlo replica method**: the posterior distribution of the PDF model parameters is approximated with the Monte Carlo Replica Method (:cite:`Costantini:2024wby`).


In the following sections we will discuss the three inference methods in detail.


Analytic fits
^^^^^^^^^^^^^
The analytic fit method is only applicable when the PDF model is linear in the parameters and 
the forward modeling is linear in the PDF.
Moreover, when using the analytic fit method, it is not possible to include non-linear constraints
such as positivity and integrability constraints.

.. note::
   Albeit not allowing for realistic PDF fits, the analytical fit method can be used to fit linear
   DIS data with no constraints and use the resulting Gaussian posterior as a prior for a realistic
   fit on an uncorrelated dataset as described in :ref:`prior distribution <bayesian_prior>`. 
   In general, this has the advantage of being computationally more efficient.

To illustrate the analytical method, let us assume a likelihood of the kind

.. math::
   :label: eq:likelihood-general

   p(D \mid \theta)
   = \frac{1}{(2\pi)^{N/2}\,\lvert\Sigma\rvert^{1/2}}
     \exp\!\Bigl(-\tfrac12\,(D - f(\theta))^T\,\Sigma^{-1}\,(D - f(\theta))\Bigr)\,,

where :math:`D` are the central values of the measured data and :math:`\Sigma` the covariance matrix.  If :math:`f(\theta)` is a linear model in :math:`\theta`,

.. math::
   :label: eq:linear-model

   f(\theta) = W\,\theta\,,

then the likelihood is Gaussian in the model parameters :math:`\theta` and can be rewritten as

.. math::
   :label: eq:likelihood-factorised

   \begin{aligned}
     p(D \mid \theta)
     &= \frac{(2\pi)^{N_{\rm params}/2}\,\bigl\lvert(W^T\Sigma^{-1}W)^{-1}\bigr\rvert^{1/2}}
            {(2\pi)^{N_{\rm dat}/2}\,\lvert\Sigma\rvert^{1/2}}
        \,\exp\!\Bigl(-\tfrac12\,(D - \hat{D})^T\,\Sigma^{-1}\,(D - \hat{D})\Bigr)\\
     &\quad\times
        \frac{\exp\!\Bigl(-\tfrac12\,
          (\theta - \hat{\theta})^T\,W^T\Sigma^{-1}W\,(\theta - \hat{\theta})
        \Bigr)}
        {(2\pi)^{N_{\rm params}/2}\,\bigl\lvert(W^T\Sigma^{-1}W)^{-1}\bigr\rvert^{1/2}}\\
     &= (2\pi)^{N_{\rm params}/2}\,\bigl\lvert(W^T\Sigma^{-1}W)^{-1}\bigr\rvert^{1/2}
        \;p(D \mid \hat{\theta})\,p(\hat{\theta}\mid\theta)\,,
   \end{aligned}

where

.. math::
   :label: eq:mle

   \hat{\theta}
   = (W^T\,\Sigma^{-1}\,W)^{-1}\,W^T\,\Sigma^{-1}\,D,
   \quad
   \hat{D} = W\,\hat{\theta},

are the maximum likelihood estimate of the parameters and the corresponding model prediction, respectively.
Moreover, :math:`p(D | \hat{\theta})` is the likelihood of the data evaluated at the maximum likelihood estimate, and

.. math::
   :label: eq:posterior-conditional

   p(\hat{\theta}\mid\theta)
   = \frac{\exp\!\Bigl(-\tfrac12\,
      (\theta - \hat{\theta})^T\,W^T\Sigma^{-1}W\,(\theta - \hat{\theta})
     \Bigr)}
     {(2\pi)^{N_{\rm params}/2}\,\bigl\lvert(W^T\Sigma^{-1}W)^{-1}\bigr\rvert^{1/2}}.

If we assume a uniform prior for the parameters :math:`\theta`, i.e.

.. math::
   :label: eq:uniform-prior

   p(\theta_i)
   = \begin{cases}
       \tfrac{1}{b_i - a_i}, & \text{if } \theta_i \in [a_i, b_i],\\
       0,                    & \text{otherwise},
     \end{cases}

then the posterior distribution becomes

.. math::
   :label: eq:posterior-uniform

   \begin{aligned}
     p(\theta \mid D)
     &\propto p(D \mid \theta)\,p(\theta)\\
     &\propto p(D \mid \theta)
        \prod_{i=1}^{N_{\rm params}}
        \frac{\Theta(\theta_i - a_i)\,\Theta(b_i - \theta_i)}{b_i - a_i}\,.
   \end{aligned}


Bayesian inference
^^^^^^^^^^^^^^^^^^

In the most general setting, that is for any type of PDF and forward model, it is recommended to use the 
Bayesian inference method which is based on a nested sampling implementation given by the 
`UltraNest <https://johannesbuchner.github.io/UltraNest/index.html>`_ package.

A tutorial on how to perform a Bayesian fit using nested sampling can be found in the
(TODOL: add link to the tutorial).


Gradient based methods
^^^^^^^^^^^^^^^^^^^^^^

Colibri supports the use of gradient-based methods, trough the `jax <https://docs.jax.dev/en/latest/quickstart.html>`_ and 
`optax <https://optax.readthedocs.io/en/latest/>`_ libraries, for the inference of the PDF model parameters.

A tutorial can be found here (TODO in tutorials).

A gradient-based method used to also perform uncertainty quantification and that can be found in colibri is the
Monte Carlo replica method.
This method consists in determining a set of fit outcomes to approximate the posterior probability 
distribution of the PDF model given a set of experimental input data. 
The input data are in turn represented as a MC sample of :math:`N_{\rm rep}` 
pseudodata replicas whose distribution (typically a multivariate normal) reproduces the covariance matrix 
of the experimental data. 
The fit outcomes are determined by minimising conditionally on a validation set the likelihood function
defined in :ref:`Likelihood function <likelihood>`.


.. note::

    As shown in the study :cite:`Costantini:2024wby`, the MC replica method is equivalent to Bayesian inference 
    only for linear PDF and forward models. In the presence of non-linearities the method shows a possible 
    bias and underestimation of the uncertainties. For this reason, we don't recommend using the MC replica method
    for non-linear PDF and forward models.
