.. _likelihood:

===================
Likelihood function 
===================


Having defined a PDF model, 
:math:`\mathbf{f}: \boldsymbol{\theta} \in \mathbb{R}^{N_{\rm params}} \to \mathbb{R}^{N_{\rm fl}\times N_{\rm x}}` 
(see the practical example in :ref:`Les Houches Tutorial <in_les_houches>`)
Colibri allows us to sample the posterior distribution of its parameters
:math:`\boldsymbol{\theta}` given a prior distribution :math:`\pi(\boldsymbol{\theta})`
and a likelihood function, :math:`\mathcal{l}(\mathbf{D} | \boldsymbol{\theta})`.

In this section, we will discuss the form of the likelihood function and its
implementation in Colibri. Note that a complementary discussion of the likelihood
function can be found in the
`NNPDF documentation <https://docs.nnpdf.science/figuresofmerit/index.html>`_

In general, it is more convenient to work in terms of the *log-likelihood*,
:math:`\mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) = \log (\mathcal{l}(\mathbf{D} | \boldsymbol{\theta}))`.
The basic form of the log-likelihood function used during sampling is proportional to a :math:`\chi^2`
function:

.. math::
    :label: eq:likelihood

     \mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) \propto -\chi^2,

where :math:`\chi^2` is defined as:

.. math::
    :label: eq:chi2

    \chi^2 = (\mathbf{D} - FK[\mathbf{f}(\boldsymbol{\theta})])^T C_{\rm t_0}^{-1} (\mathbf{D} - FK[\mathbf{f}(\boldsymbol{\theta})]),


and :math:`\mathbf{D}` is the data vector, :math:`FK[\mathbf{f}(\boldsymbol{\theta})]`
is the forward model (described in detail in :ref:`this section <theory_predictions>`)
and :math:`C_{\rm t_0}` is the :math:`t_0` covariance matrix used to avoid the d'Agostini
bias when the data has multiplicative uncertainties, as described in the
`NNPDF documentation <https://docs.nnpdf.science/figuresofmerit/index.html>`_.

During a fit, it is also possible to impose positivity and integrability constraints on
the PDFs. This is achieved similarly to how it is done in the NNPDF framework, by adding
extra Lagrange penalty terms to the likelihood function.

Positivity Constraints
^^^^^^^^^^^^^^^^^^^^^^
Positivity constraints are implemented in Colibri by adding the following penalty term
to the likelihood function:

.. math::
    \begin{align}
    \label{eq:pos_constraint}
    \mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) \to \mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) + \Lambda_{\rm pos} \sum_{k}\sum_{i} Elu_{\alpha}(-\tilde{f}_k(x_i,Q^2)) ,
    \end{align}

where by defauly :math:`Q^2 = 5 \, \mathrm{GeV^2}` and the :math:`n_i` values :math:`x_i`
given by 10 points logarithmically spaced between :math:`5 \cdot 10^{-7}`  and
:math:`10^{-1}` and 10 points linearly spaced between 0.1 and 0.9.

For a detailed description on how to include the positivity constraints in the likelihood
function during a fit see :ref:`this tutorial <running_positivity>`.

Integrability Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^
Integrability constraints are implemented in Colibri by adding the following penalty term
to the likelihood function:

.. math::
    \begin{align}
    \label{eq:int_constraint}
    \mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) \to \mathcal{L}(\mathbf{D} | \boldsymbol{\theta}) + \Lambda_{\rm int} \sum_{k}\sum_{i} \bigg[ x_i f_k(x_i, Q_0^2) \bigg]^2 ,
    \end{align}

where :math:`Q_0` is the parametrisation scale and the points :math:`x_i` run over a set
of values in the small :math:`x` region of the grid of the FK-table; in practice one
often only takes the smallest :math:`x` value of the grid to enforce the condition.
